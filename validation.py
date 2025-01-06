import os
join = os.path.join
import numpy as np
from glob import glob
import torch
from segment_anything.build_sam3D import sam_model_registry3D
from segment_anything.utils.transforms3D import ResizeLongestSide3D
from segment_anything import sam_model_registry
from tqdm import tqdm
import argparse
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk
import torchio as tio
import numpy as np
from collections import OrderedDict, defaultdict
import json
import pickle
from utils.data_loader import Dataset_Union_ALL_Val

parser = argparse.ArgumentParser()
parser.add_argument('-tdp', '--test_data_path', type=str, default='./data/validation')
parser.add_argument('-vp', '--vis_path', type=str, default='./visualization')
parser.add_argument('-cp', '--checkpoint_path', type=str, default='./ckpt/sam_med3d.pth')
parser.add_argument('--save_name', type=str, default='union_out_dice.py')

parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-mt', '--model_type', type=str, default='vit_b_ori')
parser.add_argument('-dt', '--data_type', type=str, default='Ts')

parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--split_num', type=int, default=1)
parser.add_argument('--ft2d', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=2023)

args = parser.parse_args()

SEED = args.seed
print("set seed as", SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.init()


def compute_iou(pred_mask, gt_semantic_seg):
    in_mask = np.logical_and(gt_semantic_seg, pred_mask)
    out_mask = np.logical_or(gt_semantic_seg, pred_mask)
    iou = np.sum(in_mask) / np.sum(out_mask)
    return iou

def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    if args.ft2d and ori_h < image_size and ori_w < image_size:
        top = (image_size - ori_h) // 2
        left = (image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad

def sam_decoder_inference(target_size,model, image_embeddings):
    with torch.no_grad():
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = model.prompt_encoder.get_dense_pe(),
        )

    masks = F.interpolate(low_res_masks, (target_size, target_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions

def finetune_model_predict3D(img3D, gt3D, sam_model_tune, device='cuda',  prev_masks=None):
    img3D = norm_transform(img3D.squeeze(dim=1)) # (N, C, W, H, D)
    img3D = img3D.unsqueeze(dim=1)


    if prev_masks is None:
        prev_masks = torch.zeros_like(gt3D).to(device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(args.crop_size//4,args.crop_size//4,args.crop_size//4))

    with torch.no_grad():
        image_embedding = sam_model_tune.image_encoder(img3D.to(device)) # (1, 384, 16, 16, 16)

    with torch.no_grad():
        low_res_masks, _ = sam_model_tune.mask_decoder(
            image_embeddings=image_embedding.to(device), # (B, 384, 64, 64, 64)
            image_pe=sam_model_tune.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
            )
        prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode='trilinear', align_corners=False)

        medsam_seg_prob = torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
        # convert prob to mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        pred = medsam_seg

        iou = round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4)
        dice = round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4)

    return pred, iou, dice

if __name__ == "__main__":    
    all_dataset_paths = glob(join(args.test_data_path, "*", "*"))
    all_dataset_paths = list(filter(os.path.isdir, all_dataset_paths))
    print("get", len(all_dataset_paths), "datasets")

    infer_transform = [
        tio.ToCanonical(),
        tio.CropOrPad(mask_name='label', target_shape=(args.crop_size,args.crop_size,args.crop_size)),
    ]

    test_dataset = Dataset_Union_ALL_Val(
        paths=all_dataset_paths, 
        mode="Val", 
        data_type=args.data_type, 
        transform=tio.Compose(infer_transform),
        threshold=0,
        split_num=args.split_num,
        split_idx=args.split_idx,
        pcc=False,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=None,
        batch_size=1, 
        shuffle=True
    )

    checkpoint_path = args.checkpoint_path

    device = args.device
    print("device:", device)

    sam_model_tune = sam_model_registry3D[args.model_type](checkpoint=None).to(device)
    if checkpoint_path is not None:
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict['model_state_dict']
        sam_model_tune.load_state_dict(state_dict)

    sam_trans = ResizeLongestSide3D(sam_model_tune.image_encoder.img_size)

    all_iou_list = []
    all_dice_list = []  

    out_dice = dict()
    out_dice_all = OrderedDict()

    for batch_data in tqdm(test_dataloader):
        image3D, gt3D, img_name = batch_data
        sz = image3D.size()
        if(sz[2]<args.crop_size or sz[3]<args.crop_size or sz[4]<args.crop_size):
            print("[ERROR] wrong size", sz, "for", img_name)
        modality = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_name[0]))))
        dataset = os.path.basename(os.path.dirname(os.path.dirname(img_name[0])))
        vis_root = os.path.join(os.path.dirname(__file__), args.vis_path, modality, dataset)
        pred_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{args.num_clicks-1}.nii.gz"))
        if(os.path.exists(pred_path)):
            iou_list, dice_list = [], []
            for iter in range(args.num_clicks):
                curr_pred_path = os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{iter}.nii.gz"))
                medsam_seg = sitk.GetArrayFromImage(sitk.ReadImage(curr_pred_path))
                iou_list.append(round(compute_iou(medsam_seg, gt3D[0][0].detach().cpu().numpy()), 4))
                dice_list.append(round(compute_dice(gt3D[0][0].detach().cpu().numpy().astype(np.uint8), medsam_seg), 4))
        else:
            norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
            seg_mask_list, iou, dice = finetune_model_predict3D(
                    image3D, gt3D, sam_model_tune, device=device,
                    prev_masks=None)

            os.makedirs(vis_root, exist_ok=True)
            print("save to", os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pred.nii.gz")))
            pt_path=os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", "_pt.pkl"))
            for idx, pred3D in enumerate(seg_mask_list):
                out = sitk.GetImageFromArray(pred3D)
                sitk.WriteImage(out, os.path.join(vis_root, os.path.basename(img_name[0]).replace(".nii.gz", f"_pred{idx}.nii.gz")))

        per_iou = max(iou_list)
        all_iou_list.append(per_iou)
        all_dice_list.append(max(dice_list))
        print(dice_list)
        out_dice[img_name] = max(dice_list)
        cur_dice_dict = OrderedDict()
        for i, dice in enumerate(dice_list):
            cur_dice_dict[f'{i}'] = dice
        out_dice_all[img_name[0]] = cur_dice_dict

    print('Mean IoU : ', sum(all_iou_list)/len(all_iou_list))
    print('Mean Dice: ', sum(all_dice_list)/len(all_dice_list))

    final_dice_dict = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ] = OrderedDict()
    for k, v in out_dice_all.items():
        organ = k.split('/')[-4]
        final_dice_dict[organ][k] = v

    if(args.split_num>1):
        args.save_name = args.save_name.replace('.py', f'_s{args.split_num}i{args.split_idx}.py')

    print("Save to", args.save_name)
    with open(args.save_name, 'w') as f:
        f.writelines(f'# mean dice: \t{np.mean(all_dice_list)}\n')
        f.writelines('dice_Ts = {')
        for k, v in out_dice.items():
            f.writelines(f'\'{str(k[0])}\': {v},\n')
        f.writelines('}')

    with open(args.save_name.replace('.py', '.json'), 'w') as f:
        json.dump(final_dice_dict, f, indent=4)

    print("Done")