# ShapeMamba-EM
------
This is the official repository for the paper [ShapeMamba-EM: Fine-Tuning Foundation Model with Local Shape Descriptors and Mamba Blocks for 3D EM Image Segmentation](https://papers.miccai.org/miccai-2024/paper/0151_paper.pdf).

![Overview of ShapeMamba-EM framework](asset/model.png?raw=true "Overview of ShapeMamba-EM framework")

## Requirement
#### Environmental Requirements

- Python=3.10
- torch==2.0.1
- cuda==11.7
- mamba-ssm

#### Installation
Clone this repository and then install the dependencies.
```sh
git clone git@github.com:EmmaSRH/ShapeMamba-EM.git
conda create -n shapemamba python=3.10
conda activate shapemamba
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
cd ShapeMamba-EM
pip install -r requirements.txt
```

## Training
Before start, Download the checkpoint from [ckpt section](https://github.com/uni-medical/SAM-Med3D#-checkpoint) and move the pth file into `SAM_Med3D/ckpt/` (We recommand to use `SAM-Med3D-turbo.pth`).
The dataset format is the same as [nnU-Net-style dataset](https://github.com/uni-medical/SAM-Med3Dhttps://github.com/uni-medical/SAM-Med3D?tab=readme-ov-file#1-prepare-your-training-data-from-nnu-net-style-dataset)
```sh
cd ShapeMamba-EM
python train.py --multi_gpu
```
We use 8 A800 GPUs to train our full model.

## Inference

```sh
python validation.py --seed 2023 -vp results/path -cp pth/path -tdp test/data/path --save_name save/name
```


## Acknowledgments
Our code is based on [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D), [SAMed](https://github.com/hitachinsk/SAMed), [FacT](https://github.com/JieShibo/PETL-ViT/tree/main/FacT), [mamba](https://github.com/state-spaces/mamba/tree/main), [lsd](https://github.com/funkelab/lsd/), and [Segment Anything](https://github.com/facebookresearch/segment-anything). We appreciate the authors for their great works. 


> If you find our survey useful for your research, please cite the following paper and give us a ⭐. We really do appreciate your sopport.

```
@inproceedings{shi2024shapemamba,
  title={ShapeMamba-EM: Fine-Tuning Foundation Model with Local Shape Descriptors and Mamba Blocks for 3D EM Image Segmentation},
  author={Shi, Ruohua and Pang, Qiufan and Ma, Lei and Duan, Lingyu and Huang, Tiejun and Jiang, Tingting},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={731--741},
  year={2024},
  organization={Springer}
}
```