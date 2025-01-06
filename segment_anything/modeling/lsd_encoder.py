import math
import numpy as np
import os
import sys
import torch

from funlib.learn.torch.models import UNet
from gunpowder import *
from gunpowder.ext import torch

class Convolve(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_fmaps,
            fmap_inc_factors,
            downsample_factors,
            kernel_size=(1, 1, 1)):

        super().__init__()

        self.model = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factors=fmap_inc_factors,
            downsample_factors=downsample_factors
        )

        conv = torch.nn.Conv3d

        self.conv_pass = torch.nn.Sequential(
                            conv(
                                num_fmaps,
                                out_channels,
                                kernel_size),
                            torch.nn.Sigmoid())

    def forward(self, x):

        y = self.model.forward(x)

        return self.conv_pass(y)