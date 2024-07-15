import torch
import celldetection as cd
import celldetection_scripts as cs
from skimage.data import coins
import pytest


def test_inference():
    accelerator = 'cpu'
    model = 'ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c'
    img = coins()
    print('Image:', img.shape)
    cs.cpn_inference(img, model, accelerator=accelerator)
