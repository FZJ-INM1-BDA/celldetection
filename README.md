# Cell Detection

[![Downloads](https://pepy.tech/badge/celldetection?l)](https://pepy.tech/project/celldetection)
[![Test](https://github.com/FZJ-INM1-BDA/celldetection/workflows/Test/badge.svg)](https://github.com/FZJ-INM1-BDA/celldetection/actions?query=workflow%3ATest)
[![PyPI](https://img.shields.io/pypi/v/celldetection?l)](https://pypi.org/project/celldetection/)
[![Documentation Status](https://readthedocs.org/projects/celldetection/badge/?version=latest)](https://celldetection.readthedocs.io/en/latest/?badge=latest)

## ⭐ Showcase

###### Nuclei of U2OS cells in a chemical screen

![bbbc039](https://raw.githubusercontent.com/FZJ-INM1-BDA/celldetection/main/assets/bbbc039-cpn-u22-demo.png "BBBC039 demo with CpnU22 - Find the dataset here: https://bbbc.broadinstitute.org/BBBC039")
*https://bbbc.broadinstitute.org/BBBC039 (CC0)*

###### P. vivax (malaria) infected human blood 

![bbbc041](https://raw.githubusercontent.com/FZJ-INM1-BDA/celldetection/main/assets/bbbc041-cpn-u22-demo.png "BBBC041 demo with CpnU22 - Find the dataset here: https://bbbc.broadinstitute.org/BBBC041")
*https://bbbc.broadinstitute.org/BBBC041 (CC BY-NC-SA 3.0)*

## 🛠 Install
Make sure you have [PyTorch](https://pytorch.org/get-started/locally/) installed.
### PyPI
```
pip install -U celldetection
```

### GitHub
```
pip install git+https://github.com/FZJ-INM1-BDA/celldetection.git
```
## 💡 How to train 
Here you can see some examples of how to train a detection model.
The examples already include toy data, so you can get started right away.
- [Train your own model](https://github.com/FZJ-INM1-BDA/celldetection/blob/main/demos/demo-binary.ipynb)
- [Train a model with multiple object classes](https://github.com/FZJ-INM1-BDA/celldetection/blob/main/demos/demo-multiclass.ipynb)

## 🔬 Models
`import celldetection as cd`

###### Contour Proposal Networks

- [`cd.models.CPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CPN)
- [`cd.models.CpnU22`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnU22)
- [`cd.models.CpnSlimU22`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnSlimU22)
- [`cd.models.CpnResUNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResUNet)
- [`cd.models.CpnWideU22`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnWideU22)
- [`cd.models.CpnResNet34FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet34FPN)
- [`cd.models.CpnResNet50FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet50FPN)
- [`cd.models.CpnResNet18FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet18FPN)
- [`cd.models.CpnResNeXt50FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNeXt50FPN)
- [`cd.models.CpnResNet101FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet101FPN)
- [`cd.models.CpnResNet152FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet152FPN)
- [`cd.models.CpnResNeXt101FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNeXt101FPN)
- [`cd.models.CpnResNeXt152FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNeXt152FPN)
- [`cd.models.CpnWideResNet50FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnWideResNet50FPN)
- [`cd.models.CpnWideResNet101FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnWideResNet101FPN)
- [`cd.models.CpnMobileNetV3SmallFPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnMobileNetV3SmallFPN)
- [`cd.models.CpnMobileNetV3LargeFPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnMobileNetV3LargeFPN)


###### U-Nets

- [`cd.models.U22`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.U22)
- [`cd.models.U17`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.U17)
- [`cd.models.U12`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.U12)
- [`cd.models.UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.UNet)
- [`cd.models.WideU22`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.WideU22)
- [`cd.models.SlimU22`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.SlimU22)
- [`cd.models.ResUNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.ResUNet)
- [`cd.models.UNetEncoder`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.UNetEncoder)
- [`cd.models.ResNet50UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.ResNet50UNet)
- [`cd.models.ResNet18UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.ResNet18UNet)
- [`cd.models.ResNet34UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.ResNet34UNet)
- [`cd.models.ResNet152UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.ResNet152UNet)
- [`cd.models.ResNet101UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.ResNet101UNet)
- [`cd.models.ResNeXt50UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.ResNeXt50UNet)
- [`cd.models.ResNeXt152UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.ResNeXt152UNet)
- [`cd.models.ResNeXt101UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.ResNeXt101UNet)
- [`cd.models.WideResNet50UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.WideResNet50UNet)
- [`cd.models.WideResNet101UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.WideResNet101UNet)
- [`cd.models.MobileNetV3SmallUNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.MobileNetV3SmallUNet)
- [`cd.models.MobileNetV3LargeUNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.MobileNetV3LargeUNet)


###### Feature Pyramid Networks

- [`cd.models.FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.FPN)
- [`cd.models.ResNet18FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.ResNet18FPN)
- [`cd.models.ResNet34FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.ResNet34FPN)
- [`cd.models.ResNet50FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.ResNet50FPN)
- [`cd.models.ResNeXt50FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.ResNeXt50FPN)
- [`cd.models.ResNet101FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.ResNet101FPN)
- [`cd.models.ResNet152FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.ResNet152FPN)
- [`cd.models.ResNeXt101FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.ResNeXt101FPN)
- [`cd.models.ResNeXt152FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.ResNeXt152FPN)
- [`cd.models.WideResNet50FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.WideResNet50FPN)
- [`cd.models.WideResNet101FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.WideResNet101FPN)
- [`cd.models.MobileNetV3LargeFPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.MobileNetV3LargeFPN)
- [`cd.models.MobileNetV3SmallFPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.MobileNetV3SmallFPN)


###### Residual Networks

- [`cd.models.ResNet18`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.ResNet18)
- [`cd.models.ResNet34`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.ResNet34)
- [`cd.models.ResNet50`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.ResNet50)
- [`cd.models.ResNet101`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.ResNet101)
- [`cd.models.ResNet152`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.ResNet152)
- [`cd.models.WideResNet50_2`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.WideResNet50_2)
- [`cd.models.ResNeXt50_32x4d`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.ResNeXt50_32x4d)
- [`cd.models.WideResNet101_2`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.WideResNet101_2)
- [`cd.models.ResNeXt101_32x8d`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.ResNeXt101_32x8d)
- [`cd.models.ResNeXt152_32x8d`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.resnet.ResNeXt152_32x8d)


###### Mobile Networks

- [`cd.models.MobileNetV3Large`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.mobilenetv3.MobileNetV3Large)
- [`cd.models.MobileNetV3Small`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.mobilenetv3.MobileNetV3Small)


## 📝 Citing

```
@article{UPSCHULTE2022102371,
    title = {Contour proposal networks for biomedical instance segmentation},
    journal = {Medical Image Analysis},
    volume = {77},
    pages = {102371},
    year = {2022},
    issn = {1361-8415},
    doi = {https://doi.org/10.1016/j.media.2022.102371},
    url = {https://www.sciencedirect.com/science/article/pii/S136184152200024X},
    author = {Eric Upschulte and Stefan Harmeling and Katrin Amunts and Timo Dickscheid},
    keywords = {Cell detection, Cell segmentation, Object detection, CPN},
}
```

## 🔗 Links
- [Article (sciencedirect)](https://www.sciencedirect.com/science/article/pii/S136184152200024X "Contour Proposal Networks for Biomedical Instance Segmentation")
- [PDF (sciencedirect)](https://www.sciencedirect.com/sdfe/reader/pii/S136184152200024X/pdf "Contour Proposal Networks for Biomedical Instance Segmentation")
- [Abstract (arXiv)](https://arxiv.org/abs/2104.03393 "Contour Proposal Networks for Biomedical Instance Segmentation")
- [PyPI](https://pypi.org/project/celldetection/ "CellDetection")
- [Documentation](https://docs.celldetection.org "Documentation")
