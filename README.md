# Cell Detection

[![Downloads](https://pepy.tech/badge/celldetection?l)](https://pepy.tech/project/celldetection)
[![Test](https://github.com/FZJ-INM1-BDA/celldetection/workflows/Test/badge.svg)](https://github.com/FZJ-INM1-BDA/celldetection/actions?query=workflow%3ATest)
[![PyPI](https://img.shields.io/pypi/v/celldetection?l)](https://pypi.org/project/celldetection/)

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
pip install celldetection
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
`from celldetection import models`

###### Contour Proposal Networks:
- `models.CpnU22`
- `models.CpnSlimU22`
- `models.CpnWideU22`
- `models.CpnResNet18FPN`
- `models.CpnResNet34FPN`
- `models.CpnResNet50FPN`
- `models.CpnResNet101FPN`
- `models.CpnResNet152FPN`
- `models.CpnResNeXt50FPN`
- `models.CpnResNeXt101FPN`
- `models.CpnResNeXt152FPN`
- `models.CpnWideResNet50FPN`
- `models.CpnWideResNet101FPN`
- `models.CpnMobileNetV3SmallFPN`
- `models.CpnMobileNetV3LargeFPN`
- `models.CPN`

###### U-Nets:
- `models.U22`
- `models.SlimU22`
- `models.WideU22`
- `models.U17`
- `models.U12`
- `models.UNetEncoder`
- `models.UNet`

###### Feature Pyramid Networks:
- `models.ResNet18FPN`
- `models.ResNet34FPN`
- `models.ResNet50FPN`
- `models.ResNet101FPN`
- `models.ResNet152FPN`
- `models.ResNeXt50FPN`
- `models.ResNeXt101FPN`
- `models.ResNeXt152FPN`
- `models.WideResNet50FPN`
- `models.WideResNet101FPN`
- `models.MobileNetV3SmallFPN`
- `models.MobileNetV3LargeFPN`
- `models.FPN`

###### Residual Networks:
- `models.ResNet18`
- `models.ResNet34`
- `models.ResNet50`
- `models.ResNet101`
- `models.ResNet152`
- `models.ResNeXt50_32x4d`
- `models.ResNeXt101_32x8d`
- `models.ResNeXt152_32x8d`
- `models.WideResNet50_2`
- `models.WideResNet101_2`

###### Mobile Networks:
- `models.MobileNetV3Small`
- `models.MobileNetV3Large`



## 📝 Citing

```
@misc{upschulte2021contour,
      title={Contour Proposal Networks for Biomedical Instance Segmentation}, 
      author={Eric Upschulte and Stefan Harmeling and Katrin Amunts and Timo Dickscheid},
      year={2021},
      eprint={2104.03393},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🔗 Links
- [Abstract (arXiv)](https://arxiv.org/abs/2104.03393 "Contour Proposal Networks for Biomedical Instance Segmentation")
- [PDF (arXiv)](https://arxiv.org/pdf/2104.03393 "Contour Proposal Networks for Biomedical Instance Segmentation")
- [Bibtex (arXiv)](https://arxiv.org/bibtex/2104.03393 "Contour Proposal Networks for Biomedical Instance Segmentation")
- [PyPI](https://pypi.org/project/celldetection/ "CellDetection")
