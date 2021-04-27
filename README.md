# celldetection
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
[Here is a multi-class example with toy data.](https://github.com/FZJ-INM1-BDA/celldetection/blob/main/demos/demo-multiclass.ipynb)

If your data contains images and [label images](https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.label), you can exchange the toy data with your own data.

## 🔬 Models
`from celldetection import models`
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
- `models.UNetEncoder`
- `models.UNet`
- `models.FPN`
- `models.CPN`


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