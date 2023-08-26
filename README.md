# Cell Detection

[![Downloads](https://static.pepy.tech/badge/celldetection?l)](https://pepy.tech/project/celldetection)
[![Test](https://github.com/FZJ-INM1-BDA/celldetection/workflows/Test/badge.svg)](https://github.com/FZJ-INM1-BDA/celldetection/actions?query=workflow%3ATest)
[![PyPI](https://img.shields.io/pypi/v/celldetection?l)](https://pypi.org/project/celldetection/)
[![Documentation Status](https://readthedocs.org/projects/celldetection/badge/?version=latest)](https://celldetection.readthedocs.io/en/latest/?badge=latest)

## ⭐ Showcase

###### NeurIPS 22 Cell Segmentation Competition

![neurips22](https://raw.githubusercontent.com/FZJ-INM1-BDA/celldetection/main/assets/neurips-cellseg-demo.png "NeurIPS 22 Cell Segmentation Competition - Find more information here: https://neurips.cc/Conferences/2022/CompetitionTrack")
*https://openreview.net/forum?id=YtgRjBw-7GJ*

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

## 💾 Trained models

```python
model = cd.fetch_model(model_name, check_hash=True)
```

| model name                                  | training data                                                                                                        |                                           link                                            |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------:| 
| `ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c` | BBBC039, BBBC038, Omnipose, Cellpose, Sartorius - Cell Instance Segmentation, Livecell, NeurIPS 22 CellSeg Challenge | [🔗](https://celldetection.org/torch/models/ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c.pt) |

<details>
  <summary style="font-weight: bold; color: #888888">Run a demo with a pretrained model</summary>

```python
import torch, cv2, celldetection as cd
from skimage.data import coins
from matplotlib import pyplot as plt

# Load pretrained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = cd.fetch_model('ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c', check_hash=True).to(device)
model.eval()

# Load input
img = coins()
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
print(img.dtype, img.shape, (img.min(), img.max()))

# Run model
with torch.no_grad():
    x = cd.to_tensor(img, transpose=True, device=device, dtype=torch.float32)
    x = x / 255  # ensure 0..1 range
    x = x[None]  # add batch dimension: Tensor[3, h, w] -> Tensor[1, 3, h, w]
    y = model(x)

# Show results for each batch item
contours = y['contours']
for n in range(len(x)):
    cd.imshow_row(x[n], x[n], figsize=(16, 9), titles=('input', 'contours'))
    cd.plot_contours(contours[n])
    plt.show()
```

</details>

## 🔬 Architectures

```python
import celldetection as cd
```

<details>
  <summary style="font-weight: bold; color: #888888">Contour Proposal Networks</summary>

- [`cd.models.CPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CPN)
- [`cd.models.CpnU22`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnU22)
- [`cd.models.CPNCore`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CPNCore)
- [`cd.models.CpnResUNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResUNet)
- [`cd.models.CpnSlimU22`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnSlimU22)
- [`cd.models.CpnWideU22`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnWideU22)
- [`cd.models.CpnResNet18FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet18FPN)
- [`cd.models.CpnResNet34FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet34FPN)
- [`cd.models.CpnResNet50FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet50FPN)
- [`cd.models.CpnResNeXt50FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNeXt50FPN)
- [`cd.models.CpnResNet101FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet101FPN)
- [`cd.models.CpnResNet152FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet152FPN)
- [`cd.models.CpnResNet18UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet18UNet)
- [`cd.models.CpnResNet34UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet34UNet)
- [`cd.models.CpnResNet50UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet50UNet)
- [`cd.models.CpnResNeXt101FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNeXt101FPN)
- [`cd.models.CpnResNeXt152FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNeXt152FPN)
- [`cd.models.CpnResNeXt50UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNeXt50UNet)
- [`cd.models.CpnResNet101UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet101UNet)
- [`cd.models.CpnResNet152UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNet152UNet)
- [`cd.models.CpnResNeXt101UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNeXt101UNet)
- [`cd.models.CpnResNeXt152UNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnResNeXt152UNet)
- [`cd.models.CpnWideResNet50FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnWideResNet50FPN)
- [`cd.models.CpnWideResNet101FPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnWideResNet101FPN)
- [`cd.models.CpnMobileNetV3LargeFPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnMobileNetV3LargeFPN)
- [`cd.models.CpnMobileNetV3SmallFPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnMobileNetV3SmallFPN)

</details>

<details>
  <summary style="font-weight: bold; color: #888888">PyTorch Image Models (timm)</summary>

Also have a look at [Timm Documentation](https://huggingface.co/docs/timm/index).

```python
import timm

timm.list_models(filter='*')  # explore available models
```

- [`cd.models.CpnTimmMaNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnTimmMaNet)
- [`cd.models.CpnTimmUNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnTimmUNet)
- [`cd.models.TimmEncoder`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.timmodels.TimmEncoder)
- [`cd.models.TimmFPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.TimmFPN)
- [`cd.models.TimmMaNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.manet.TimmMaNet)
- [`cd.models.TimmUNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.TimmUNet)

</details>

<details>
  <summary style="font-weight: bold; color: #888888">Segmentation Models PyTorch (smp)</summary>

```python
import segmentation_models_pytorch as smp

smp.encoders.get_encoder_names()  # explore available models
```

```python
encoder = cd.models.SmpEncoder(encoder_name='mit_b5', pretrained='imagenet')
```

Find a list of [Smp Encoders](https://smp.readthedocs.io/en/latest/encoders.html) in the `smp` documentation.

- [`cd.models.CpnSmpMaNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnSmpMaNet)
- [`cd.models.CpnSmpUNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.cpn.CpnSmpUNet)
- [`cd.models.SmpEncoder`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.smp.SmpEncoder)
- [`cd.models.SmpFPN`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.fpn.SmpFPN)
- [`cd.models.SmpMaNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.manet.SmpMaNet)
- [`cd.models.SmpUNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.unet.SmpUNet)

</details>

<details>
    <summary style="font-weight: bold; color: #888888">U-Nets</summary>

```python
# U-Nets are available in 2D and 3D
import celldetection as cd

model = cd.models.ResNeXt50UNet(in_channels=3, out_channels=1, nd=3)
```

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

</details>

<details>
    <summary style="font-weight: bold; color: #888888">MA-Nets</summary>

```python
# Many MA-Nets are available in 2D and 3D
import celldetection as cd

encoder = cd.models.ConvNeXtSmall(in_channels=3, nd=3)
model = cd.models.MaNet(encoder, out_channels=1, nd=3)
```

- [`cd.models.MaNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.manet.MaNet)
- [`cd.models.SmpMaNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.manet.SmpMaNet)
- [`cd.models.TimmMaNet`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.manet.TimmMaNet)

</details>

<details>
    <summary style="font-weight: bold; color: #888888">Feature Pyramid Networks</summary>

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

</details>

<details>
    <summary style="font-weight: bold; color: #888888">ConvNeXt Networks</summary>

```python
# ConvNeXt Networks are available in 2D and 3D
import celldetection as cd

model = cd.models.ConvNeXtSmall(in_channels=3, nd=3)
```

- [`cd.models.ConvNeXt`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.convnext.MaNet)
- [`cd.models.ConvNeXtTiny`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.convnext.ConvNeXtTiny)
- [`cd.models.ConvNeXtSmall`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.convnext.ConvNeXtSmall)
- [`cd.models.ConvNeXtBase`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.convnext.ConvNeXtBase)
- [`cd.models.ConvNeXtLarge`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.convnext.ConvNeXtLarge)

</details>

<details>
    <summary style="font-weight: bold; color: #888888">Residual Networks</summary>

```python
# Residual Networks are available in 2D and 3D
import celldetection as cd

model = cd.models.ResNet50(in_channels=3, nd=3)
```

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

</details>

<details>
    <summary style="font-weight: bold; color: #888888">Mobile Networks</summary>

- [`cd.models.MobileNetV3Large`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.mobilenetv3.MobileNetV3Large)
- [`cd.models.MobileNetV3Small`](https://docs.celldetection.org/en/latest/celldetection.models.html#celldetection.models.mobilenetv3.MobileNetV3Small)

</details>

## 🐳 Docker

Find us on Docker Hub: https://hub.docker.com/r/ericup/celldetection

You can pull the latest version of `celldetection` via:
```
docker pull ericup/celldetection:latest
```

<details>
    <summary style="font-weight: bold; color: #888888">CPN inference via Docker with GPU</summary>

```
docker run --rm \
  -v $PWD/docker/outputs:/outputs/ \
  -v $PWD/docker/inputs/:/inputs/ \
  -v $PWD/docker/models/:/models/ \
  --gpus="device=0" \
  celldetection:latest /bin/bash -c \
  "python cpn_inference.py --tile_size=1024 --stride=768 --precision=32-true"
```
</details>
<details>
    <summary style="font-weight: bold; color: #888888">CPN inference via Docker with CPU</summary>

```
docker run --rm \
  -v $PWD/docker/outputs:/outputs/ \
  -v $PWD/docker/inputs/:/inputs/ \
  -v $PWD/docker/models/:/models/ \
  celldetection:latest /bin/bash -c \
  "python cpn_inference.py --tile_size=1024 --stride=768 --precision=32-true --accelerator=cpu"
```
</details>



### Apptainer

You can also pull our Docker images for the use with [Apptainer](https://apptainer.org/) (formerly [Singularity](https://github.com/apptainer/singularity)) with this command:

```
apptainer pull --dir . --disable-cache docker://ericup/celldetection:latest
```


## 🤗 Hugging Face Spaces

Find us on Hugging Face and upload your own images for segmentation: https://huggingface.co/spaces/ericup/celldetection

<details>
    <summary style="font-weight: bold; color: #888888">Hugging Face API</summary>

### Python

```python
import requests

response = requests.post("https://ericup-celldetection.hf.space/run/predict", json={
    "data": [
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
        "ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c",
    ]
}).json()

data = response["data"]
```

### Javascript

```javascript
const response = await fetch("https://ericup-celldetection.hf.space/run/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
        data: [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==",
            "ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c",
        ]
    })
});

const data = await response.json();

```

</details>

## 🧑‍💻 Napari Plugin

Find our Napari Plugin here: https://github.com/FZJ-INM1-BDA/celldetection-napari </br>
Find out more about Napari here: https://napari.org
![bbbc039](https://raw.githubusercontent.com/FZJ-INM1-BDA/celldetection-napari/main/assets/coins-demo.png "Napari Plugin")
You can install it via pip:
```
pip install git+https://github.com/FZJ-INM1-BDA/celldetection-napari.git
```

## 🏆 Awards

- [NeurIPS 2022 Cell Segmentation Challenge](https://neurips22-cellseg.grand-challenge.org/): Winner Finalist Award

## 📝 Citing

If you find this work useful, please consider giving a **star** ⭐️ and **citation**:

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
- [PyPI](https://pypi.org/project/celldetection/ "CellDetection")
- [Documentation](https://docs.celldetection.org "Documentation")

## 🧑‍🔬 Thanks!

[![Stargazers repo roster for @FZJ-INM1-BDA/celldetection](https://reporoster.com/stars/FZJ-INM1-BDA/celldetection)](https://github.com/FZJ-INM1-BDA/celldetection/stargazers)
[![Forkers repo roster for @FZJ-INM1-BDA/celldetection](https://reporoster.com/forks/FZJ-INM1-BDA/celldetection)](https://github.com/FZJ-INM1-BDA/celldetection/network/members)