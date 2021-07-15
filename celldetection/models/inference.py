import torch
from ..util.util import asnumpy

__all__ = ['Inference']


class Inference:
    def __init__(self, model, device=None, amp=False, transforms=None):
        self.transforms = transforms
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.use_amp = amp

    def __call__(self, inputs):
        if self.transforms is not None:
            inputs = self.transforms(inputs)
        inputs = torch.as_tensor(inputs, device=self.device, dtype=torch.float32)
        if inputs.ndim == 2:
            inputs = inputs[None]
        if inputs.ndim == 3:
            inputs = inputs[None]
        with torch.cuda.amp.autocast(self.use_amp):
            out = self.model(inputs)
        return asnumpy(out)
