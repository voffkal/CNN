"""Gradio-демо: повреждена ли машина на фото. Предсказание + тепловая карта Grad-CAM.

Модель и выравнивание фото — те же, что в car_damage_PyTorch_v2.ipynb.
"""
import io
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

CHECKPOINT = Path(__file__).with_name("car_damage_mobilenetv3_2026-10-25.pt")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
COLORMAP = matplotlib.colormaps["jet"]


class CarDamageMobileNet(nn.Module):
    # как в тетрадке: на вход uint8, внутри /255 и нормализация; аугментации на инференсе нет

    def __init__(self):
        super().__init__()
        self.net = models.mobilenet_v3_large(weights=None)
        self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, 2)
        self.features = self.net.features

    def forward(self, x):
        x = x.float() / 255.0
        return self.net((x - MEAN) / STD)


checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
model = CarDamageMobileNet()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
SIZE = tuple(checkpoint["target_size"])
SIDE, QUALITY = checkpoint["equalise_side"], checkpoint["equalise_quality"]


# ZeroGPU не запускает Space без функции с @spaces.GPU, а бесплатный Gradio-тариф — только ZeroGPU.
# Заглушка проходит эту проверку и никогда не вызывается: фото на CPU — ~0,1 с, а декоратор на
# classify() ставил бы каждый клик в очередь за GPU и тратил дневную квоту.
@spaces.GPU(duration=1)
def _gpu_probe():
    return "модель работает на CPU"


def equalise(image):
    # то же выравнивание, что при обучении: 192 px по длинной стороне, JPEG качества 50
    image = image.convert("RGB")
    image.thumbnail((SIDE, SIDE), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=QUALITY)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def grad_cam(image):
    captured = {}
    handle = model.features.register_forward_hook(lambda module, inputs, output: captured.update(a=output))
    logits = model(image[None])
    handle.remove()
    gradients = torch.autograd.grad(logits[0, 0], captured["a"])[0]
    cam = F.relu((gradients.mean(dim=(2, 3), keepdim=True) * captured["a"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().numpy(), logits.softmax(dim=1)[0].detach().numpy()


def classify(image):
    if image is None:
        return None, None
    picture = np.asarray(equalise(image).resize(SIZE, Image.BILINEAR))
    heat, probabilities = grad_cam(torch.from_numpy(picture.copy()).permute(2, 0, 1))
    overlay = picture * 0.55 + COLORMAP(heat)[..., :3] * 255 * 0.45
    return ({"повреждена": float(probabilities[0]), "целая": float(probabilities[1])},
            Image.fromarray(overlay.astype(np.uint8)))


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Фото машины"),
    outputs=[gr.Label(label="Предсказание"), gr.Image(label="Куда смотрит модель (Grad-CAM для «повреждена»)")],
    title="Car damage: повреждена ли машина",
    description=(
        "MobileNetV3, дообученная на ~1 500 фото. "
        f"Accuracy {checkpoint['test_accuracy']:.1%} на 460 отложенных фото. "
        "Каждое фото сначала уменьшается до 192 px и пересжимается — как при обучении, — "
        "чтобы модель не могла опираться на качество снимка. "
        "Тепловая карта показывает, какие области подняли оценку «повреждена»."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
