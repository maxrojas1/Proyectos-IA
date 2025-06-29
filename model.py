# model.py
import io
import torch
import torch.nn.functional as F
import numpy as np
from train import SmallCNN
from torchvision import transforms
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

# 1) Carga y configuración del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN(num_classes=2).to(device)
state = torch.load("small_model.pth", map_location=device)
model.load_state_dict(state)
model.eval()

# 2) Transformación de entrada (128×128)
_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

# 3) GradCAM para activaciones y gradientes
torch.manual_seed(0)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def get_cam(self, x):
        outputs = self.model(x)
        score = outputs[0,1] - outputs[0,0]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        grads = self.gradients[0]
        weights = grads.mean(dim=(1,2), keepdim=True)
        acts = self.activations[0]
        cam = F.relu((weights * acts).sum(dim=0))
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(128,128), mode='bilinear', align_corners=False
        )[0,0]
        return cam.cpu().numpy()

# Instancia GradCAM en la última capa convolucional
gradcam = GradCAM(model, model.features[6])


def predict_with_overlay(img: Image.Image, alpha: float = 0.5):
    """
    Retorna:
      - message: texto de clasificación
      - overlay: PIL.Image con heatmap superpuesto, mismo tamaño original
    """
    orig_w, orig_h = img.size
    resized = img.resize((128,128))
    x = _transform(resized).unsqueeze(0).to(device)

    # Predicción
    with torch.no_grad():
        out = model(x)
        pred = int(out.argmax(dim=1).item())
    message = (
        "Imágen no hecha por inteligencia artificial" if pred == 0 else
        "Imágen hecha por inteligencia artificial"
    )

    # Generar CAM combinado
    cam = gradcam.get_cam(x)  # rango [0,1]
    cmap = LinearSegmentedColormap.from_list('gwred', ['green','white','red'])
    heat_np = cmap(cam)[:, :, :3]
    heat_img = Image.fromarray((heat_np * 255).astype(np.uint8)).convert('RGBA')
    heat_img.putalpha(int(alpha * 255))

    gray_rgba = resized.convert('L').convert('RGBA')
    overlay_small = Image.alpha_composite(gray_rgba, heat_img)
    overlay = overlay_small.resize((orig_w, orig_h), resample=Image.BILINEAR)

    return message, overlay
