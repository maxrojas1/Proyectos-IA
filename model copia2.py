import torch
from train import SmallCNN
from torchvision import transforms
from PIL import Image

# 1. Replica arquitectura y carga pesos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SmallCNN(num_classes=2).to(device)
sd     = torch.load("small_model.pth", map_location=device)
model.load_state_dict(sd)
model.eval()

# 2. Preprocesamiento idéntico al training
_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def predict_image(img: Image.Image) -> str:
    """
    Recibe un PIL.Image, lo preprocesa, obtiene la predicción
    y devuelve el texto correspondiente.
    """
    x = _transform(img).unsqueeze(0).to(device)  # (1,3,128,128)
    with torch.no_grad():
        out = model(x)                           # (1,2)
        pred = int(out.argmax(dim=1).item())     # 0 o 1

    if pred == 0:
        return "Imágen no hecha por inteligencia artificial"
    else:
        return "Imágen hecha por inteligencia artificial"
