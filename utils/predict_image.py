from PIL import Image
import io
import torch
from torchvision import transforms

def predict_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")   # ✅ Force RGB (3 channels only)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = disease_model(img_tensor)
        _, predicted = outputs.max(1)
        return class_names[predicted.item()]
