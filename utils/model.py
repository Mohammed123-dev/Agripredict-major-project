import torch 
import torch.nn as nn
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms

# -------------------------
# ConvBlock helper
# -------------------------
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


# -------------------------
# ResNet9 Model
# -------------------------
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) 
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )
        
        self.conv3 = ConvBlock(128, 256, pool=True) 
        self.conv4 = ConvBlock(256, 512, pool=True) 
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases)
        )
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# -------------------------
# Prediction helper function
# -------------------------
def predict_image(img_bytes, disease_model, class_names):
    # Load uploaded image
    img = Image.open(io.BytesIO(img_bytes))

    # ✅ Fix: Convert RGBA → RGB (always 3 channels)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Preprocess (ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, H, W]

    # Prediction
    disease_model.eval()
    with torch.no_grad():
        outputs = disease_model(img_tensor)
        _, predicted = outputs.max(1)
        return class_names[predicted.item()]
