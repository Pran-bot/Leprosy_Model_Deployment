# =====================================================
# FASTAPI — LEPROSY RISK SCORING
# =====================================================

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# =====================================================
# Model Download
# =====================================================
import os
import requests

MODEL_PATH = "models/best_model.pth"
MODEL_URL = "https://drive.google.com/file/d/1IHxuh3VNNR0E3cUwsr-V5yLJ_nRz3VPd/view?usp=drive_link"  # replace this

# Create folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()  # stops if download failed
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded!")



# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/best_model.pth"

CONF_NEGATIVE = 0.30  # below = NOT leprosy

app = FastAPI(title="Leprosy Risk Scoring API")

# =====================================================
# MODEL DEFINITION
# =====================================================
class LeprosyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=0
        )
        self.binary = nn.Linear(768, 2)
        self.subtype = nn.Linear(768, 4)

    def forward(self, x):
        feat = self.backbone(x)
        return self.binary(feat), self.subtype(feat)

# =====================================================
# LOAD MODEL
# =====================================================
model = LeprosyNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully")

# =====================================================
# TRANSFORMS
# =====================================================
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# PREDICTION LOGIC
# =====================================================
def predict(image: Image.Image):
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = model(img)
        probs = torch.softmax(logits, dim=1)

    score = probs[0, 1].item()

    if score < CONF_NEGATIVE:
        decision = "NOT LEPROSY"
    else:
        decision = "LEPROSY"

    return score, decision


# =====================================================
# API ENDPOINT
# =====================================================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    score, decision = predict(image)

    return JSONResponse({
        "filename": file.filename,
        "score": round(score, 4),
        "decision": decision
    })
