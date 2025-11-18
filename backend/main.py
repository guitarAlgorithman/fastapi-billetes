from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ----------------------------
# CONFIGURACIÓN
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "modelo_billetes.pt"  # pon este archivo en la misma carpeta

# OJO: este orden debe coincidir con train_dataset.classes
CLASS_NAMES = ["apto", "no_apto"]

# Transformaciones IGUALES a las de validación/inferencia
infer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# ESQUEMAS
# ----------------------------
class ImageBase64(BaseModel):
    image_base64: str

# ----------------------------
# FASTAPI
# ----------------------------
app = FastAPI(
    title="API Clasificación Billetes",
    description="Test + modelo: clasifica billetes como 'apto' o 'no_apto' desde base64.",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción restringe esto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# UTILIDADES DE IMAGEN
# ----------------------------
def decode_base64_to_image(image_base64: str) -> Image.Image:
    """Decodifica base64 (con o sin cabecera data:...) y devuelve un objeto PIL.Image."""
    b64_str = image_base64
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(b64_str)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {e}")


# ----------------------------
# ENDPOINT DE TEST (SIN MODELO)
# ----------------------------
@app.get("/")
def root():
    return {"mensaje": "API de billetes OK. Usa /test-imagen-base64 o /predict-billete-base64"}

@app.post("/test-imagen-base64")
def test_imagen_base64(data: ImageBase64):
    img = decode_base64_to_image(data.image_base64)
    width, height = img.size

    print(f"[TEST] Imagen recibida. Tamaño: {width}x{height}")

    return {
        "ok": True,
        "width": width,
        "height": height,
        "mode": img.mode
    }


# ----------------------------
# MODELO
# ----------------------------
def load_model(weights_path: str):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 clases: apto / no_apto
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def predict_pil_image(img: Image.Image, model: torch.nn.Module):
    x = infer_transforms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    label = CLASS_NAMES[pred_idx.item()]
    return label, float(conf.item())


# Cargar el modelo UNA VEZ al arrancar
try:
    model = load_model(MODEL_PATH)
    print(f"[MODELO] Cargado correctamente desde {MODEL_PATH} en {DEVICE}")
except Exception as e:
    print(f"[ERROR MODELO] No se pudo cargar {MODEL_PATH}: {e}")
    model = None


# ----------------------------
# ENDPOINT DE PREDICCIÓN
# ----------------------------
@app.post("/predict-billete-base64")
def predict_billete_base64(data: ImageBase64):
    if model is None:
        # Si el modelo no cargó, no sigas
        raise HTTPException(
            status_code=500,
            detail="Modelo no cargado. Revisa que modelo_billetes.pt exista y sea correcto."
        )

    img = decode_base64_to_image(data.image_base64)

    label, conf = predict_pil_image(img, model)

    # También lo mostramos en consola para debug
    print(f"[PREDICCIÓN] clase={label}, confianza={conf:.4f}")

    return {
        "clase": label,
        "confianza": conf
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
