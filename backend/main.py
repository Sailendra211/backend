from pathlib import Path
import base64
import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
import torch

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

from model import CLASS_NAMES, load_model
from nlp_detector import generate_adversarial_text, perturb_text, predict
from src.attacks.fgsm import FGSMAttack
from src.data.preprocessing import preprocess
from src.detection.detection_engine import DetectionEngine
from src.text.detector import MODE_CLEAN_INPUT, MODE_GENERATED_ATTACK, detect_text


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze for semantic attacks.")
    num_variants: int = Field(default=5, ge=1, le=10)


LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}


app = FastAPI(title="Adversarial Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model()
detector = DetectionEngine(
    model=model,
    device=device,
    transform_names=[
        "gaussian_noise",
        "brightness",
        "blur",
        "jpeg_like",
        "resize_recover",
    ],
)
attacker = FGSMAttack(model, device)


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _tensor_to_data_url(image_tensor: torch.Tensor) -> str:
    image = image_tensor.detach().cpu()
    if image.dim() == 4:
        image = image[0]

    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).mul(255).byte().numpy()
    pil_image = Image.fromarray(image)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


async def _read_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unable to decode the uploaded image.") from exc


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": True,
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    image = await _read_image(file)
    tensor = preprocess(image).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    pred = int(probs.argmax(dim=1).item())
    conf = float(probs.max().item())
    label = CLASS_NAMES[pred]

    result = detector.detector.analyze_sample(tensor)

    return {
        "prediction": {
            "class_id": pred,
            "label": label,
            "confidence": conf,
        },
        "adversarial": {
            "is_adversarial": bool(int(result["is_adversarial"].item())),
            "score": _as_float(result["detection_score"]),
            "consistency": _as_float(result["consistency_ratio"]),
            "confidence_drop": _as_float(result["avg_confidence_drop"]),
            "entropy_increase": _as_float(result["entropy_increase"]),
            "margin_drop": _as_float(result["margin_drop"]),
            "variance": _as_float(result["prob_variance"]),
            "kl_divergence": _as_float(result["avg_kl_divergence"]),
        },
    }


@app.post("/generate-attack")
async def generate_attack(
    file: UploadFile = File(...),
    epsilon: float = Form(0.05, ge=0.0, le=0.1),
):
    image = await _read_image(file)
    tensor = preprocess(image).to(device).detach().clone().requires_grad_(True)

    output = model(tensor)
    clean_probs = torch.softmax(output, dim=1)
    clean_pred = int(clean_probs.argmax(dim=1).item())
    clean_conf = float(clean_probs.max().item())
    label = output.argmax(dim=1)

    loss = torch.nn.functional.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()

    gradients = tensor.grad.detach()
    adv = attacker.fgsm_attack(tensor.detach(), epsilon, gradients)

    with torch.no_grad():
        adv_logits = model(adv)
        adv_probs = torch.softmax(adv_logits, dim=1)

    adv_pred = int(adv_probs.argmax(dim=1).item())
    adv_conf = float(adv_probs.max().item())
    detection_result = detector.detector.analyze_sample(adv)

    return {
        "epsilon": float(epsilon),
        "original_prediction": {
            "class_id": clean_pred,
            "label": CLASS_NAMES[clean_pred],
            "confidence": clean_conf,
        },
        "adversarial_prediction": {
            "class_id": adv_pred,
            "label": CLASS_NAMES[adv_pred],
            "confidence": adv_conf,
        },
        "adversarial": {
            "is_adversarial": bool(int(detection_result["is_adversarial"].item())),
            "score": _as_float(detection_result["detection_score"]),
            "consistency": _as_float(detection_result["consistency_ratio"]),
            "confidence_drop": _as_float(detection_result["avg_confidence_drop"]),
            "entropy_increase": _as_float(detection_result["entropy_increase"]),
            "margin_drop": _as_float(detection_result["margin_drop"]),
            "variance": _as_float(detection_result["prob_variance"]),
            "kl_divergence": _as_float(detection_result["avg_kl_divergence"]),
        },
        "adversarial_image_url": _tensor_to_data_url(adv),
        "message": "Adversarial example generated",
    }


@app.post("/analyze-text")
async def analyze_text_api(payload: TextAnalysisRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    result = detect_text(text, mode=MODE_CLEAN_INPUT, num_variants=payload.num_variants)

    original_pred, original_conf, _ = predict(text)

    return {
        "text": text,
        "original_prediction": {
            "label": LABEL_MAP.get(original_pred, str(original_pred)),
            "confidence": float(original_conf),
        },
        "adversarial": {
            "is_adversarial": bool(result["is_adversarial"]),
            "score": float(result["score"]),
            "consistency": float(result["consistency"]),
            "confidence_drop": float(result["confidence_drop"]),
            "reason": result["reason"],
            "mode": result["mode"],
            "signals": result["signals"],
        },
        "variants": result["variants"],
    }


@app.post("/generate-text-attack")
async def generate_text_attack_api(payload: TextAnalysisRequest):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    attack_result = generate_adversarial_text(text, target_label=0)
    original_detection = detect_text(text, mode=MODE_CLEAN_INPUT, num_variants=payload.num_variants)
    attacked_analysis = detect_text(
        attack_result["adversarial_text"],
        mode=MODE_GENERATED_ATTACK,
        num_variants=payload.num_variants,
    )

    return {
        "original_text": attack_result["original_text"],
        "adversarial_text": attack_result["adversarial_text"],
        "original_prediction": attack_result["original_prediction"],
        "adversarial_prediction": attack_result["adversarial_prediction"],
        "attack_success": attack_result["attack_success"],
        "target_label": attack_result["target_label"],
        "changes": attack_result["changes"],
        "original_detection": {
            "is_adversarial": bool(original_detection["is_adversarial"]),
            "score": float(original_detection["score"]),
            "consistency": float(original_detection["consistency"]),
            "confidence_drop": float(original_detection["confidence_drop"]),
            "reason": original_detection["reason"],
            "mode": original_detection["mode"],
            "signals": original_detection["signals"],
        },
        "adversarial": {
            "is_adversarial": bool(attacked_analysis["is_adversarial"]),
            "score": float(attacked_analysis["score"]),
            "consistency": float(attacked_analysis["consistency"]),
            "confidence_drop": float(attacked_analysis["confidence_drop"]),
            "reason": attacked_analysis["reason"],
            "mode": attacked_analysis["mode"],
            "signals": attacked_analysis["signals"],
        },
        "variants": attacked_analysis["variants"],
    }
