from dataclasses import dataclass
import re
from typing import Dict, List

from src.text.attacker import perturb_text
from src.text.classifier import get_perplexity, predict, predict_batch

MODE_CLEAN_INPUT = "clean_input"
MODE_GENERATED_ATTACK = "generated_attack"

NORMALIZE_PATTERN = re.compile(r"[^a-zA-Z\s]")
CHAR_ATTACK_PATTERNS = [
    re.compile(r"(?:[a-zA-Z]\.){2,}[a-zA-Z]?", re.IGNORECASE),
    re.compile(r"[a-zA-Z]+[0-9@\$][a-zA-Z]+", re.IGNORECASE),
]


@dataclass(frozen=True)
class DetectorWeights:
    prior_bias: float
    threshold: float
    confidence_threshold: float
    perplexity_threshold: float
    perplexity_ratio_threshold: float
    perturbation_flip_threshold: float
    perturbation_confidence_drop_threshold: float
    regex_weight: float
    squeezing_weight: float
    confidence_weight: float
    perplexity_weight: float
    perturbation_weight: float


DETECTOR_CONFIGS = {
    MODE_CLEAN_INPUT: DetectorWeights(
        prior_bias=0.0,
        threshold=1.25,
        confidence_threshold=0.62,
        perplexity_threshold=1400.0,
        perplexity_ratio_threshold=2.6,
        perturbation_flip_threshold=0.38,
        perturbation_confidence_drop_threshold=0.12,
        regex_weight=1.0,
        squeezing_weight=0.75,
        confidence_weight=0.20,
        perplexity_weight=0.55,
        perturbation_weight=0.45,
    ),
    MODE_GENERATED_ATTACK: DetectorWeights(
        prior_bias=0.6,
        threshold=0.58,
        confidence_threshold=0.88,
        perplexity_threshold=850.0,
        perplexity_ratio_threshold=1.7,
        perturbation_flip_threshold=0.18,
        perturbation_confidence_drop_threshold=0.05,
        regex_weight=1.0,
        squeezing_weight=0.80,
        confidence_weight=0.25,
        perplexity_weight=0.65,
        perturbation_weight=0.70,
    ),
}


def _normalize_text(text: str) -> str:
    normalized = NORMALIZE_PATTERN.sub("", text).lower()
    return " ".join(normalized.split())


def _build_perturbation_variants(text: str, num_variants: int) -> List[str]:
    words = text.split()
    variants: List[str] = []

    for index in range(min(len(words), max(2, num_variants // 2 + 1))):
        removed = " ".join(words[:index] + words[index + 1 :]).strip()
        if removed and removed != text:
            variants.append(removed)

    while len(variants) < num_variants:
        variant = perturb_text(text)
        if variant not in variants:
            variants.append(variant)
        elif variant == text and len(variants) >= max(1, num_variants - 1):
            break

    return variants[:num_variants]


def _regex_signal(text: str):
    matches = [pattern.pattern for pattern in CHAR_ATTACK_PATTERNS if pattern.search(text)]
    return {
        "matches": matches,
        "flag": bool(matches),
    }


def _squeezing_signal(text: str):
    squeezed_text = _normalize_text(text)
    original_pred, _, _ = predict(text)
    squeezed_pred, _, _ = predict(squeezed_text or text)
    return {
        "squeezed_text": squeezed_text,
        "flag": bool(squeezed_text and squeezed_pred != original_pred),
        "squeezed_label": squeezed_pred,
    }


def _confidence_signal(text: str, confidence_threshold: float):
    pred, conf, _ = predict(text)
    return {
        "label_id": pred,
        "confidence": float(conf),
        "flag": bool(conf < confidence_threshold),
    }


def _perplexity_signal(text: str, config: DetectorWeights):
    normalized_text = _normalize_text(text) or text
    raw_perplexity = get_perplexity(text)
    normalized_perplexity = get_perplexity(normalized_text)
    perplexity_ratio = raw_perplexity / max(normalized_perplexity, 1e-6)

    high_perplexity_flag = raw_perplexity > config.perplexity_threshold
    high_ratio_flag = perplexity_ratio > config.perplexity_ratio_threshold
    flag = bool((high_perplexity_flag and high_ratio_flag) or raw_perplexity > config.perplexity_threshold * 1.6)

    return {
        "raw_perplexity": float(raw_perplexity),
        "normalized_perplexity": float(normalized_perplexity),
        "perplexity_ratio": float(perplexity_ratio),
        "flag": flag,
    }


def _perturbation_signal(text: str, num_variants: int, config: DetectorWeights):
    original_pred, original_conf, original_probs = predict(text)
    original_label_conf = float(original_probs[original_pred].item())
    variants = _build_perturbation_variants(text, num_variants)

    if not variants:
        return {
            "variants": [],
            "flip_rate": 0.0,
            "consistency": 1.0,
            "confidence_drop": 0.0,
            "flag": False,
        }

    preds, _, probs = predict_batch(variants)
    original_label_confs = [float(prob[original_pred].item()) for prob in probs]
    flips = sum(1 for pred in preds if pred != original_pred)
    flip_rate = flips / len(preds)
    consistency = 1.0 - flip_rate
    avg_conf = sum(original_label_confs) / len(original_label_confs)
    confidence_drop = original_label_conf - avg_conf
    flag = bool(
        flip_rate >= config.perturbation_flip_threshold
        or confidence_drop >= config.perturbation_confidence_drop_threshold
    )

    detailed_variants = []
    for index, variant in enumerate(variants):
        detailed_variants.append(
            {
                "id": str(index + 1),
                "text": variant,
                "prediction": "POSITIVE" if preds[index] == 1 else "NEGATIVE",
                "confidence": float(max(probs[index]).item()),
            }
        )

    return {
        "variants": detailed_variants,
        "flip_rate": float(flip_rate),
        "consistency": float(consistency),
        "confidence_drop": float(confidence_drop),
        "flag": flag,
    }


def _compose_reason(signals: Dict[str, object]) -> str:
    if signals["regex_char_attack"]["flag"]:
        return "Character-level attack pattern detected"
    if signals["squeezing_flip"]["flag"]:
        return "Prediction flipped after input squeezing"
    if signals["perplexity"]["flag"]:
        return "Perplexity and structural anomaly threshold exceeded"
    if signals["perturbation"]["flag"]:
        return "Prediction instability under perturbations"
    if signals["confidence"]["flag"]:
        return "Low classifier confidence"
    if signals["prior_bias"]["flag"]:
        return "Generated-attack mode prior elevated the adversarial score"
    return "No strong adversarial indicators"


def detect_text(text: str, mode: str = MODE_CLEAN_INPUT, num_variants: int = 5):
    config = DETECTOR_CONFIGS[mode]
    regex_signal = _regex_signal(text)
    squeezing_signal = _squeezing_signal(text)
    confidence_signal = _confidence_signal(text, config.confidence_threshold)
    perplexity_signal = _perplexity_signal(text, config)
    perturbation_signal = _perturbation_signal(text, num_variants, config)

    score = float(config.prior_bias)
    score += config.regex_weight if regex_signal["flag"] else 0.0
    score += config.squeezing_weight if squeezing_signal["flag"] else 0.0
    score += config.confidence_weight if confidence_signal["flag"] else 0.0
    score += config.perplexity_weight if perplexity_signal["flag"] else 0.0
    score += config.perturbation_weight if perturbation_signal["flag"] else 0.0

    signals = {
        "regex_char_attack": {
            "matches": regex_signal["matches"],
            "flag": regex_signal["flag"],
        },
        "squeezing_flip": {
            "flag": squeezing_signal["flag"],
            "squeezed_text": squeezing_signal["squeezed_text"],
        },
        "confidence": {
            "value": confidence_signal["confidence"],
            "flag": confidence_signal["flag"],
        },
        "prior_bias": {
            "value": float(config.prior_bias),
            "flag": bool(config.prior_bias > 0.0),
        },
        "perplexity": {
            "raw": perplexity_signal["raw_perplexity"],
            "normalized": perplexity_signal["normalized_perplexity"],
            "ratio": perplexity_signal["perplexity_ratio"],
            "flag": perplexity_signal["flag"],
        },
        "perturbation": {
            "flip_rate": perturbation_signal["flip_rate"],
            "consistency": perturbation_signal["consistency"],
            "confidence_drop": perturbation_signal["confidence_drop"],
            "flag": perturbation_signal["flag"],
        },
    }

    return {
        "is_adversarial": bool(score >= config.threshold or regex_signal["flag"]),
        "score": float(score),
        "mode": mode,
        "reason": _compose_reason(signals),
        "consistency": float(perturbation_signal["consistency"]),
        "confidence_drop": float(perturbation_signal["confidence_drop"]),
        "signals": signals,
        "variants": perturbation_signal["variants"],
    }


def analyze_text(text: str, num_variants: int = 5):
    return detect_text(text, mode=MODE_CLEAN_INPUT, num_variants=num_variants)
