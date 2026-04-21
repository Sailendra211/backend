import os
from pathlib import Path
import random
import re
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault(
    "TA_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / ".cache" / "textattack"),
)

import torch
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper

from src.text.classifier import (
    LABEL_MAP,
    label_name,
    predict,
    sentiment_model,
    sentiment_tokenizer,
)

TOKEN_PATTERN = re.compile(r"\w+(?:-\w+)*|\s+|[^\w\s]")

soft_replacements = {
    "actor": ["performer", "cast member"],
    "amazing": ["remarkable", "striking"],
    "awful": ["rough", "harsh"],
    "bad": ["poor", "weak"],
    "beautiful": ["elegant", "vivid"],
    "compelling": ["engaging", "absorbing"],
    "delightful": ["pleasant", "amiable"],
    "engrossing": ["absorbing", "engaging"],
    "excellent": ["strong", "solid"],
    "flat": ["bland", "plain"],
    "good": ["solid", "strong"],
    "great": ["strong", "excellent"],
    "incredible": ["impressive", "remarkable"],
    "interesting": ["intriguing", "curious"],
    "lifeless": ["hollow", "muted"],
    "moving": ["touching", "affecting"],
    "movie": ["film", "feature"],
    "story": ["narrative", "plot"],
    "terrible": ["awful", "grim"],
    "wonderful": ["lovely", "excellent"],
}

class LocalHuggingFaceWrapper(HuggingFaceModelWrapper):
    def __call__(self, text_input_list):
        if isinstance(text_input_list, str):
            text_input_list = [text_input_list]
        encoded = self.tokenizer(
            text_input_list,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model(**encoded)
        return outputs.logits


_model_wrapper = LocalHuggingFaceWrapper(sentiment_model, sentiment_tokenizer)
_textfooler_attack = TextFoolerJin2019.build(_model_wrapper)


def _restore_case(source_word: str, replacement: str) -> str:
    if source_word.isupper():
        return replacement.upper()
    if source_word[:1].isupper():
        return replacement.capitalize()
    return replacement


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def _join_tokens(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def _replace_token(tokens: Sequence[str], index: int, replacement: str) -> str:
    updated = list(tokens)
    updated[index] = replacement
    return _join_tokens(updated)


def _soft_options(word: str) -> List[str]:
    return [_restore_case(word, replacement) for replacement in soft_replacements.get(word.lower(), [])]


def perturb_text(text: str) -> str:
    tokens = _tokenize(text)
    changed = False

    for index, token in enumerate(tokens):
        if not re.fullmatch(r"\w+(?:-\w+)*", token):
            continue

        options = _soft_options(token)
        if options and random.random() < 0.45:
            tokens[index] = random.choice(options)
            changed = True

    if not changed:
        for index, token in enumerate(tokens):
            if not re.fullmatch(r"\w+(?:-\w+)*", token):
                continue
            options = _soft_options(token)
            if options:
                tokens[index] = random.choice(options)
                changed = True
                break

    return _join_tokens(tokens) if changed else text


def _derive_changes(original_text: str, adversarial_text: str):
    original_tokens = _tokenize(original_text)
    adversarial_tokens = _tokenize(adversarial_text)
    changes = []

    for original_token, adversarial_token in zip(original_tokens, adversarial_tokens):
        if original_token == adversarial_token:
            continue
        if re.fullmatch(r"\s+", original_token) or re.fullmatch(r"\s+", adversarial_token):
            continue
        changes.append(
            {
                "from": original_token,
                "to": adversarial_token,
            }
        )

    if not changes and original_text != adversarial_text:
        changes.append(
            {
                "from": original_text,
                "to": adversarial_text,
            }
        )

    return changes


def generate_adversarial_text(text: str, target_label: int = 0, max_steps: int = 4):
    original_pred, original_conf, original_probs = predict(text)
    try:
        attack_result = _textfooler_attack.attack(text, original_pred)
        perturbed_text = attack_result.perturbed_result.attacked_text.text
    except Exception:
        perturbed_text = text

    best_pred, best_conf, best_probs = predict(perturbed_text)
    changes = _derive_changes(text, perturbed_text)
    attack_success = bool(best_pred == target_label and perturbed_text != text)

    return {
        "original_text": text,
        "adversarial_text": perturbed_text,
        "original_prediction": {
            "label": label_name(original_pred),
            "confidence": float(original_conf),
            "probabilities": original_probs.tolist(),
        },
        "adversarial_prediction": {
            "label": label_name(best_pred),
            "confidence": float(best_conf),
            "probabilities": best_probs.tolist(),
        },
        "attack_success": attack_success,
        "target_label": label_name(target_label),
        "changes": changes,
    }
