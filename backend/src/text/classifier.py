import math
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

BASE_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
PERPLEXITY_MODEL_NAME = "distilgpt2"
SENTIMENT_MODEL_PATH = (
    BASE_HF_CACHE
    / "models--distilbert-base-uncased-finetuned-sst-2-english"
    / "snapshots"
    / "714eb0fa89d2f80546fda750413ed43d93601a13"
)
PERPLEXITY_MODEL_PATH = (
    BASE_HF_CACHE
    / "models--distilgpt2"
    / "snapshots"
    / "2290a62682d06624634c1f46a6ad5be0f47f38aa"
)
LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentiment_tokenizer = AutoTokenizer.from_pretrained(
    str(SENTIMENT_MODEL_PATH),
    local_files_only=True,
)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    str(SENTIMENT_MODEL_PATH),
    local_files_only=True,
)
sentiment_model.to(device)
sentiment_model.eval()

perplexity_tokenizer = AutoTokenizer.from_pretrained(
    str(PERPLEXITY_MODEL_PATH),
    local_files_only=True,
)
perplexity_model = AutoModelForCausalLM.from_pretrained(
    str(PERPLEXITY_MODEL_PATH),
    local_files_only=True,
)
perplexity_model.to(device)
perplexity_model.eval()

if perplexity_tokenizer.pad_token is None:
    perplexity_tokenizer.pad_token = perplexity_tokenizer.eos_token


def label_name(label_id: int) -> str:
    return LABEL_MAP.get(label_id, str(label_id))


def _prepare_batch(texts: Sequence[str]):
    encoded = sentiment_tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def predict(text: str) -> Tuple[int, float, torch.Tensor]:
    preds, confs, probs = predict_batch([text])
    return preds[0], confs[0], probs[0]


def predict_batch(texts: Sequence[str]) -> Tuple[List[int], List[float], torch.Tensor]:
    inputs = _prepare_batch(texts)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).detach().cpu()

    preds = probs.argmax(dim=1).tolist()
    confs = probs.max(dim=1).values.tolist()
    return preds, confs, probs


def prediction_summary(text: str):
    pred, conf, probs = predict(text)
    return {
        "label": label_name(pred),
        "confidence": float(conf),
        "probabilities": probs.tolist(),
    }


def get_perplexity(text: str) -> float:
    normalized_text = text.strip() or "."
    tokens = perplexity_tokenizer(
        normalized_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        outputs = perplexity_model(**tokens, labels=tokens["input_ids"])

    loss = float(outputs.loss.item())
    return float(math.exp(min(loss, 20.0)))


def embedding_similarity(text_a: str, text_b: str) -> float:
    texts = [text_a, text_b]
    inputs = _prepare_batch(texts)
    with torch.no_grad():
        outputs = sentiment_model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)

    similarity = F.cosine_similarity(pooled[0].unsqueeze(0), pooled[1].unsqueeze(0))
    return float(similarity.item())
