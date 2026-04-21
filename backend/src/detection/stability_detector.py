# src/detection/stability_detector.py

import torch
import torch.nn.functional as F

from src.detection.image_transforms import ImageTransforms


class StabilityDetector:
    """
    Prediction-stability-based adversarial detector.

    Improved version:
    - Uses multiple stability signals instead of only two
    - Returns a continuous detection score
    - Defers thresholding to the final step
    """

    def __init__(
      self,
      model,
      device,
      transform_names=None,
      consistency_threshold=0.6,
      confidence_drop_threshold=0.2,
      entropy_increase_threshold=0.15,
      margin_drop_threshold=0.15,
      score_threshold=0.30,
      alpha=0.25,
      beta=0.25,
      gamma=0.05,
      delta=0.05,
      eta=0.15,
      zeta=0.25,   # NEW: KL divergence weight
  ):
      self.model = model
      self.device = device
      self.model.eval()

      self.transforms = ImageTransforms(device=device)

      self.transform_names = transform_names or [
          "gaussian_noise",
          "brightness",
          "blur",
          "jpeg_like",
          "resize_recover",
      ]

      # legacy thresholds
      self.consistency_threshold = consistency_threshold
      self.confidence_drop_threshold = confidence_drop_threshold

      # new thresholds
      self.entropy_increase_threshold = entropy_increase_threshold
      self.margin_drop_threshold = margin_drop_threshold
      self.score_threshold = score_threshold

      # score weights
      self.alpha = alpha   # inconsistency
      self.beta = beta     # confidence drop
      self.gamma = gamma   # entropy increase
      self.delta = delta   # margin drop
      self.eta = eta       # prob variance
      self.zeta = zeta     # KL divergence

    def _ensure_batch(self, x):
        if x.dim() == 3:
            return x.unsqueeze(0), True
        elif x.dim() == 4:
            return x, False
        else:
            raise ValueError(f"Expected input shape (C,H,W) or (B,C,H,W), got {x.shape}")

    def _get_predictions(self, x):
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = probs.max(dim=1).values

        return logits, probs, preds, confs

    def _entropy(self, probs):
        return -(probs * torch.log(probs + 1e-8)).sum(dim=1)

    def _top2_margin(self, probs):
        top2 = torch.topk(probs, k=2, dim=1).values
        return top2[:, 0] - top2[:, 1]

    def _kl_divergence(self, p, q):
        """
        KL(p || q), averaged per sample over class dimension.
        p, q shape: (B, C)
        """
        return (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(dim=1)

    def compute_detection_score(
        self,
        consistency_ratio,
        confidence_drop,
        entropy_increase,
        margin_drop,
        prob_variance,
        avg_kl_divergence,   # NEW
    ):
        score = (
            self.alpha * (1.0 - consistency_ratio) +
            self.beta * confidence_drop +
            self.gamma * torch.clamp(entropy_increase, min=0.0) +
            self.delta * torch.clamp(margin_drop, min=0.0) +
            self.eta * prob_variance +
            self.zeta * avg_kl_divergence
        )
        return score

    def analyze_sample(self, x):
        x, was_single = self._ensure_batch(x)
        x = x.to(self.device)

        _, orig_probs, orig_preds, orig_confs = self._get_predictions(x)

        orig_entropy = self._entropy(orig_probs)
        orig_margin = self._top2_margin(orig_probs)

        transformed_outputs = {}

        transformed_preds = []
        transformed_target_confs = []
        transformed_entropies = []
        transformed_margins = []
        transformed_probs = []
        transformed_kls = []

        for tname in self.transform_names:
            transformed = self.transforms.get_selected_transforms(x, [tname])[tname]
            _, t_probs, t_preds, t_confs = self._get_predictions(transformed)

            transformed_outputs[tname] = {
                "preds": t_preds.detach().cpu(),
                "max_confs": t_confs.detach().cpu(),
            }

            transformed_preds.append(t_preds.unsqueeze(0))
            transformed_probs.append(t_probs.unsqueeze(0))

            orig_class_conf = t_probs.gather(1, orig_preds.unsqueeze(1)).squeeze(1)
            transformed_target_confs.append(orig_class_conf.unsqueeze(0))

            t_entropy = self._entropy(t_probs)
            transformed_entropies.append(t_entropy.unsqueeze(0))

            t_margin = self._top2_margin(t_probs)
            transformed_margins.append(t_margin.unsqueeze(0))

            kl = self._kl_divergence(orig_probs, t_probs)
            transformed_kls.append(kl.unsqueeze(0))

        transformed_preds = torch.cat(transformed_preds, dim=0)               # (T, B)
        transformed_probs = torch.cat(transformed_probs, dim=0)               # (T, B, C)
        transformed_target_confs = torch.cat(transformed_target_confs, dim=0) # (T, B)
        transformed_entropies = torch.cat(transformed_entropies, dim=0)       # (T, B)
        transformed_margins = torch.cat(transformed_margins, dim=0)           # (T, B)
        transformed_kls = torch.cat(transformed_kls, dim=0)                   # (T, B)

        matches = (transformed_preds == orig_preds.unsqueeze(0)).float()
        consistency_ratio = matches.mean(dim=0)

        avg_transformed_conf = transformed_target_confs.mean(dim=0)
        confidence_drop = orig_confs - avg_transformed_conf

        avg_transformed_entropy = transformed_entropies.mean(dim=0)
        entropy_increase = avg_transformed_entropy - orig_entropy

        avg_transformed_margin = transformed_margins.mean(dim=0)
        margin_drop = orig_margin - avg_transformed_margin

        prob_variance = transformed_target_confs.var(dim=0, unbiased=False)

        avg_kl_divergence = transformed_kls.mean(dim=0)

        detection_score = self.compute_detection_score(
            consistency_ratio=consistency_ratio,
            confidence_drop=confidence_drop,
            entropy_increase=entropy_increase,
            margin_drop=margin_drop,
            prob_variance=prob_variance,
            avg_kl_divergence=avg_kl_divergence,
        )

        is_adversarial = detection_score >= self.score_threshold

        result = {
            "original_preds": orig_preds.detach().cpu(),
            "original_confs": orig_confs.detach().cpu(),
            "consistency_ratio": consistency_ratio.detach().cpu(),
            "avg_confidence_drop": confidence_drop.detach().cpu(),
            "entropy_increase": entropy_increase.detach().cpu(),
            "margin_drop": margin_drop.detach().cpu(),
            "prob_variance": prob_variance.detach().cpu(),
            "avg_kl_divergence": avg_kl_divergence.detach().cpu(),
            "detection_score": detection_score.detach().cpu(),
            "is_adversarial": is_adversarial.detach().cpu(),
            "transformed_outputs": transformed_outputs,
        }

        if was_single:
            return {
                "original_pred": result["original_preds"][0].item(),
                "original_conf": result["original_confs"][0].item(),
                "consistency_ratio": result["consistency_ratio"][0].item(),
                "avg_confidence_drop": result["avg_confidence_drop"][0].item(),
                "entropy_increase": result["entropy_increase"][0].item(),
                "margin_drop": result["margin_drop"][0].item(),
                "prob_variance": result["prob_variance"][0].item(),
                "avg_kl_divergence": result["avg_kl_divergence"][0].item(),
                "detection_score": result["detection_score"][0].item(),
                "is_adversarial": bool(result["is_adversarial"][0].item()),
                "transformed_outputs": {
                    name: {
                        "pred": data["preds"][0].item(),
                        "max_conf": data["max_confs"][0].item(),
                    }
                    for name, data in transformed_outputs.items()
                },
            }

        return result

    def detect(self, x):
        result = self.analyze_sample(x)
        return result["is_adversarial"] if "is_adversarial" in result else result

    def batch_summary(self, x, labels=None):
        result = self.analyze_sample(x)

        summary = {
            "original_preds": result["original_preds"],
            "original_confs": result["original_confs"],
            "consistency_ratio": result["consistency_ratio"],
            "avg_confidence_drop": result["avg_confidence_drop"],
            "entropy_increase": result["entropy_increase"],
            "margin_drop": result["margin_drop"],
            "prob_variance": result["prob_variance"],
            "avg_kl_divergence": result["avg_kl_divergence"],
            "detection_score": result["detection_score"],
            "is_adversarial": result["is_adversarial"],
            "num_flagged": int(result["is_adversarial"].sum().item()),
            "batch_size": int(result["is_adversarial"].shape[0]),
        }

        if labels is not None:
            labels = labels.detach().cpu()
            summary["labels"] = labels
            summary["clean_accuracy_on_batch"] = float(
                (result["original_preds"] == labels).float().mean().item()
            )

        return summary