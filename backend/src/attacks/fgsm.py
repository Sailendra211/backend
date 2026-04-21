from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


@dataclass
class AttackResult:
    epsilon: float
    clean_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    examples: List[Tuple[torch.Tensor, torch.Tensor, int, int, int]]
    # each example = (clean_image, adv_image, true_label, clean_pred, adv_pred)


class FGSMAttack:
    """
    FGSM attack utility for PyTorch image classifiers.
    Supports both raw [0,1] images and normalized inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        self.use_normalization = mean is not None and std is not None
        if self.use_normalization:
            self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
            self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        else:
            self.mean = None
            self.std = None

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_normalization:
            return x
        return x * self.std + self.mean

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_normalization:
            return x
        return (x - self.mean) / self.std

    def fgsm_attack(self, images: torch.Tensor, epsilon: float, gradients: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM.

        If normalization is used, perturbation is applied in pixel space [0,1],
        then the result is normalized again for the model.
        """
        if self.use_normalization:
            images_denorm = self.denormalize(images)
            perturbed = images_denorm + epsilon * gradients.sign()
            perturbed = torch.clamp(perturbed, 0.0, 1.0)
            perturbed = self.normalize(perturbed)
            return perturbed

        perturbed = images + epsilon * gradients.sign()
        perturbed = torch.clamp(perturbed, 0.0, 1.0)
        return perturbed

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(images)
        return outputs.argmax(dim=1)

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            clean_preds, adv_images, adv_preds
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        images_for_grad = images.clone().detach().requires_grad_(True)

        outputs = self.model(images_for_grad)
        clean_preds = outputs.argmax(dim=1)

        loss = self.criterion(outputs, labels)
        self.model.zero_grad()
        loss.backward()

        gradients = images_for_grad.grad.detach()
        adv_images = self.fgsm_attack(images_for_grad.detach(), epsilon, gradients)

        with torch.no_grad():
            adv_outputs = self.model(adv_images)
            adv_preds = adv_outputs.argmax(dim=1)

        return clean_preds.detach(), adv_images.detach(), adv_preds.detach()

    def evaluate(
        self,
        dataloader,
        epsilon: float,
        max_examples: int = 8,
    ) -> AttackResult:
        """
        Evaluates model robustness against FGSM attack.

        Metrics:
        - clean_accuracy: accuracy on original images
        - adversarial_accuracy: accuracy on adversarial images
        - attack_success_rate: among originally-correct samples, fraction flipped by attack
        """
        total = 0
        clean_correct = 0
        adv_correct = 0

        originally_correct = 0
        successful_attacks = 0

        saved_examples: List[Tuple[torch.Tensor, torch.Tensor, int, int, int]] = []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            clean_preds, adv_images, adv_preds = self.attack_batch(images, labels, epsilon)

            total += labels.size(0)
            clean_correct += (clean_preds == labels).sum().item()
            adv_correct += (adv_preds == labels).sum().item()

            correct_mask = clean_preds == labels
            attack_success_mask = correct_mask & (adv_preds != labels)

            originally_correct += correct_mask.sum().item()
            successful_attacks += attack_success_mask.sum().item()

            if len(saved_examples) < max_examples:
                for i in range(images.size(0)):
                    if len(saved_examples) >= max_examples:
                        break
                    if attack_success_mask[i].item():
                        saved_examples.append(
                            (
                                images[i].detach().cpu(),
                                adv_images[i].detach().cpu(),
                                int(labels[i].item()),
                                int(clean_preds[i].item()),
                                int(adv_preds[i].item()),
                            )
                        )

        clean_accuracy = clean_correct / total if total > 0 else 0.0
        adversarial_accuracy = adv_correct / total if total > 0 else 0.0
        attack_success_rate = (
            successful_attacks / originally_correct if originally_correct > 0 else 0.0
        )

        return AttackResult(
            epsilon=epsilon,
            clean_accuracy=clean_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            attack_success_rate=attack_success_rate,
            examples=saved_examples,
        )

    def run_multiple_epsilons(
        self,
        dataloader,
        epsilons: List[float],
        max_examples_per_epsilon: int = 8,
    ) -> Dict[float, AttackResult]:
        results: Dict[float, AttackResult] = {}
        for epsilon in epsilons:
            results[epsilon] = self.evaluate(
                dataloader=dataloader,
                epsilon=epsilon,
                max_examples=max_examples_per_epsilon,
            )
        return results