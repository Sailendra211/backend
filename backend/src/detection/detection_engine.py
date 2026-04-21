# src/detection/detection_engine.py

import torch

from src.detection.stability_detector import StabilityDetector


class DetectionEngine:
    """
    Runs prediction-stability-based adversarial detection on clean and
    adversarial samples, and computes detection metrics.

    Detection label convention:
        0 -> clean
        1 -> adversarial
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
        zeta=0.25,
    ):
        self.model = model
        self.device = device

        self.detector = StabilityDetector(
            model=model,
            device=device,
            transform_names=transform_names,
            consistency_threshold=consistency_threshold,
            confidence_drop_threshold=confidence_drop_threshold,
            entropy_increase_threshold=entropy_increase_threshold,
            margin_drop_threshold=margin_drop_threshold,
            score_threshold=score_threshold,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            eta=eta,
            zeta=zeta,
        )

    def _to_cpu_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return torch.tensor(x)

    def _compute_confusion_counts(self, y_true, y_pred):
        """
        Binary classification confusion matrix counts:
            TP: predicted adversarial and actually adversarial
            TN: predicted clean and actually clean
            FP: predicted adversarial but actually clean
            FN: predicted clean but actually adversarial
        """
        y_true = self._to_cpu_tensor(y_true).int()
        y_pred = self._to_cpu_tensor(y_pred).int()

        tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
        tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
        fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
        fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

        return tp, tn, fp, fn

    def _safe_div(self, num, den):
        return num / den if den != 0 else 0.0

    def compute_metrics(self, y_true, y_pred):
        """
        Compute binary detection metrics.
        """
        tp, tn, fp, fn = self._compute_confusion_counts(y_true, y_pred)

        accuracy = self._safe_div(tp + tn, tp + tn + fp + fn)
        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        specificity = self._safe_div(tn, tn + fp)
        f1 = self._safe_div(2 * precision * recall, precision + recall)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def evaluate_batch(self, images, true_detection_labels, class_labels=None):
        """
        Evaluate a batch for detection.

        Args:
            images: tensor of shape (B, C, H, W)
            true_detection_labels: tensor/list where
                                   0 = clean, 1 = adversarial
            class_labels: optional ground-truth class labels for classifier

        Returns:
            dict with per-batch predictions, scores, and metrics
        """
        result = self.detector.analyze_sample(images)

        pred_detection_labels = result["is_adversarial"].int().cpu()
        true_detection_labels = self._to_cpu_tensor(true_detection_labels).int()

        metrics = self.compute_metrics(true_detection_labels, pred_detection_labels)

        output = {
            "true_detection_labels": true_detection_labels,
            "pred_detection_labels": pred_detection_labels,
            "original_preds": result["original_preds"],
            "original_confs": result["original_confs"],
            "consistency_ratio": result["consistency_ratio"],
            "avg_confidence_drop": result["avg_confidence_drop"],
            "entropy_increase": result["entropy_increase"],
            "margin_drop": result["margin_drop"],
            "prob_variance": result["prob_variance"],
            "avg_kl_divergence": result["avg_kl_divergence"],
            "detection_score": result["detection_score"],
            "metrics": metrics,
        }

        if class_labels is not None:
            class_labels = self._to_cpu_tensor(class_labels).long()
            output["class_labels"] = class_labels
            output["classifier_accuracy"] = float(
                (result["original_preds"] == class_labels).float().mean().item()
            )

        return output

    def evaluate_loader(self, dataloader, attack_label=0):
        """
        Evaluate detection on a single dataloader.

        Args:
            dataloader: yields (images, labels) or (images, labels, ...)
            attack_label:
                0 -> all samples in this loader are clean
                1 -> all samples in this loader are adversarial

        Returns:
            dict with aggregate predictions, scores, and metrics
        """
        all_true_detection = []
        all_pred_detection = []

        all_class_labels = []
        all_classifier_preds = []

        all_original_confs = []
        all_consistency = []
        all_conf_drops = []
        all_entropy_increase = []
        all_margin_drop = []
        all_prob_variance = []
        all_avg_kl = []
        all_detection_scores = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            else:
                raise ValueError("Dataloader must return tuple/list like (images, labels)")

            images = images.to(self.device)

            batch_size = images.size(0)
            true_detection_labels = torch.full((batch_size,), attack_label, dtype=torch.int64)

            result = self.detector.analyze_sample(images)
            pred_detection = result["is_adversarial"].int().cpu()

            all_true_detection.append(true_detection_labels)
            all_pred_detection.append(pred_detection)

            all_classifier_preds.append(result["original_preds"])
            all_original_confs.append(result["original_confs"])
            all_consistency.append(result["consistency_ratio"])
            all_conf_drops.append(result["avg_confidence_drop"])
            all_entropy_increase.append(result["entropy_increase"])
            all_margin_drop.append(result["margin_drop"])
            all_prob_variance.append(result["prob_variance"])
            all_avg_kl.append(result["avg_kl_divergence"])
            all_detection_scores.append(result["detection_score"])

            if labels is not None:
                all_class_labels.append(labels.detach().cpu())

        all_true_detection = torch.cat(all_true_detection, dim=0)
        all_pred_detection = torch.cat(all_pred_detection, dim=0)

        all_classifier_preds = torch.cat(all_classifier_preds, dim=0)
        all_original_confs = torch.cat(all_original_confs, dim=0)
        all_consistency = torch.cat(all_consistency, dim=0)
        all_conf_drops = torch.cat(all_conf_drops, dim=0)
        all_entropy_increase = torch.cat(all_entropy_increase, dim=0)
        all_margin_drop = torch.cat(all_margin_drop, dim=0)
        all_prob_variance = torch.cat(all_prob_variance, dim=0)
        all_avg_kl = torch.cat(all_avg_kl, dim=0)
        all_detection_scores = torch.cat(all_detection_scores, dim=0)

        metrics = self.compute_metrics(all_true_detection, all_pred_detection)

        output = {
            "true_detection_labels": all_true_detection,
            "pred_detection_labels": all_pred_detection,
            "original_preds": all_classifier_preds,
            "original_confs": all_original_confs,
            "consistency_ratio": all_consistency,
            "avg_confidence_drop": all_conf_drops,
            "entropy_increase": all_entropy_increase,
            "margin_drop": all_margin_drop,
            "prob_variance": all_prob_variance,
            "avg_kl_divergence": all_avg_kl,
            "detection_score": all_detection_scores,
            "metrics": metrics,
            "num_flagged": int(all_pred_detection.sum().item()),
            "num_samples": int(all_pred_detection.shape[0]),
            "flag_rate": float(all_pred_detection.float().mean().item()),
            "avg_detection_score": float(all_detection_scores.mean().item()),
        }

        if len(all_class_labels) > 0:
            all_class_labels = torch.cat(all_class_labels, dim=0)
            output["class_labels"] = all_class_labels
            output["classifier_accuracy"] = float(
                (all_classifier_preds == all_class_labels).float().mean().item()
            )

        return output

    def evaluate_clean_and_adv_loaders(self, clean_loader, adv_loader):
        """
        Evaluate detection on separate clean and adversarial dataloaders,
        then combine the results.

        Args:
            clean_loader: dataloader of clean samples
            adv_loader: dataloader of adversarial samples

        Returns:
            dict with:
            - clean-only metrics
            - adversarial-only metrics
            - combined metrics
        """
        clean_results = self.evaluate_loader(clean_loader, attack_label=0)
        adv_results = self.evaluate_loader(adv_loader, attack_label=1)

        combined_true = torch.cat(
            [clean_results["true_detection_labels"], adv_results["true_detection_labels"]],
            dim=0,
        )
        combined_pred = torch.cat(
            [clean_results["pred_detection_labels"], adv_results["pred_detection_labels"]],
            dim=0,
        )

        combined_original_preds = torch.cat(
            [clean_results["original_preds"], adv_results["original_preds"]],
            dim=0,
        )
        combined_original_confs = torch.cat(
            [clean_results["original_confs"], adv_results["original_confs"]],
            dim=0,
        )
        combined_consistency = torch.cat(
            [clean_results["consistency_ratio"], adv_results["consistency_ratio"]],
            dim=0,
        )
        combined_conf_drops = torch.cat(
            [clean_results["avg_confidence_drop"], adv_results["avg_confidence_drop"]],
            dim=0,
        )
        combined_entropy_increase = torch.cat(
            [clean_results["entropy_increase"], adv_results["entropy_increase"]],
            dim=0,
        )
        combined_margin_drop = torch.cat(
            [clean_results["margin_drop"], adv_results["margin_drop"]],
            dim=0,
        )
        combined_prob_variance = torch.cat(
            [clean_results["prob_variance"], adv_results["prob_variance"]],
            dim=0,
        )
        combined_avg_kl = torch.cat(
            [clean_results["avg_kl_divergence"], adv_results["avg_kl_divergence"]],
            dim=0,
        )
        combined_detection_scores = torch.cat(
            [clean_results["detection_score"], adv_results["detection_score"]],
            dim=0,
        )

        combined_metrics = self.compute_metrics(combined_true, combined_pred)

        combined_output = {
            "true_detection_labels": combined_true,
            "pred_detection_labels": combined_pred,
            "original_preds": combined_original_preds,
            "original_confs": combined_original_confs,
            "consistency_ratio": combined_consistency,
            "avg_confidence_drop": combined_conf_drops,
            "entropy_increase": combined_entropy_increase,
            "margin_drop": combined_margin_drop,
            "prob_variance": combined_prob_variance,
            "avg_kl_divergence": combined_avg_kl,
            "detection_score": combined_detection_scores,
            "metrics": combined_metrics,
            "clean_results": clean_results,
            "adv_results": adv_results,
            "num_flagged": int(combined_pred.sum().item()),
            "num_samples": int(combined_pred.shape[0]),
            "flag_rate": float(combined_pred.float().mean().item()),
            "avg_detection_score": float(combined_detection_scores.mean().item()),
        }

        if "class_labels" in clean_results and "class_labels" in adv_results:
            combined_class_labels = torch.cat(
                [clean_results["class_labels"], adv_results["class_labels"]],
                dim=0,
            )
            combined_output["class_labels"] = combined_class_labels
            combined_output["classifier_accuracy"] = float(
                (combined_original_preds == combined_class_labels).float().mean().item()
            )

        return combined_output

    def print_metrics(self, metrics, title="Detection Metrics"):
        """
        Nicely print metrics dictionary.
        """
        print(f"\n{title}")
        print("-" * len(title))
        print(f"Accuracy    : {metrics['accuracy']:.4f}")
        print(f"Precision   : {metrics['precision']:.4f}")
        print(f"Recall      : {metrics['recall']:.4f}")
        print(f"Specificity : {metrics['specificity']:.4f}")
        print(f"F1 Score    : {metrics['f1_score']:.4f}")
        print(f"TP          : {metrics['tp']}")
        print(f"TN          : {metrics['tn']}")
        print(f"FP          : {metrics['fp']}")
        print(f"FN          : {metrics['fn']}")

    def print_score_stats(self, results, title="Score Statistics"):
        """
        Print summary statistics for the detector signals.
        """
        print(f"\n{title}")
        print("-" * len(title))

        if "detection_score" in results:
            print(f"Avg Detection Score : {results['detection_score'].float().mean().item():.4f}")
        if "consistency_ratio" in results:
            print(f"Avg Consistency     : {results['consistency_ratio'].float().mean().item():.4f}")
        if "avg_confidence_drop" in results:
            print(f"Avg Confidence Drop : {results['avg_confidence_drop'].float().mean().item():.4f}")
        if "entropy_increase" in results:
            print(f"Avg Entropy Increase: {results['entropy_increase'].float().mean().item():.4f}")
        if "margin_drop" in results:
            print(f"Avg Margin Drop     : {results['margin_drop'].float().mean().item():.4f}")
        if "prob_variance" in results:
            print(f"Avg Prob Variance   : {results['prob_variance'].float().mean().item():.4f}")
        if "avg_kl_divergence" in results:
            print(f"Avg KL Divergence   : {results['avg_kl_divergence'].float().mean().item():.4f}")

        if "num_flagged" in results and "num_samples" in results:
            print(f"Flagged Samples     : {results['num_flagged']}/{results['num_samples']}")
        if "flag_rate" in results:
            print(f"Flag Rate           : {results['flag_rate']:.4f}")

    def summarize_results(self, results, title="Detection Summary"):
        """
        Print a compact summary from evaluate_loader or
        evaluate_clean_and_adv_loaders output.
        """
        print(f"\n{title}")
        print("=" * len(title))

        if "clean_results" in results and "adv_results" in results:
            self.print_metrics(results["clean_results"]["metrics"], title="Clean Loader Metrics")
            self.print_score_stats(results["clean_results"], title="Clean Loader Score Stats")

            self.print_metrics(results["adv_results"]["metrics"], title="Adversarial Loader Metrics")
            self.print_score_stats(results["adv_results"], title="Adversarial Loader Score Stats")

            self.print_metrics(results["metrics"], title="Combined Metrics")
            self.print_score_stats(results, title="Combined Score Stats")
        else:
            self.print_metrics(results["metrics"], title=title)
            self.print_score_stats(results, title=f"{title} Score Stats")

        if "classifier_accuracy" in results:
            print(f"\nClassifier Accuracy: {results['classifier_accuracy']:.4f}")