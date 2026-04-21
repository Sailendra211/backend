# src/detection/image_transforms.py

import torch
import torch.nn.functional as F


class ImageTransforms:
    """
    A collection of lightweight image transformations used for
    prediction-stability-based adversarial detection.

    Expected image tensor shape:
        - single image: (C, H, W)
        - batch: (B, C, H, W)

    Assumes pixel values are in [0, 1].
    """

    def __init__(self, device=None):
        self.device = device

    def _ensure_batch(self, x):
        """
        Convert (C, H, W) to (1, C, H, W) if needed.
        """
        if x.dim() == 3:
            return x.unsqueeze(0), True
        elif x.dim() == 4:
            return x, False
        else:
            raise ValueError(f"Expected input of shape (C,H,W) or (B,C,H,W), got {x.shape}")

    def _restore_shape(self, x, was_single):
        """
        Convert (1, C, H, W) back to (C, H, W) if original input was single image.
        """
        if was_single:
            return x.squeeze(0)
        return x

    def _clamp(self, x):
        return torch.clamp(x, 0.0, 1.0)

    def horizontal_flip(self, x):
        """
        Horizontal flip.
        """
        x, was_single = self._ensure_batch(x)
        x = torch.flip(x, dims=[3])  # flip width dimension
        return self._restore_shape(x, was_single)

    def add_gaussian_noise(self, x, std=0.02):
        """
        Add small Gaussian noise.
        """
        x, was_single = self._ensure_batch(x)
        noise = torch.randn_like(x) * std
        x = self._clamp(x + noise)
        return self._restore_shape(x, was_single)

    def adjust_brightness(self, x, factor=1.1):
        """
        Brightness scaling.
        factor > 1.0 increases brightness
        factor < 1.0 decreases brightness
        """
        x, was_single = self._ensure_batch(x)
        x = self._clamp(x * factor)
        return self._restore_shape(x, was_single)

    def gaussian_blur(self, x, kernel_size=3):
        """
        Simple average blur approximation using avg_pool2d.
        Keeps output same size.
        """
        x, was_single = self._ensure_batch(x)

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        padding = kernel_size // 2
        x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
        x = self._clamp(x)
        return self._restore_shape(x, was_single)

    def jpeg_like_compression(self, x, levels=32):
        """
        Simulate crude compression by quantizing intensity levels.
        Higher levels = less compression.
        """
        x, was_single = self._ensure_batch(x)
        x = torch.round(x * levels) / levels
        x = self._clamp(x)
        return self._restore_shape(x, was_single)

    def resize_recover(self, x, scale_factor=0.9):
        """
        Downscale and then upscale back to original size.
        This simulates mild resampling artifacts.
        """
        x, was_single = self._ensure_batch(x)
        b, c, h, w = x.shape

        new_h = max(1, int(h * scale_factor))
        new_w = max(1, int(w * scale_factor))

        down = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        up = self._clamp(up)

        return self._restore_shape(up, was_single)

    def get_all_transforms(self, x):
        """
        Apply all supported transforms and return them in a dictionary.
        """
        return {
            "original": x,
            "hflip": self.horizontal_flip(x),
            "gaussian_noise": self.add_gaussian_noise(x, std=0.02),
            "brightness": self.adjust_brightness(x, factor=1.1),
            "blur": self.gaussian_blur(x, kernel_size=3),
            "jpeg_like": self.jpeg_like_compression(x, levels=32),
            "resize_recover": self.resize_recover(x, scale_factor=0.9),
        }

    def get_selected_transforms(self, x, transform_names):
        """
        Apply only selected transforms by name.

        Example:
            transform_names = ["gaussian_noise", "blur", "brightness"]
        """
        outputs = {"original": x}

        for name in transform_names:
            if name == "hflip":
                outputs[name] = self.horizontal_flip(x)
            elif name == "gaussian_noise":
                outputs[name] = self.add_gaussian_noise(x, std=0.02)
            elif name == "brightness":
                outputs[name] = self.adjust_brightness(x, factor=1.1)
            elif name == "blur":
                outputs[name] = self.gaussian_blur(x, kernel_size=3)
            elif name == "jpeg_like":
                outputs[name] = self.jpeg_like_compression(x, levels=32)
            elif name == "resize_recover":
                outputs[name] = self.resize_recover(x, scale_factor=0.9)
            else:
                raise ValueError(f"Unknown transform name: {name}")

        return outputs