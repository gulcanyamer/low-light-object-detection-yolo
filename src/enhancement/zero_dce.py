"""Zero-DCE (Zero-Reference Deep Curve Estimation) low-light enhancement.

Architecture matches the official implementation from
Li et al., CVPR 2020: https://github.com/Li-Chongyi/Zero-DCE
"""

import cv2
import numpy as np
import torch
import torch.nn as nn


class ZeroDCENet(nn.Module):
    """Official Zero-DCE architecture (enhance_net_nopool).

    Applies 8 iterations of the learned curve to the input image,
    each iteration split off consecutively from a single 24-channel head.
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # Split into 8 curve maps, applied as 8 consecutive iterations
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        return enhance_image


class ZeroDCEEnhancer:
    """Loads a trained Zero-DCE checkpoint and applies it to BGR images."""

    def __init__(self, checkpoint_path, device="cuda"):
        self.device = device
        self.model = ZeroDCENet().to(device).eval()
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def enhance(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(tensor)
        out_np = out.squeeze().permute(1, 2, 0).cpu().numpy()
        out_np = (out_np * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
