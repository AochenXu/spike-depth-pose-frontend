from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SimpleEncoder(nn.Module):
    """简化版编码器：5 层 conv，下采样 1/32，输出 1024 通道特征。"""

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64, 7, 2)   # 1/2
        self.conv2 = ConvBlock(64, 128, 5, 2)           # 1/4
        self.conv3 = ConvBlock(128, 256, 3, 2)          # 1/8
        self.conv4 = ConvBlock(256, 512, 3, 2)          # 1/16
        self.conv5 = ConvBlock(512, 1024, 3, 2)         # 1/32

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward_features(self, x):
        feats = []
        x = self.conv1(x)
        feats.append(x)
        x = self.conv2(x)
        feats.append(x)
        x = self.conv3(x)
        feats.append(x)
        x = self.conv4(x)
        feats.append(x)
        x = self.conv5(x)
        feats.append(x)
        return feats


class DepthDecoder(nn.Module):
    """Monodepth2-style skip decoder with a single final depth output."""

    def __init__(self, num_ch_enc=(64, 128, 256, 512, 1024), use_skips: bool = True):
        super().__init__()
        self.num_ch_enc = list(num_ch_enc)
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.use_skips = bool(use_skips)

        self.upconv_40 = ConvBlock(self.num_ch_enc[-1], self.num_ch_dec[4])
        self.upconv_41 = ConvBlock(self.num_ch_dec[4] + (self.num_ch_enc[3] if self.use_skips else 0), self.num_ch_dec[4])

        self.upconv_30 = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])
        self.upconv_31 = ConvBlock(self.num_ch_dec[3] + (self.num_ch_enc[2] if self.use_skips else 0), self.num_ch_dec[3])

        self.upconv_20 = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])
        self.upconv_21 = ConvBlock(self.num_ch_dec[2] + (self.num_ch_enc[1] if self.use_skips else 0), self.num_ch_dec[2])

        self.upconv_10 = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])
        self.upconv_11 = ConvBlock(self.num_ch_dec[1] + (self.num_ch_enc[0] if self.use_skips else 0), self.num_ch_dec[1])

        self.upconv_00 = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
        self.upconv_01 = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])
        self.out_conv = nn.Conv2d(self.num_ch_dec[0], 1, 3, 1, 1)

    @staticmethod
    def _depth_from_logits(x):
        inv_min = 1.0 / 80.0
        inv_max = 1.0 / 0.5
        inv_depth = inv_min + (inv_max - inv_min) * torch.sigmoid(x)
        depth = 1.0 / inv_depth
        return torch.clamp(depth, min=0.5, max=80.0)

    def forward(self, feats, input_size):
        if not isinstance(feats, (list, tuple)):
            raise TypeError("DepthDecoder expects a list of encoder features")
        H, W = input_size
        f1, f2, f3, f4, f5 = feats

        x = self.upconv_40(f5)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_skips:
            x = torch.cat([x, f4], dim=1)
        x = self.upconv_41(x)

        x = self.upconv_30(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_skips:
            x = torch.cat([x, f3], dim=1)
        x = self.upconv_31(x)

        x = self.upconv_20(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_skips:
            x = torch.cat([x, f2], dim=1)
        x = self.upconv_21(x)

        x = self.upconv_10(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_skips:
            x = torch.cat([x, f1], dim=1)
        x = self.upconv_11(x)

        x = self.upconv_00(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upconv_01(x)
        x = self.out_conv(x)
        depth = self._depth_from_logits(x)
        return depth[:, :, :H, :W]


class PoseDecoder(nn.Module):
    """从编码特征预测相对位姿 (tx,ty,tz, rx,ry,rz)。"""

    def __init__(self, in_channels=1024):
        super().__init__()
        self.conv = ConvBlock(in_channels, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 6)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.01)
        nn.init.zeros_(self.fc.bias)
        self.trans_scale = 0.2
        self.rot_scale = 0.2

    def forward(self, feat):
        x = self.conv(feat)
        x = self.pool(x).view(x.size(0), -1)
        raw = torch.tanh(self.fc(x))
        trans = raw[:, :3] * self.trans_scale
        rot = raw[:, 3:] * self.rot_scale
        return torch.cat([trans, rot], dim=1)


class PairPoseNet(nn.Module):
    """使用双帧拼接输入预测相对位姿。"""

    def __init__(self, in_channels=6):
        super().__init__()
        self.encoder = SimpleEncoder(in_channels=in_channels)
        self.decoder = PoseDecoder(in_channels=1024)

    def forward(self, img_ref: torch.Tensor, img_src: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([img_ref, img_src], dim=1)
        feat = self.encoder(pair)
        return self.decoder(feat)

    @torch.no_grad()
    def init_from_ann_encoder(self, ann_encoder: SimpleEncoder) -> None:
        """Use a trained RGB encoder to initialize the pairwise pose encoder.

        The first convolution is expanded from 3 to 6 input channels by
        splitting the original RGB weights evenly across the reference/source
        image channels, which keeps the activation scale roughly unchanged.
        """

        ann_state = ann_encoder.state_dict()
        pair_state = self.encoder.state_dict()
        for key, value in pair_state.items():
            if key == "conv1.conv.weight":
                ann_w = ann_state[key]
                expanded = torch.cat([ann_w * 0.5, ann_w * 0.5], dim=1)
                value.copy_(expanded)
            elif key in ann_state and ann_state[key].shape == value.shape:
                value.copy_(ann_state[key])

        self.encoder.load_state_dict(pair_state)


class MonoDepthSNN_RGB(nn.Module):
    """第一阶段：先用普通 RGB encoder + decoder 跑通自监督单目深度/位姿。"""

    def __init__(self):
        super().__init__()
        self.encoder = SimpleEncoder(in_channels=3)
        self.depth_decoder = DepthDecoder(num_ch_enc=(64, 128, 256, 512, 1024))
        self.pose_decoder = PoseDecoder(in_channels=1024)
        self.pair_pose_net = PairPoseNet(in_channels=6)

    def forward(self, img):
        """img: [B,3,H,W]  返回 depth, pose (t->t+1)。"""
        B, _, H, W = img.shape
        feats = self.encoder.forward_features(img)
        feat = feats[-1]
        depth = self.depth_decoder(feats, (H, W))
        pose = self.pose_decoder(feat)
        return depth, pose

    def predict_pose_pair(self, img_ref: torch.Tensor, img_src: torch.Tensor) -> torch.Tensor:
        return self.pair_pose_net(img_ref, img_src)

    def init_pair_pose_from_ann_encoder(self, ann_encoder: SimpleEncoder) -> None:
        self.pair_pose_net.init_from_ann_encoder(ann_encoder)


# ===================== SNN 版本编码器 + 权重映射 =====================


class LIFNeuron(nn.Module):
    """简化 LIF 神经元，用于 SNN 编码器。

    注意：这里只实现了一个最简单的有状态单步版本，适合作为
    "ANN 权重映射后 + 轻微微调" 的起点。
    """

    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0, learnable_threshold: bool = False):
        super().__init__()
        self.tau = float(tau)
        self.v_reset = float(v_reset)
        threshold_tensor = torch.tensor(float(v_threshold))
        if learnable_threshold:
            self.v_threshold = nn.Parameter(threshold_tensor)
        else:
            self.register_buffer("v_threshold", threshold_tensor)
        self.register_buffer("v", torch.tensor(0.0), persistent=False)
        self.last_spike_rate = 0.0

    def reset_state(self):
        self.v = torch.tensor(0.0, device=self.v.device)
        self.last_spike_rate = 0.0

    def forward(self, x):
        if self.v.ndim == 0 or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        alpha = max(0.0, 1.0 - 1.0 / max(1.0, self.tau))
        self.v = self.v * alpha + x
        threshold = torch.clamp(self.v_threshold, min=1e-3)
        spike = (self.v >= threshold).float()
        self.last_spike_rate = float(spike.mean().detach().item())
        v_post = self.v * (1.0 - spike) + self.v_reset * spike
        out = spike + v_post * (1.0 - spike)
        self.v = v_post.detach()
        return out.detach() + x - x.detach()

    def get_threshold(self) -> float:
        return float(torch.clamp(self.v_threshold.detach(), min=1e-3).item())


def rate_encode(img: torch.Tensor, num_steps: int) -> torch.Tensor:
    img = torch.clamp(img, 0.0, 1.0)
    if num_steps <= 1:
        return img.unsqueeze(1)
    rand = torch.rand(
        img.size(0),
        num_steps,
        img.size(1),
        img.size(2),
        img.size(3),
        device=img.device,
        dtype=img.dtype,
    )
    return (rand < img.unsqueeze(1)).float()


class SNNConvBlock(nn.Module):
    """与 ConvBlock 拓扑对齐的脉冲卷积块，用于 ANN→SNN 权重拷贝。"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, tau=2.0, v_threshold=1.0, learnable_threshold: bool = False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = LIFNeuron(tau=tau, v_threshold=v_threshold, v_reset=0.0, learnable_threshold=learnable_threshold)
        self.last_spike_rate = 0.0

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lif(x)
        self.last_spike_rate = self.lif.last_spike_rate
        return x


class SNNEncoderFromANN(nn.Module):
    """结构上与 SimpleEncoder 对齐的 SNN 编码器，用于加载 ANN 预训练权重。

    使用方式：
      ann_enc = SimpleEncoder(...)
      # 1) 在深度任务上预训练 ann_enc
      # 2) state = ann_enc.state_dict()
      snn_enc = SNNEncoderFromANN(in_channels=3)
      snn_enc.load_from_ann(state)
    """

    def __init__(self, in_channels=3, tau=2.0, v_threshold=1.0, learnable_threshold: bool = False):
        super().__init__()
        self.conv1 = SNNConvBlock(in_channels, 64, 7, 2, tau, v_threshold, learnable_threshold)
        self.conv2 = SNNConvBlock(64, 128, 5, 2, tau, v_threshold, learnable_threshold)
        self.conv3 = SNNConvBlock(128, 256, 3, 2, tau, v_threshold, learnable_threshold)
        self.conv4 = SNNConvBlock(256, 512, 3, 2, tau, v_threshold, learnable_threshold)
        self.conv5 = SNNConvBlock(512, 1024, 3, 2, tau, v_threshold, learnable_threshold)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward_features(self, x):
        feats = []
        x = self.conv1(x)
        feats.append(x)
        x = self.conv2(x)
        feats.append(x)
        x = self.conv3(x)
        feats.append(x)
        x = self.conv4(x)
        feats.append(x)
        x = self.conv5(x)
        feats.append(x)
        return feats

    def reset_state(self):
        for module in self.modules():
            if isinstance(module, LIFNeuron):
                module.reset_state()

    def get_spike_stats(self) -> Dict[str, float]:
        return {
            "conv1": self.conv1.last_spike_rate,
            "conv2": self.conv2.last_spike_rate,
            "conv3": self.conv3.last_spike_rate,
            "conv4": self.conv4.last_spike_rate,
            "conv5": self.conv5.last_spike_rate,
        }

    def get_threshold_stats(self) -> Dict[str, float]:
        return {
            "conv1": self.conv1.lif.get_threshold(),
            "conv2": self.conv2.lif.get_threshold(),
            "conv3": self.conv3.lif.get_threshold(),
            "conv4": self.conv4.lif.get_threshold(),
            "conv5": self.conv5.lif.get_threshold(),
        }

    @torch.no_grad()
    def load_from_ann(self, ann_state_dict: dict):
        """从 ANN(SimpleEncoder) 的 state_dict 中拷贝卷积和 BN 参数。

        规则：
          convX.conv.weight/bias -> convX.conv.weight/bias
          convX.bn.*            -> convX.bn.*
        LIF 参数 (tau, v_threshold, v_reset) 使用当前初始化，不做映射。
        """

        mapping = [
            ("conv1", self.conv1),
            ("conv2", self.conv2),
            ("conv3", self.conv3),
            ("conv4", self.conv4),
            ("conv5", self.conv5),
        ]

        for prefix, snn_block in mapping:
            # 卷积
            for name in ["weight", "bias"]:
                key = f"{prefix}.conv.{name}"
                if key in ann_state_dict and getattr(snn_block.conv, name) is not None:
                    getattr(snn_block.conv, name).copy_(ann_state_dict[key])

            # BN
            for name in ["weight", "bias", "running_mean", "running_var"]:
                key = f"{prefix}.bn.{name}"
                if key in ann_state_dict and hasattr(snn_block.bn, name):
                    getattr(snn_block.bn, name).copy_(ann_state_dict[key])

        return self


class MonoDepthSNN_Spike(nn.Module):
    """使用 ANN 预训练权重初始化的 SNN 编码器 + 原深度/位姿头。"""

    def __init__(self, tau=2.0, time_steps=4, v_threshold=0.25, input_encoding="rate", learnable_threshold: bool = False):
        super().__init__()
        self.encoder = SNNEncoderFromANN(
            in_channels=3,
            tau=tau,
            v_threshold=v_threshold,
            learnable_threshold=learnable_threshold,
        )
        self.depth_decoder = DepthDecoder(num_ch_enc=(64, 128, 256, 512, 1024))
        self.pose_decoder = PoseDecoder(in_channels=1024)
        self.pair_pose_net = PairPoseNet(in_channels=6)
        self.time_steps = int(time_steps)
        self.input_encoding = str(input_encoding)
        self.last_spike_stats: Dict[str, float] = {}
        self.last_avg_spike_rate = 0.0

    def init_from_ann_encoder(self, ann_encoder: SimpleEncoder):
        """从一个已训练好的 SimpleEncoder 拷贝权重到 SNN encoder。"""
        state = ann_encoder.state_dict()
        self.encoder.load_from_ann(state)
        self.pair_pose_net.init_from_ann_encoder(ann_encoder)

    def _forward_single_step(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = img.shape
        feats = self.encoder.forward_features(img)
        feat = feats[-1]
        depth = self.depth_decoder(feats, (H, W))
        pose = self.pose_decoder(feat)
        return depth, pose

    def forward_features(self, img: torch.Tensor, num_steps: Optional[int] = None, input_encoding: Optional[str] = None):
        steps = int(num_steps or self.time_steps)
        encoding = str(input_encoding or self.input_encoding)
        self.encoder.reset_state()

        if encoding == "rate":
            inputs = rate_encode(img, max(1, steps))
        elif encoding == "analog":
            inputs = img.unsqueeze(1).repeat(1, max(1, steps), 1, 1, 1)
        else:
            raise ValueError(f"Unsupported input_encoding: {encoding}")

        feat_sum = None
        depth_sum = None
        pose_sum = None
        step_rates = []
        for step_idx in range(max(1, steps)):
            step_input = inputs[:, step_idx]
            B, _, H, W = step_input.shape
            feats = self.encoder.forward_features(step_input)
            feat = feats[-1]
            depth = self.depth_decoder(feats, (H, W))
            pose = self.pose_decoder(feat)
            feat_sum = feat if feat_sum is None else feat_sum + feat
            depth_sum = depth if depth_sum is None else depth_sum + depth
            pose_sum = pose if pose_sum is None else pose_sum + pose
            stats = self.encoder.get_spike_stats()
            step_rates.extend(list(stats.values()))

        feat = feat_sum / max(1, steps)
        depth = depth_sum / max(1, steps)
        pose = pose_sum / max(1, steps)
        self.last_spike_stats = self.encoder.get_spike_stats()
        self.last_avg_spike_rate = float(sum(step_rates) / max(1, len(step_rates)))
        return feat, depth, pose

    def forward(self, img: torch.Tensor, num_steps: Optional[int] = None, input_encoding: Optional[str] = None):
        _, depth, pose = self.forward_features(img, num_steps=num_steps, input_encoding=input_encoding)
        return depth, pose

    def predict_pose_pair(self, img_ref: torch.Tensor, img_src: torch.Tensor) -> torch.Tensor:
        return self.pair_pose_net(img_ref, img_src)

    def get_spike_stats(self) -> Dict[str, float]:
        stats = dict(self.last_spike_stats)
        stats["avg_spike_rate"] = self.last_avg_spike_rate
        return stats

    def get_threshold_stats(self) -> Dict[str, float]:
        return self.encoder.get_threshold_stats()
