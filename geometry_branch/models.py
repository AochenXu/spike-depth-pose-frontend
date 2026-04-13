from typing import Dict, Optional, Sequence, Tuple

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


class DepthDecoder(nn.Module):
    """单尺度深度解码器，输出与输入同分辨率的深度图。"""

    def __init__(
        self,
        in_channels=1024,
        channel_scale: float = 1.0,
        min_depth: float = 0.5,
        max_depth: float = 80.0,
    ):
        super().__init__()
        scale = max(0.25, float(channel_scale))
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        if self.max_depth <= self.min_depth:
            raise ValueError("max_depth must be greater than min_depth")

        def scaled(base: int) -> int:
            value = int(round(base * scale / 8.0) * 8)
            return max(16, value)

        ch1 = scaled(512)
        ch2 = scaled(256)
        ch3 = scaled(128)
        ch4 = scaled(64)
        ch5 = scaled(32)
        self.channel_scale = scale
        self.up1 = ConvBlock(in_channels, ch1)
        self.up2 = ConvBlock(ch1, ch2)
        self.up3 = ConvBlock(ch2, ch3)
        self.up4 = ConvBlock(ch3, ch4)
        self.up5 = ConvBlock(ch4, ch5)
        self.out_conv = nn.Conv2d(ch5, 1, 3, 1, 1)

    def forward(self, feat, input_size):
        H, W = input_size
        x = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up4(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up5(x)
        x = self.out_conv(x)

        inv_min = 1.0 / self.max_depth
        inv_max = 1.0 / self.min_depth
        inv_depth = inv_min + (inv_max - inv_min) * torch.sigmoid(x)
        depth = 1.0 / inv_depth
        depth = torch.clamp(depth, min=self.min_depth, max=self.max_depth)
        return depth[:, :, :H, :W]


class ShallowStaticFusion(nn.Module):
    """Lightweight shallow RGB branch with gated fusion into the SNN trunk."""

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 16, 7, 2)
        self.conv2 = ConvBlock(16, 32, 5, 2)
        self.proj1 = nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0)
        self.proj2 = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0)
        self.gate1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.gate2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        return self.proj1(x1), self.proj2(x2)

    def fuse(self, snn_x1: torch.Tensor, snn_x2: torch.Tensor, img: torch.Tensor, fusion_weight: float) -> Tuple[torch.Tensor, torch.Tensor]:
        static_x1, static_x2 = self.forward(img)
        gate1 = torch.sigmoid(self.gate1(torch.cat([snn_x1, static_x1], dim=1)))
        fused_x1 = snn_x1 + float(fusion_weight) * gate1 * static_x1
        gate2 = torch.sigmoid(self.gate2(torch.cat([snn_x2, static_x2], dim=1)))
        fused_x2 = snn_x2 + float(fusion_weight) * gate2 * static_x2
        return fused_x1, fused_x2

    @torch.no_grad()
    def init_from_ann_encoder(self, ann_encoder: SimpleEncoder) -> None:
        ann_state = ann_encoder.state_dict()
        own_state = self.state_dict()
        own_state["conv1.conv.weight"].copy_(ann_state["conv1.conv.weight"][:16])
        if own_state["conv1.conv.bias"] is not None and ann_state["conv1.conv.bias"] is not None:
            own_state["conv1.conv.bias"].copy_(ann_state["conv1.conv.bias"][:16])
        for name in ["weight", "bias", "running_mean", "running_var"]:
            own_state[f"conv1.bn.{name}"].copy_(ann_state[f"conv1.bn.{name}"][:16])
        own_state["conv2.conv.weight"].copy_(ann_state["conv2.conv.weight"][:32, :16])
        if own_state["conv2.conv.bias"] is not None and ann_state["conv2.conv.bias"] is not None:
            own_state["conv2.conv.bias"].copy_(ann_state["conv2.conv.bias"][:32])
        for name in ["weight", "bias", "running_mean", "running_var"]:
            own_state[f"conv2.bn.{name}"].copy_(ann_state[f"conv2.bn.{name}"][:32])
        for name in ["proj1", "proj2"]:
            nn.init.xavier_uniform_(getattr(self, name).weight, gain=0.1)
            nn.init.zeros_(getattr(self, name).bias)
        for name in ["gate1", "gate2"]:
            nn.init.zeros_(getattr(self, name).weight)
            nn.init.zeros_(getattr(self, name).bias)
        self.load_state_dict(own_state)


class PoseDecoder(nn.Module):
    """从编码特征预测相对位姿 (tx,ty,tz, rx,ry,rz)。"""

    def __init__(self, in_channels=1024, hidden_channels: int = 256, mlp_hidden: int = 128):
        super().__init__()
        hidden_channels = max(64, int(hidden_channels))
        mlp_hidden = max(32, int(mlp_hidden))
        self.conv1 = ConvBlock(in_channels, hidden_channels)
        self.conv2 = ConvBlock(hidden_channels, hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(hidden_channels, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, 6)
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)
        nn.init.zeros_(self.fc2.bias)
        self.trans_scale = 0.2
        self.rot_scale = 0.2

    def forward(self, feat):
        x = self.conv1(feat)
        x = self.conv2(x)
        x = self.pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        raw = torch.tanh(self.fc2(x))
        trans = raw[:, :3] * self.trans_scale
        rot = raw[:, 3:] * self.rot_scale
        return torch.cat([trans, rot], dim=1)


class PairPoseNet(nn.Module):
    """使用双帧拼接输入预测相对位姿。"""

    def __init__(
        self,
        in_channels=6,
        use_diff_channel: bool = False,
        pose_hidden_channels: int = 256,
        pose_mlp_hidden: int = 128,
        input_normalization: bool = False,
    ):
        super().__init__()
        self.use_diff_channel = bool(use_diff_channel)
        self.input_normalization = bool(input_normalization)
        self.encoder = SimpleEncoder(in_channels=in_channels)
        self.decoder = PoseDecoder(
            in_channels=1024,
            hidden_channels=pose_hidden_channels,
            mlp_hidden=pose_mlp_hidden,
        )

    @staticmethod
    def _normalize_image(img: torch.Tensor) -> torch.Tensor:
        mean = img.mean(dim=(2, 3), keepdim=True)
        std = img.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        return (img - mean) / std

    def forward(self, img_ref: torch.Tensor, img_src: torch.Tensor) -> torch.Tensor:
        if self.input_normalization:
            img_ref = self._normalize_image(img_ref)
            img_src = self._normalize_image(img_src)
        pair_inputs = [img_ref, img_src]
        if self.use_diff_channel:
            pair_inputs.append(torch.abs(img_ref - img_src))
        pair = torch.cat(pair_inputs, dim=1)
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
                if self.use_diff_channel:
                    expanded = torch.cat([ann_w / 3.0, ann_w / 3.0, ann_w / 3.0], dim=1)
                else:
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
        self.depth_decoder = DepthDecoder(in_channels=1024)
        self.pose_decoder = PoseDecoder(in_channels=1024)
        self.pair_pose_net = PairPoseNet(in_channels=6)

    def forward(self, img):
        """img: [B,3,H,W]  返回 depth, pose (t->t+1)。"""
        B, _, H, W = img.shape
        feat = self.encoder(img)
        depth = self.depth_decoder(feat, (H, W))
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

    def __init__(
        self,
        tau=2.0,
        v_threshold=1.0,
        v_reset=0.0,
        learnable_threshold: bool = False,
        output_mode: str = "mixed",
    ):
        super().__init__()
        self.tau = float(tau)
        self.v_reset = float(v_reset)
        self.output_mode = str(output_mode)
        threshold_tensor = torch.tensor(float(v_threshold))
        if learnable_threshold:
            self.v_threshold = nn.Parameter(threshold_tensor)
        else:
            self.register_buffer("v_threshold", threshold_tensor)
        self.register_buffer("v", torch.tensor(0.0), persistent=False)
        self.last_spike_rate = 0.0
        self.last_spike_tensor = None

    def reset_state(self):
        self.v = torch.tensor(0.0, device=self.v.device)
        self.last_spike_rate = 0.0
        self.last_spike_tensor = None

    def forward(self, x):
        if self.v.ndim == 0 or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        alpha = max(0.0, 1.0 - 1.0 / max(1.0, self.tau))
        self.v = self.v * alpha + x
        threshold = torch.clamp(self.v_threshold, min=1e-3)
        spike = (self.v >= threshold).float()
        self.last_spike_rate = float(spike.mean().detach().item())
        self.last_spike_tensor = spike.detach()
        v_post = self.v * (1.0 - spike) + self.v_reset * spike
        if self.output_mode == "spike":
            out = spike
        elif self.output_mode == "membrane":
            out = v_post
        else:
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


def latency_encode(img: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Time-to-first-spike coding for static RGB frames.

    Higher intensity pixels fire earlier. Each location emits at most one spike
    across the temporal window, which makes early steps much sparser than
    repeating dense analog frames.
    """

    img = torch.clamp(img, 0.0, 1.0)
    if num_steps <= 1:
        return (img > 0.5).float().unsqueeze(1)
    spike_time = ((1.0 - img) * float(num_steps - 1)).round().long()
    spike_time = torch.clamp(spike_time, min=0, max=num_steps - 1)
    valid = img > 1e-4
    spikes = []
    for step in range(num_steps):
        spikes.append(((spike_time == step) & valid).float())
    return torch.stack(spikes, dim=1)


def delta_latency_encode(
    img: torch.Tensor,
    img_prev: torch.Tensor,
    num_steps: int,
    delta_threshold: float = 0.03,
) -> torch.Tensor:
    """Encode temporal changes while retaining current-frame spatial support.

    The spike timing is determined by |img - img_prev| and the emitted spike
    amplitude follows the current RGB value, so the sequence carries both
    temporal saliency and current spatial appearance.
    """

    img = torch.clamp(img, 0.0, 1.0)
    img_prev = torch.clamp(img_prev, 0.0, 1.0)
    diff = torch.abs(img - img_prev)
    if num_steps <= 1:
        return (img * (diff > delta_threshold).float()).unsqueeze(1)
    diff_norm = diff / (diff.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    spike_time = ((1.0 - diff_norm) * float(num_steps - 1)).round().long()
    spike_time = torch.clamp(spike_time, min=0, max=num_steps - 1)
    valid = diff > delta_threshold
    spikes = []
    for step in range(num_steps):
        spikes.append(img * ((spike_time == step) & valid).float())
    return torch.stack(spikes, dim=1)


def delta_latency_anchor_encode(
    img: torch.Tensor,
    img_prev: torch.Tensor,
    num_steps: int,
    delta_threshold: float = 0.03,
    anchor_weight: float = 0.2,
) -> torch.Tensor:
    """Motion-aware delta latency with a weak static anchor on the final step.

    Pure delta coding can wash out large static regions because only changed
    pixels spike. We keep the temporal event stream intact and inject a single
    low-amplitude anchor frame at the last step so the network still sees a
    coarse snapshot of the current scene layout.
    """

    spikes = delta_latency_encode(
        img=img,
        img_prev=img_prev,
        num_steps=num_steps,
        delta_threshold=delta_threshold,
    )
    if spikes.size(1) <= 0:
        return spikes
    anchor = torch.clamp(img, 0.0, 1.0) * max(0.0, float(anchor_weight))
    spikes[:, -1] = torch.clamp(spikes[:, -1] + anchor, 0.0, 1.0)
    return spikes


def normalize_sparse_layers(sparse_layers: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if sparse_layers is None:
        return ()
    if isinstance(sparse_layers, str):
        items = [item.strip() for item in sparse_layers.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in sparse_layers if str(item).strip()]
    valid = {"conv1", "conv2", "conv3", "conv4", "conv5"}
    return tuple(item for item in items if item in valid)


class SNNConvBlock(nn.Module):
    """与 ConvBlock 拓扑对齐的脉冲卷积块，用于 ANN→SNN 权重拷贝。"""

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        tau=2.0,
        v_threshold=1.0,
        learnable_threshold: bool = False,
        lif_output_mode: str = "mixed",
        sparse_exec: bool = False,
        sparse_activity_threshold: float = 0.05,
        sparse_fallback_ratio: float = 0.35,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.lif = LIFNeuron(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=0.0,
            learnable_threshold=learnable_threshold,
            output_mode=lif_output_mode,
        )
        self.last_spike_rate = 0.0
        self.sparse_exec = bool(sparse_exec)
        self.sparse_activity_threshold = float(sparse_activity_threshold)
        self.sparse_fallback_ratio = float(sparse_fallback_ratio)
        self.last_active_ratio = 1.0
        self.last_used_sparse = 0.0

    def compute_output_active_mask(self, x: torch.Tensor) -> torch.Tensor:
        active_in = x.abs().amax(dim=1, keepdim=True) > self.sparse_activity_threshold
        active_out = F.max_pool2d(
            active_in.float(),
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
        )
        return (active_out > 0.5).float()

    def _sparse_conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        active_out = self.compute_output_active_mask(x) > 0.5
        B, _, H_out, W_out = active_out.shape
        active_mask = active_out.view(B, -1)
        active_ratio = float(active_mask.float().mean().item())
        self.last_active_ratio = active_ratio
        if active_ratio <= 0.0:
            self.last_used_sparse = 1.0
            return x.new_zeros((B, self.conv.out_channels, H_out, W_out))
        if active_ratio >= self.sparse_fallback_ratio:
            self.last_used_sparse = 0.0
            return self.conv(x)

        patches = F.unfold(
            x,
            kernel_size=self.conv.kernel_size,
            dilation=self.conv.dilation,
            padding=self.conv.padding,
            stride=self.conv.stride,
        )
        patches = patches.transpose(1, 2).contiguous()
        flat_patches = patches.view(-1, patches.size(-1))
        flat_mask = active_mask.reshape(-1)
        selected = flat_patches[flat_mask]

        weight = self.conv.weight.view(self.conv.out_channels, -1)
        out_selected = selected @ weight.t()
        if self.conv.bias is not None:
            out_selected = out_selected + self.conv.bias.unsqueeze(0)

        out_flat = x.new_zeros((B * H_out * W_out, self.conv.out_channels))
        out_flat[flat_mask] = out_selected
        self.last_used_sparse = 1.0
        return out_flat.view(B, H_out * W_out, self.conv.out_channels).transpose(1, 2).reshape(B, self.conv.out_channels, H_out, W_out)

    def forward(self, x):
        self.last_active_ratio = 1.0
        self.last_used_sparse = 0.0
        if self.sparse_exec:
            x = self._sparse_conv_forward(x)
        else:
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

    def __init__(
        self,
        in_channels=3,
        tau=2.0,
        v_threshold=1.0,
        learnable_threshold: bool = False,
        lif_output_mode: str = "mixed",
        sparse_exec: bool = False,
        sparse_layers: Optional[Sequence[str]] = None,
        sparse_activity_threshold: float = 0.05,
        sparse_fallback_ratio: float = 0.35,
    ):
        super().__init__()
        sparse_layer_set = set(normalize_sparse_layers(sparse_layers))
        block_kwargs = {
            "tau": tau,
            "v_threshold": v_threshold,
            "learnable_threshold": learnable_threshold,
            "lif_output_mode": lif_output_mode,
            "sparse_activity_threshold": sparse_activity_threshold,
            "sparse_fallback_ratio": sparse_fallback_ratio,
        }
        self.conv1 = SNNConvBlock(in_channels, 64, 7, 2, sparse_exec=sparse_exec and "conv1" in sparse_layer_set, **block_kwargs)
        self.conv2 = SNNConvBlock(64, 128, 5, 2, sparse_exec=sparse_exec and "conv2" in sparse_layer_set, **block_kwargs)
        self.conv3 = SNNConvBlock(128, 256, 3, 2, sparse_exec=sparse_exec and "conv3" in sparse_layer_set, **block_kwargs)
        self.conv4 = SNNConvBlock(256, 512, 3, 2, sparse_exec=sparse_exec and "conv4" in sparse_layer_set, **block_kwargs)
        self.conv5 = SNNConvBlock(512, 1024, 3, 2, sparse_exec=sparse_exec and "conv5" in sparse_layer_set, **block_kwargs)
        self.sparse_layer_names = [name for name in ["conv1", "conv2", "conv3", "conv4", "conv5"] if getattr(self, name).sparse_exec]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

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

    def get_sparse_stats(self) -> Dict[str, float]:
        stats = {}
        active_ratios = []
        used_sparse = []
        enabled_active = []
        enabled_used = []
        for name in ["conv1", "conv2", "conv3", "conv4", "conv5"]:
            block = getattr(self, name)
            stats[f"{name}_active_ratio"] = float(block.last_active_ratio)
            stats[f"{name}_used_sparse"] = float(block.last_used_sparse)
            stats[f"{name}_sparse_enabled"] = float(block.sparse_exec)
            active_ratios.append(float(block.last_active_ratio))
            used_sparse.append(float(block.last_used_sparse))
            if block.sparse_exec:
                enabled_active.append(float(block.last_active_ratio))
                enabled_used.append(float(block.last_used_sparse))
        stats["avg_active_ratio_all"] = float(sum(active_ratios) / max(1, len(active_ratios)))
        stats["avg_used_sparse_all"] = float(sum(used_sparse) / max(1, len(used_sparse)))
        stats["avg_active_ratio_enabled"] = float(sum(enabled_active) / max(1, len(enabled_active)))
        stats["avg_used_sparse_enabled"] = float(sum(enabled_used) / max(1, len(enabled_used)))
        stats["avg_active_ratio"] = stats["avg_active_ratio_enabled"] if enabled_active else stats["avg_active_ratio_all"]
        stats["avg_used_sparse"] = stats["avg_used_sparse_enabled"] if enabled_used else stats["avg_used_sparse_all"]
        stats["num_sparse_layers"] = float(len(enabled_active))
        return stats

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

    def __init__(
        self,
        tau=2.0,
        time_steps=4,
        v_threshold=0.25,
        input_encoding="rate",
        learnable_threshold: bool = False,
        lif_output_mode: str = "mixed",
        sparse_exec: bool = False,
        sparse_layers: Optional[Sequence[str]] = None,
        sparse_activity_threshold: float = 0.05,
        sparse_fallback_ratio: float = 0.35,
        delta_anchor_weight: float = 0.2,
        decoder_channel_scale: float = 1.0,
        min_depth: float = 0.5,
        max_depth: float = 80.0,
        pose_hidden_channels: int = 256,
        pose_mlp_hidden: int = 128,
        pose_input_normalization: bool = False,
        hybrid_static_branch: bool = False,
        hybrid_static_weight: float = 0.5,
        hybrid_pose_diff: bool = False,
    ):
        super().__init__()
        self.encoder = SNNEncoderFromANN(
            in_channels=3,
            tau=tau,
            v_threshold=v_threshold,
            learnable_threshold=learnable_threshold,
            lif_output_mode=lif_output_mode,
            sparse_exec=sparse_exec,
            sparse_layers=sparse_layers,
            sparse_activity_threshold=sparse_activity_threshold,
            sparse_fallback_ratio=sparse_fallback_ratio,
        )
        self.hybrid_static_branch = bool(hybrid_static_branch)
        self.hybrid_static_weight = float(hybrid_static_weight)
        self.hybrid_pose_diff = bool(hybrid_pose_diff)
        self.static_encoder = ShallowStaticFusion() if self.hybrid_static_branch else None
        self.decoder_channel_scale = float(decoder_channel_scale)
        self.depth_decoder = DepthDecoder(
            in_channels=1024,
            channel_scale=self.decoder_channel_scale,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        self.pose_decoder = PoseDecoder(
            in_channels=1024,
            hidden_channels=pose_hidden_channels,
            mlp_hidden=pose_mlp_hidden,
        )
        pose_in_channels = 9 if self.hybrid_pose_diff else 6
        self.pair_pose_net = PairPoseNet(
            in_channels=pose_in_channels,
            use_diff_channel=self.hybrid_pose_diff,
            pose_hidden_channels=pose_hidden_channels,
            pose_mlp_hidden=pose_mlp_hidden,
            input_normalization=pose_input_normalization,
        )
        self.time_steps = int(time_steps)
        self.input_encoding = str(input_encoding)
        self.delta_anchor_weight = float(delta_anchor_weight)
        self.last_spike_stats: Dict[str, float] = {}
        self.last_avg_spike_rate = 0.0
        self.last_sparse_stats: Dict[str, float] = {}
        self.last_temporal_tensors: Dict[str, torch.Tensor] = {}

    def init_from_ann_encoder(self, ann_encoder: SimpleEncoder):
        """从一个已训练好的 SimpleEncoder 拷贝权重到 SNN encoder。"""
        state = ann_encoder.state_dict()
        self.encoder.load_from_ann(state)
        if self.static_encoder is not None:
            self.static_encoder.init_from_ann_encoder(ann_encoder)
        self.pair_pose_net.init_from_ann_encoder(ann_encoder)

    @staticmethod
    def _average_stats(stats_list):
        if not stats_list:
            return {}
        keys = set()
        for item in stats_list:
            keys.update(item.keys())
        out = {}
        for key in keys:
            values = [float(item[key]) for item in stats_list if key in item]
            out[key] = float(sum(values) / max(1, len(values)))
        return out

    def _forward_single_step(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = img.shape
        feat = self._forward_encoder_with_hybrid(img, img)
        depth = self.depth_decoder(feat, (H, W))
        pose = self.pose_decoder(feat)
        return depth, pose

    def _forward_encoder_with_hybrid(self, step_input: torch.Tensor, raw_img: torch.Tensor, collect_temporal: bool = False):
        x = self.encoder.conv1(step_input)
        temporal = {}
        if collect_temporal:
            spike_tensor = self.encoder.conv1.lif.last_spike_tensor
            if spike_tensor is None:
                spike_tensor = torch.zeros_like(x)
            temporal = {
                "conv1_response": x.abs().mean(dim=1, keepdim=True),
                "conv1_spike": spike_tensor.mean(dim=1, keepdim=True),
                "conv1_active": self.encoder.conv1.compute_output_active_mask(step_input),
            }
        if self.static_encoder is not None:
            static_x1, static_x2 = self.static_encoder(raw_img)
            gate1 = torch.sigmoid(self.static_encoder.gate1(torch.cat([x, static_x1], dim=1)))
            x = x + self.hybrid_static_weight * gate1 * static_x1
        x = self.encoder.conv2(x)
        if self.static_encoder is not None:
            gate2 = torch.sigmoid(self.static_encoder.gate2(torch.cat([x, static_x2], dim=1)))
            x = x + self.hybrid_static_weight * gate2 * static_x2
        x = self.encoder.conv3(x)
        x = self.encoder.conv4(x)
        x = self.encoder.conv5(x)
        if collect_temporal:
            return x, temporal
        return x

    def forward_features(self, img: torch.Tensor, num_steps: Optional[int] = None, input_encoding: Optional[str] = None):
        steps = int(num_steps or self.time_steps)
        encoding = str(input_encoding or self.input_encoding)
        self.encoder.reset_state()

        if encoding == "rate":
            inputs = rate_encode(img, max(1, steps))
        elif encoding == "latency":
            inputs = latency_encode(img, max(1, steps))
        elif encoding in {"delta_latency", "delta_latency_anchor"}:
            raise ValueError(f"{encoding} requires img_prev and should be called via forward(..., img_prev=...)")
        elif encoding == "analog":
            inputs = img.unsqueeze(1).repeat(1, max(1, steps), 1, 1, 1)
        else:
            raise ValueError(f"Unsupported input_encoding: {encoding}")

        feat_sum = None
        step_rates = []
        sparse_stats_steps = []
        conv1_response_steps = []
        conv1_spike_steps = []
        conv1_active_steps = []
        for step_idx in range(max(1, steps)):
            step_input = inputs[:, step_idx]
            feat, temporal = self._forward_encoder_with_hybrid(step_input, img, collect_temporal=True)
            feat_sum = feat if feat_sum is None else feat_sum + feat
            stats = self.encoder.get_spike_stats()
            step_rates.extend(list(stats.values()))
            sparse_stats_steps.append(self.encoder.get_sparse_stats())
            conv1_response_steps.append(temporal["conv1_response"])
            conv1_spike_steps.append(temporal["conv1_spike"])
            conv1_active_steps.append(temporal["conv1_active"])

        feat = feat_sum / max(1, steps)
        B, _, H, W = img.shape
        depth = self.depth_decoder(feat, (H, W))
        pose = self.pose_decoder(feat)
        self.last_spike_stats = self.encoder.get_spike_stats()
        self.last_avg_spike_rate = float(sum(step_rates) / max(1, len(step_rates)))
        self.last_sparse_stats = self._average_stats(sparse_stats_steps)
        self.last_temporal_tensors = {
            "conv1_response_steps": torch.stack(conv1_response_steps, dim=1),
            "conv1_spike_steps": torch.stack(conv1_spike_steps, dim=1),
            "conv1_active_steps": torch.stack(conv1_active_steps, dim=1),
        }
        return feat, depth, pose

    def forward(
        self,
        img: torch.Tensor,
        num_steps: Optional[int] = None,
        input_encoding: Optional[str] = None,
        img_prev: Optional[torch.Tensor] = None,
    ):
        encoding = str(input_encoding or self.input_encoding)
        if encoding in {"delta_latency", "delta_latency_anchor"}:
            if img_prev is None:
                raise ValueError(f"{encoding} encoding requires img_prev")
            steps = int(num_steps or self.time_steps)
            self.encoder.reset_state()
            if encoding == "delta_latency_anchor":
                inputs = delta_latency_anchor_encode(
                    img,
                    img_prev,
                    max(1, steps),
                    anchor_weight=self.delta_anchor_weight,
                )
            else:
                inputs = delta_latency_encode(img, img_prev, max(1, steps))
            feat_sum = None
            step_rates = []
            sparse_stats_steps = []
            conv1_response_steps = []
            conv1_spike_steps = []
            conv1_active_steps = []
            for step_idx in range(max(1, steps)):
                step_input = inputs[:, step_idx]
                feat, temporal = self._forward_encoder_with_hybrid(step_input, img, collect_temporal=True)
                feat_sum = feat if feat_sum is None else feat_sum + feat
                step_rates.extend(list(self.encoder.get_spike_stats().values()))
                sparse_stats_steps.append(self.encoder.get_sparse_stats())
                conv1_response_steps.append(temporal["conv1_response"])
                conv1_spike_steps.append(temporal["conv1_spike"])
                conv1_active_steps.append(temporal["conv1_active"])
            B, _, H, W = img.shape
            feat = feat_sum / max(1, steps)
            depth = self.depth_decoder(feat, (H, W))
            pose = self.pose_decoder(feat)
            self.last_spike_stats = self.encoder.get_spike_stats()
            self.last_avg_spike_rate = float(sum(step_rates) / max(1, len(step_rates)))
            self.last_sparse_stats = self._average_stats(sparse_stats_steps)
            self.last_temporal_tensors = {
                "conv1_response_steps": torch.stack(conv1_response_steps, dim=1),
                "conv1_spike_steps": torch.stack(conv1_spike_steps, dim=1),
                "conv1_active_steps": torch.stack(conv1_active_steps, dim=1),
            }
            return depth, pose
        _, depth, pose = self.forward_features(img, num_steps=num_steps, input_encoding=encoding)
        return depth, pose

    def predict_pose_pair(self, img_ref: torch.Tensor, img_src: torch.Tensor) -> torch.Tensor:
        return self.pair_pose_net(img_ref, img_src)

    def get_spike_stats(self) -> Dict[str, float]:
        stats = dict(self.last_spike_stats)
        stats["avg_spike_rate"] = self.last_avg_spike_rate
        return stats

    def get_threshold_stats(self) -> Dict[str, float]:
        return self.encoder.get_threshold_stats()

    def get_sparse_stats(self) -> Dict[str, float]:
        return dict(self.last_sparse_stats)

    def get_temporal_response_tensors(self) -> Dict[str, torch.Tensor]:
        return dict(self.last_temporal_tensors)
