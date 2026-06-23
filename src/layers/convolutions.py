import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum


def get_pad(k, d=1):
    """
    计算 Padding 以保持特征图尺寸不变 (Same Padding)。
    支持普通卷积和膨胀卷积。
    有效核尺寸 k_eff = k + (k-1)*(d-1)
    """
    if isinstance(k, int):
        k_eff = k + (k - 1) * (d - 1)
        return k_eff // 2
    
    # 处理 tuple 情况，例如 (1, 3)
    d_h, d_w = (d, d) if isinstance(d, int) else d
    k_h, k_w = k
    
    k_h_eff = k_h + (k_h - 1) * (d_h - 1)
    k_w_eff = k_w + (k_w - 1) * (d_w - 1)
    return (k_h_eff // 2, k_w_eff // 2)

class EncBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,  # [新增参数] 默认为1，兼容旧代码
        norm_kwargs={},
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        bias=True,
        residual=False,
    ):
        super().__init__()
        self.use_conv_shortcut = use_conv_shortcut
        self.norm1 = norm_fn(**norm_kwargs)
        
        # 动态计算 Padding
        pad = get_pad(kernel_size, dilation)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation, # 应用 Dilation
            padding_mode=pad_mode,
            bias=bias,
        )
        self.norm2 = norm_fn(**norm_kwargs)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation, # 应用 Dilation
            padding_mode=pad_mode,
            bias=bias,
        )
        self.activation_fn = activation_fn()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                padding_mode=pad_mode,
                bias=bias,
            )
        self.residual = residual

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.use_conv_shortcut or residual.shape != x.shape:
            residual = self.shortcut(residual)
        if self.residual:
            return x + residual
        return x


def encoder(in_dim, hidden_dim, kernel_size=1, ks_res=1, dilation=1, num_layers=2, bias=True, num_groups=8, residual=False):
    """
    Encoder 工厂函数，增加了 dilation 参数支持
    """
    # 计算第一层升维卷积的 padding
    stem_pad = get_pad(kernel_size, dilation)
    
    return nn.Sequential(
        nn.Conv2d(
            in_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=stem_pad,
            dilation=dilation,
            padding_mode="reflect",
            bias=bias,
        ),
        *[
            EncBlock(
                hidden_dim,
                hidden_dim,
                kernel_size=ks_res,
                dilation=dilation, # 传递 dilation
                pad_mode="reflect",
                norm_fn=nn.GroupNorm,
                norm_kwargs={"num_groups": num_groups, "num_channels": hidden_dim},
                activation_fn=nn.SiLU,
                use_conv_shortcut=False,
                bias=bias,
                residual=residual,
            )
            for _ in range(num_layers)
        ],
    )


# ================= Original Implementations =================

class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        heads_rope=1,
        use_encoder=True,
        rope_base=None,
        rope_rescale=None,
        img_layers=2,
    ):
        super().__init__()
        self.use_encoder = use_encoder
        self.out_channels = out_channels
        
        # Calculate channels for each branch
        weights = [4, 4, 4, 4]
        total_weight = sum(weights)
        self.branch_channels = []
        accumulated = 0
        for i in range(3):
            c = (out_channels * weights[i]) // total_weight
            self.branch_channels.append(c)
            accumulated += c
        self.branch_channels.append(out_channels - accumulated)
            
        # Branch 1: 1x1 Encoder
        self.enc_1x1 = encoder(in_channels, self.branch_channels[0], kernel_size=1, ks_res=1, num_layers=img_layers)
        
        # Branch 2: 1x3 Encoder
        self.enc_1x3 = encoder(in_channels, self.branch_channels[1], kernel_size=(1, 3), ks_res=(1, 3), num_layers=img_layers)
        
        # Branch 3: 3x1 Encoder
        self.enc_3x1 = encoder(in_channels, self.branch_channels[2], kernel_size=(3, 1), ks_res=(3, 1), num_layers=img_layers)
        
        # Branch 4: 3x3 Encoder
        self.enc_3x3 = encoder(in_channels, self.branch_channels[3], kernel_size=3, ks_res=3, num_layers=img_layers)

    def forward_encoder(self, x):
        if self.use_encoder:
            x_1x1 = self.enc_1x1(x)
            x_1x3 = self.enc_1x3(x)
            x_3x1 = self.enc_3x1(x)
            x_3x3 = self.enc_3x3(x)
            x = torch.cat([x_1x1, x_1x3, x_3x1, x_3x3], dim=1)
        return x

    def forward(self, x):
        x = self.forward_encoder(x)
        return x

class ParallelImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        heads_rope=1,
        use_encoder=True,
        rope_base=None,
        rope_rescale=None,
        img_layers=2,
    ):
        super().__init__()
        self.use_encoder = use_encoder
        self.out_channels = out_channels
        
        weights = [4, 4, 4, 4]
        total_weight = sum(weights)
        self.branch_channels = []
        accumulated = 0
        for i in range(3):
            c = (out_channels * weights[i]) // total_weight
            self.branch_channels.append(c)
            accumulated += c
        self.branch_channels.append(out_channels - accumulated)
            
        self.enc_1x1 = encoder(in_channels, self.branch_channels[0], kernel_size=1, ks_res=1, num_layers=2)
        self.enc_1x3 = encoder(in_channels, self.branch_channels[1], kernel_size=(1, 3), ks_res=(1, 3), num_layers=1)
        self.enc_3x1 = encoder(in_channels, self.branch_channels[2], kernel_size=(3, 1), ks_res=(3, 1), num_layers=1)
        self.enc_3x3 = encoder(in_channels, self.branch_channels[3], kernel_size=3, ks_res=3, num_layers=1)

        # CUDA Streams Initialization
        self.streams = [torch.cuda.Stream() for _ in range(4)]

    def forward_encoder(self, x):
        if not self.use_encoder:
            return x

        default_stream = torch.cuda.current_stream()
        results = [None] * 4
        branches = [self.enc_1x1, self.enc_1x3, self.enc_3x1, self.enc_3x3]

        for i, (stream, branch) in enumerate(zip(self.streams, branches)):
            with torch.cuda.stream(stream):
                stream.wait_stream(default_stream)
                results[i] = branch(x)

        for i, stream in enumerate(self.streams):
            results[i].record_stream(default_stream)
            default_stream.wait_stream(stream)
        
        x = torch.cat(results, dim=1)
        return x

    def forward(self, x):
        return self.forward_encoder(x)


class ImageEncoderDecoupled(nn.Module):
    """
    实现了全方位空间分解 (Spatial Decomposition) 的编码器。
    将 3x3 空间感知解耦为 4 个独立的分支：
    1. Center (1x1): 中心点信息
    2. Horizontal (1x3): 水平方向特征
    3. Vertical (3x1): 垂直方向特征
    4. Corners (2x2, dilation=2): 四角特征与高频纹理
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        heads_rope=1,
        use_encoder=True,
        rope_base=None,
        rope_rescale=None,
        img_layers=2,
    ):
        super().__init__()
        self.use_encoder = use_encoder
        self.out_channels = out_channels
        
        # 通道分配：class ImageEncoderDecoupled(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        heads_rope=1,
        use_encoder=True,
        rope_base=None,
        rope_rescale=None,
        img_layers=2,
    ):
        super().__init__()
        self.use_encoder = use_encoder
        self.out_channels = out_channels
        
        # 通道分配：4个分支均分
        weights = [4, 4, 4, 4] 
        total_weight = sum(weights)
        self.branch_channels = []
        accumulated = 0
        for i in range(3):
            c = (out_channels * weights[i]) // total_weight
            self.branch_channels.append(c)
            accumulated += c
        self.branch_channels.append(out_channels - accumulated)
            
        # --- 分支定义 ---
        
        # Branch 1: Center (中心点)
        self.enc_center = encoder(
            in_channels, self.branch_channels[0], 
            kernel_size=1, ks_res=1, dilation=1, num_layers=2
        )
        
        # Branch 2: Horizontal (水平腰带)
        self.enc_horiz = encoder(
            in_channels, self.branch_channels[1], 
            kernel_size=(1, 3), ks_res=(1, 3), dilation=1, num_layers=1
        )
        
        # Branch 3: Vertical (垂直脊柱)
        self.enc_vert = encoder(
            in_channels, self.branch_channels[2], 
            kernel_size=(3, 1), ks_res=(3, 1), dilation=1, num_layers=1
        )
        
        # Branch 4: Corners (四个角落)
        self.enc_corners = encoder(
            in_channels, self.branch_channels[3], 
            kernel_size=2, ks_res=2, dilation=2, num_layers=1
        )

        # [新增] CUDA Streams 初始化：为4个分支各准备一个流
        self.streams = [torch.cuda.Stream() for _ in range(4)]

    def forward_encoder(self, x):
        if not self.use_encoder:
            return x

        # [新增] 并行处理每个空间分量
        default_stream = torch.cuda.current_stream()
        results = [None] * 4
        branches = [self.enc_center, self.enc_horiz, self.enc_vert, self.enc_corners]

        # 并发启动 4 个分支的计算
        for i, (stream, branch) in enumerate(zip(self.streams, branches)):
            with torch.cuda.stream(stream):
                # 确保当前流的操作在默认流处理完输入数据之后才开始
                stream.wait_stream(default_stream)
                results[i] = branch(x)

        # 同步流并拼接结果
        for i, stream in enumerate(self.streams):
            # 将该 tensor 的使用记录在默认流上，防止被过早释放
            results[i].record_stream(default_stream)
            # 默认流需要等待所有的分支流计算完毕
            default_stream.wait_stream(stream)
            
        # 融合特征
        x = torch.cat(results, dim=1)
        return x

    def forward(self, x):
        x = self.forward_encoder(x)
        return x

