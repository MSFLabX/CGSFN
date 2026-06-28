import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class ffn(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn1(x)


class Cross_block(nn.Module):

    def __init__(self, in_c1, in_c2, mid_c, cro) -> None:
        super().__init__()
        self.cro = cro
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c1, out_channels=mid_c, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_c2, out_channels=mid_c, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1, stride=2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=mid_c, out_channels=mid_c, kernel_size=(3, 3), padding=1, stride=2),
        )
        self.act = SCAM(mid_c)

    def forward(self, Y, Z):
        Y1 = self.conv1(Y)
        Z1 = self.conv2(Z)
        if self.cro:
            c = self.act(Y1, Z1)
            Y1 = self.down1(Y1)
            Z1 = self.down1(Z1)
            return c, Y1, Z1
        Y1 = self.down1(Y1)
        Z1 = self.down2(Z1)
        return Y1, Z1


class UpDe(nn.Module):

    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), padding=1),
        )
        self.conv = nn.Conv2d(in_channels=out_c + out_c, out_channels=out_c, kernel_size=(3, 3), padding=1)

    def forward(self, infr, latt):
        x = self.up(latt)
        out = torch.cat((x, infr), dim=1)
        out = self.conv(out)
        return out


class SCAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = nn.LayerNorm(c)
        self.norm_r = nn.LayerNorm(c)
        self.norm = nn.LayerNorm(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(x_l).permute(0, 2, 3, 1)
        Q_r_T = self.r_proj1(x_r).permute(0, 2, 1, 3)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)

        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        out = F_r2l + F_l2r
        return x_l + x_r


class Block(nn.Module):
    def __init__(self, in_c1, in_c2, out_c):
        super().__init__()
        self.Embedding1 = nn.Sequential(
            nn.Linear(in_c1, out_c),
            nn.LayerNorm(out_c)
        )
        # self.patch = nn.Conv2d(in_channels=in_c1, out_channels=in_c1, kernel_size=8, stride=8)
        self.Embedding2 = nn.Sequential(
            nn.Linear(in_c2, out_c),
            nn.LayerNorm(out_c)
        )
        self.attn = DualT(dim=out_c, heads=4, dim_head=16)
        self.FFN = ffn(out_c, out_c)

    def forward(self, X, Y):
        H = X.size(2)
        E1 = rearrange(X, 'B c H W -> B (H W) c', H=H)
        E1 = self.Embedding1(E1)
        E2 = rearrange(Y, 'B c H W -> B (H W) c', H=H)
        E2 = self.Embedding2(E2)
        attn = self.attn(E1, E2)
        out = self.FFN(attn) + attn
        out = rearrange(out, 'B (H W) C -> B C H W', H=H)
        return out


class DualT(nn.Module):

    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv_1 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_2 = nn.Linear(dim, inner_dim * 3, bias=False)
        # self.to_qkv_3 = nn.Linear(dim * 2, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        # print(x.shape, y.shape)
        b, n, c, h = x.shape[0], x.shape[1], x.shape[2], self.heads
        qkv1 = self.to_qkv_1(x).chunk(3, dim=-1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv1)
        qkv2 = self.to_qkv_2(y).chunk(3, dim=-1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h=h), qkv2)

        dots1 = torch.einsum('b h i d, b h j d -> b h i j', q1, k1) * self.scale
        attn1 = dots1.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        dots2 = torch.einsum('b h i d, b h j d -> b h i j', q2, k2) * self.scale
        attn2 = dots2.softmax(dim=-1)
        out2 = torch.einsum('b h i j, b h j d -> b h i d', attn2, v2)
        out2 = rearrange(out2, 'b h d n -> b n (h d)')

        input = torch.add(out1, out2)
        output = self.norm(self.to_out(input))
        # output = output + x + y
        return output

class DepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MSAA(nn.Module):
    def __init__(self, in_channels, alpter=8):
        super(MSAA, self).__init__()

        self.spatial_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.multi_scale_convs = nn.ModuleList([
            DepthConv(in_channels, in_channels // alpter, kernel_size=3, padding=1),
            DepthConv(in_channels, in_channels // alpter, kernel_size=5, padding=2),
            DepthConv(in_channels, in_channels // alpter, kernel_size=7, padding=3)
        ])

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels // alpter, in_channels // alpter, 1),
            nn.ReLU(),
            SFA_Improved1(in_channels=in_channels // alpter),
            nn.Conv2d(in_channels // alpter, 1, 1),
            nn.Sigmoid()
        )

        self.channel_agg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

        self.spa_cov = nn.Conv2d(in_channels // alpter, in_channels, kernel_size=1)

        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        spatial_x = self.spatial_proj(x)
        multi_scale = sum([conv(spatial_x) for conv in self.multi_scale_convs])
        spatial_attn = self.spatial_conv(multi_scale)
        spatial_refined = self.spa_cov(multi_scale * spatial_attn)
        temp = spatial_refined

        output = self.fusion(temp) + x

        return output


class NWI(nn.Module):
    def __init__(self, in_channels):
        super(NWI, self).__init__()

    def forward(self, x):
        w = x.shape[-1]
        gap_w = F.adaptive_avg_pool2d(x, (1, w))
        gap_h = F.adaptive_avg_pool2d(x, (w, 1))
        return gap_w + gap_h


class SFA_Improved1(nn.Module):
    def __init__(self, in_channels, num_branches=3, window_size=8):
        super(SFA_Improved1, self).__init__()
        self.window_size = window_size
        alpter = window_size

        self.conv1x1_list = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // alpter, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels // alpter, kernel_size=5, padding=2),
            nn.Conv2d(in_channels, in_channels // alpter, kernel_size=7, padding=3)
        ])

        self.nwi = NWI(in_channels // alpter)

        self.depthwise_conv_focus = nn.Conv2d(
            in_channels // alpter, in_channels // alpter, kernel_size=3, stride=1, padding=1,
            groups=in_channels // alpter, bias=False)

        self.cov_x = nn.Conv2d(
            in_channels, in_channels // alpter, kernel_size=3, stride=1, padding=1,
            groups=in_channels // alpter, bias=False)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels // alpter, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        B, _, H, W = x.shape

        branch1, branch2, branch3 = [conv(x) for conv in self.conv1x1_list]

        def window_partition(tensor, win_size):
            B, C, H, W = tensor.shape
            tensor = tensor.view(B, C, H // win_size, win_size, W // win_size, win_size)
            tensor = tensor.permute(0, 2, 4, 1, 3, 5).contiguous()
            return tensor.view(-1, C, win_size, win_size)  # [B*n_win, C, win, win]

        def window_reverse(windows, win_size, H, W):
            B = int(windows.shape[0] / (H * W / win_size / win_size))
            C = windows.shape[1]
            x = windows.view(B, H // win_size, W // win_size, C, win_size, win_size)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            return x.view(B, C, H, W)

        win1 = window_partition(branch1, self.window_size)
        win2 = window_partition(branch2, self.window_size)

        Bw, Cw, Hw, Ww = win1.shape
        win1_flat = win1.view(Bw, Cw, -1).transpose(1, 2)  # [Bw, win*win, C]
        win2_flat = win2.view(Bw, Cw, -1)  # [Bw, C, win*win]

        att_map = torch.bmm(win1_flat, win2_flat)  # [Bw, win*win, win*win]
        att_map = F.softmax(att_map, dim=-1)  # softmax归一化

        branch3_win = window_partition(branch3, self.window_size)
        branch3_flat = branch3_win.view(Bw, Cw, -1)
        attended_flat = torch.bmm(branch3_flat, att_map)
        attended_win = attended_flat.view(Bw, Cw, Hw, Ww)
        branch3_att1 = window_reverse(attended_win, self.window_size, H, W)

        branch3_att = self.nwi(branch3_att1)

        focus_feat = self.depthwise_conv_focus(branch3_att1)
        fused = branch3_att + focus_feat

        out = self.final_conv(fused)
        return out

class FeedForward_1(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ffn(x)


class Cross_Self_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.5):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv_1 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_2 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y):
        b, N, _ = x.shape
        h = self.heads
        qkv1 = self.to_qkv_1(x).chunk(3, dim=-1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b N (h d) -> b h N d', h=h), qkv1)

        qkv2 = self.to_qkv_2(y).chunk(3, dim=-1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b N (h d) -> b h N d', h=h), qkv2)

        dots1 = torch.einsum('b h i d, b h j d -> b h i j', q2, k1) * self.scale
        attn1 = dots1.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', attn1, v1)
        out1 = rearrange(out1, 'b h N d -> b N (h d)')

        dots2 = torch.einsum('b h i d, b h j d -> b h i j', q1, k2) * self.scale
        attn2 = dots2.softmax(dim=-1)
        out2 = torch.einsum('b h N N, b h N d -> b h N d', attn2, v2)
        out2 = rearrange(out2, 'b h N d -> b N (h d)')

        input1 = torch.add(out1, out2)
        input2 = self.to_out(input1)

        output = input2 + x + y

        return output


class CSABlock(nn.Module):
    def __init__(self, in_c1, in_c2):
        super().__init__()
        self.Embedding1 = nn.Conv2d(in_c1, in_c1, (1, 1))
        self.Embedding2 = nn.Conv2d(in_c2, in_c1, (1, 1))

        self.atten = Cross_Self_Attention(dim=in_c1, heads=4, dim_head=18)
        self.FFN = FeedForward_1(in_c1, in_c1 // 2, 0.0)

    def forward(self, X, Y):
        H = X.size(2)
        E1 = self.Embedding1(X)
        E11 = rearrange(E1, 'B c H W -> B (H W) c', H=H)

        E2 = self.Embedding2(Y)

        E22 = rearrange(E2, 'B c H W -> B (H W) c', H=H)

        attn = self.atten(E11, E22)

        out = self.FFN(attn)
        out = attn + out

        output = rearrange(out, 'B (H W) C -> B C H W', H=H)

        return output

class CSANet(nn.Module):
    def __init__(self, in_c1, in_c2, in_c3, out_c):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c1, out_channels=in_c3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_c2, out_channels=in_c3, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
        )

        self.CSABlock1 = CSABlock(in_c3, in_c3)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_c3, out_channels=out_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
        )

    def forward(self, X, Y):
        X = self.conv1(X)
        Y = self.conv2(Y)
        out = self.CSABlock1(X, Y)
        out = self.conv3(out)
        return out

class CACW(nn.Module):

    def __init__(self, num_features: int, hidden_dim: int = 64):
        super(CACW, self).__init__()
        self.num_features = num_features
        self.mlp = nn.Sequential(
            nn.Linear(num_features * num_features, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, num_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_samples, num_features = x.size()
        assert num_features == self.num_features, "输入特征数与模块初始化特征数不匹配"

        mean_x = x.mean(dim=1, keepdim=True)
        centered_x = x - mean_x
        cov_matrix = torch.bmm(centered_x.transpose(1, 2), centered_x) / (num_samples - 1)

        norms = torch.norm(centered_x, dim=1, keepdim=True)
        denominator = norms * norms.transpose(1, 2)
        normalized_cov = cov_matrix / denominator.clamp(min=1e-8)

        flattened_cov = normalized_cov.view(batch_size, -1)
        weights = self.mlp(flattened_cov)
        return weights.unsqueeze(-1)


class IFW(nn.Module):

    def __init__(self, channels: int, hidden_dim: int = 64):
        super(IFW, self).__init__()
        self.cacw = CACW(num_features=channels, hidden_dim=hidden_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature.size()

        feature_reshaped = feature.view(batch_size, channels, -1).transpose(1, 2)

        channel_weights = self.cacw(feature_reshaped)

        weights_reshaped = channel_weights.unsqueeze(-1)

        adjusted_feature = feature * weights_reshaped
        return adjusted_feature


class Net(nn.Module):
    def __init__(self, hs_band, ms_band, mid=64):
        super(Net, self).__init__()
        self.intro1 = nn.Sequential(
            nn.Conv2d(in_channels=hs_band, out_channels=mid, kernel_size=(1, 1), padding=0),
            nn.Conv2d(in_channels=mid, out_channels=mid, kernel_size=(3, 3), padding=1),
        )
        self.intro2 = nn.Sequential(
            nn.Conv2d(in_channels=ms_band, out_channels=mid, kernel_size=(1, 1), padding=0),
            nn.Conv2d(in_channels=mid, out_channels=mid, kernel_size=(3, 3), padding=1),
        )

        self.En1 = Cross_block(in_c1=mid, in_c2=mid, mid_c=64, cro=False)
        self.En2 = Cross_block(in_c1=64, in_c2=64, mid_c=96, cro=False)
        self.En3 = Cross_block(in_c1=96, in_c2=96, mid_c=128, cro=False)

        self.ifw1 = IFW(channels=64)
        self.ifw2 = IFW(channels=96)
        self.ifw3 = IFW(channels=128)

        self.csa1 = CSANet(64, 64, 96, 96)
        self.csa2 = CSANet(96, 96, 128, 128)
        self.csa3 = CSANet(128, 128, 256, 256)

        self.covup1 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(1, 1), padding=0)
        self.covup2 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(1, 1), padding=0)
        self.covup3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), padding=0)

        self.msaa1 = MSAA(96)
        self.msaa2 = MSAA(128)
        self.msaa3 = MSAA(256)

        self.cDe1 = UpDe(in_c=256, out_c=128)
        self.cDe2 = UpDe(in_c=128, out_c=96)
        self.cDe3 = UpDe(in_c=96, out_c=64)

        self.yz_mid = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(1, 1))
        )

        self.tail = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.Conv2d(64, hs_band, kernel_size=(1, 1))
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(3, 3), padding=1),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(3, 3), padding=1),
        )

    def forward(self, y, z):
        Y0 = F.interpolate(y, scale_factor=8, mode='bicubic', align_corners=False)
        Y = self.intro1(Y0)
        Z = self.intro2(z)
        y1, z1 = self.En1(Y, Z)
        y2, z2 = self.En2(y1, z1)
        y3, z3 = self.En3(y2, z2)
        y1_ifw = self.ifw1(y1)
        z1_ifw = self.ifw1(z1)
        y2_ifw = self.ifw2(y2)
        z2_ifw = self.ifw2(z2)
        y3_ifw = self.ifw3(y3)
        z3_ifw = self.ifw3(z3)
        yz_csa1 = self.csa1(y1_ifw, z1_ifw) + self.covup1(y1) + self.covup1(z1)
        yz_csa2 = self.csa2(y2_ifw, z2_ifw) + self.covup2(y2) + self.covup2(z2)
        yz_csa3 = self.csa3(y3_ifw, z3_ifw) + self.covup3(y3) + self.covup3(z3)
        yz_msaa1 = self.msaa1(yz_csa1)
        yz_msaa2 = self.msaa2(yz_csa2)
        yz_msaa3 = self.msaa3(yz_csa3)

        yz_mid = self.yz_mid(torch.cat((Y, Z), dim=1))

        yz_cde1 = self.cDe1(yz_msaa2, yz_msaa3)
        yz_cde2 = self.cDe2(yz_msaa1, yz_cde1)
        yz_cde3 = self.cDe3(yz_mid, yz_cde2)

        out = self.conv(torch.cat((self.up3(yz_cde1), yz_cde2), dim=1))
        out = self.tail(torch.cat((self.up4(out), yz_cde3), dim=1)) + Y0

        return out


if __name__ == '__main__':
    y = torch.randn(1, 31, 8, 8)
    z = torch.randn(1, 3, 64, 64)
    net = Net(31, 3)
    out = net(y, z)
    print(out.shape)
