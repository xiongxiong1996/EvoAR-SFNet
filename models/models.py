import torch
import torch.nn as nn
import torch.nn.functional as F
from .ARConv import ARConv
from .FDConv import FDConv

class ComplexGaborLayer(nn.Module):
    '''
        Complex Gabor nonlinearity 

        Inputs:
            input: Input features
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''

    def __init__(self, omega0=30.0, sigma0=10.0, trainable=True):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

    def forward(self, input):
        input = input.permute(0, -2, -1, 1)

        omega = self.omega_0 * input
        scale = self.scale_0 * input
        # return torch.exp(1j * omega - scale.abs().square())
        return torch.exp(1j * omega - scale.abs().square()).permute(0, -1, 1, 2)

class ConvDown(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
 
        if dsconv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1)
            )
 
    def forward(self, x):
        return self.conv(x)
 
 
class ConvUp(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
 
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2, 0)
        if dsconv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2, bias=False),
                nn.Conv2d(in_channels // 2, in_channels // 2, 1, 1, 0)
            )
        else:
            self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1)
 
    def forward(self, x, y):
        x = F.leaky_relu(self.conv1(x))
        x = x + y
        x = F.leaky_relu(self.conv2(x))
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pan_feat, ms_feat):
        b, c, h, w = pan_feat.size()
        proj_query = self.query_conv(pan_feat).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(ms_feat).view(b, -1, h * w)
        attention = self.softmax(torch.bmm(proj_query, proj_key))  # [B, HW, HW]
        proj_value = self.value_conv(ms_feat).view(b, c, -1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(b, c, h, w)
        return out + pan_feat  # 残差融合

class ARConv_Block(nn.Module):
    def __init__(self, in_planes, flag=False):
        super(ARConv_Block, self).__init__()
        self.flag = flag
        self.conv1 = ARConv(in_planes, in_planes, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = ARConv(in_planes, in_planes, 3, 1, 1)
 
    def forward(self, x, epoch, hw_range):
        res = self.conv1(x, epoch, hw_range)
        res = self.relu(res)
        res = self.conv2(res, epoch, hw_range)
        x = x + res
        return x
    
class FDConv_Block(nn.Module):
    def __init__(self, in_planes, flag=False):
        super(FDConv_Block, self).__init__()
        self.flag = flag
        self.conv1 = FDConv(in_planes, in_planes, kernel_size=3, padding=1, kernel_num=in_planes//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = FDConv(in_planes, in_planes, kernel_size=3, padding=1, kernel_num=in_planes//2)
 
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        x = x + res
        return x


class ImplicitDecoder(nn.Module):
    def __init__(self, in_channels, freq_dim=31, hidden_dims=[128,128,128], omega=30, scale=10.0):
        super().__init__()

        last_dim_K = in_channels 
        last_dim_Q = freq_dim

        self.K = nn.ModuleList()
        self.Q = nn.ModuleList()
        
        for hidden_dim in hidden_dims:
            self.K.append(nn.Sequential(nn.Conv2d(last_dim_K, hidden_dim, 1),
                                        nn.ReLU()))
            self.Q.append(nn.Sequential(nn.Conv2d(last_dim_Q, hidden_dim, 1),
                                        ComplexGaborLayer(omega0=omega,
                                                        sigma0=scale,
                                                        trainable=True)))
            last_dim_K = hidden_dim + in_channels
            last_dim_Q = hidden_dim

        self.last_layer = nn.Conv2d(hidden_dims[-1], in_channels, 1)

    def step(self, x, y):
        k = self.K[0](x).real
        q = k * self.Q[0](y)
        q = q.real
        for i in range(1, len(self.K)):
            k = self.K[i](torch.cat([q, x], dim=1)).real
            q = k * self.Q[i](q)
            q = q.real
        q = self.last_layer(q)
        return q

    def forward(self, INR_feat, freq_feat):
        output = self.step(INR_feat, freq_feat)
        return output


class EvoARFSNet(nn.Module):
    def __init__(self, pan_channels=1, lms_channels=8, fusion_type="implicit"):
        super(EvoARFSNet, self).__init__()
        self.fusion_type = fusion_type
        # head conv
        self.head_conv = nn.Conv2d(pan_channels + lms_channels, 32, 3, 1, 1)
        # space branch
        self.rb1 = ARConv_Block(32, 32)
        self.down1 = ConvDown(32)
        self.rb2 = ARConv_Block(64, 64)
        self.down2 = ConvDown(64)
        self.rb3 = ARConv_Block(128, 128)
        self.up1 = ConvUp(128)
        self.rb4 = ARConv_Block(64, 64)
        self.up2 = ConvUp(64)
        self.rb5 = ARConv_Block(32, 32)
        self.cross_attn = CrossAttention(channels=128)
        self.feedback_proj = nn.Conv2d(lms_channels, 128, kernel_size=1)
        self.tail_conv_s = nn.Conv2d(32, lms_channels, 3, 1, 1)
        # frequency branch
        self.fdrb1 = FDConv_Block(32, 32)
        self.fddown1 = ConvDown(32)
        self.fdrb2 = FDConv_Block(64, 64)
        self.fddown2 = ConvDown(64)
        self.fdrb3 = FDConv_Block(128, 128)
        self.fdup1 = ConvUp(128)
        self.fdrb4 = FDConv_Block(64, 64)
        self.fdup2 = ConvUp(64)
        self.fdrb5 = FDConv_Block(32, 32)
        self.cross_attn_fd = CrossAttention(channels=128)
        self.feedback_proj_fd = nn.Conv2d(lms_channels, 128, kernel_size=1)
        self.tail_conv_f = nn.Conv2d(32, lms_channels, 3, 1, 1)
        # fusion
        self.implicit_decoder = ImplicitDecoder(
            in_channels=32,
            freq_dim=32,
            hidden_dims=[64, 64, 64],
            omega=30,
            scale=10.0
        )
        self.explicit_gate = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.concat_fuse = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.tail_conv = nn.Conv2d(32, lms_channels, 3, 1, 1)

    def forward_ar_branch(self, x, epoch, hw_range):
        x1 = self.rb1(x, epoch, hw_range)
        x2 = self.down1(x1)
        x2 = self.rb2(x2, epoch, hw_range)
        x3 = self.down2(x2)
        x3 = self.rb3(x3, epoch, hw_range)
        x4 = self.up1(x3, x2)
        x4 = self.rb4(x4, epoch, hw_range)
        x5 = self.up2(x4, x1)
        x5 = self.rb5(x5, epoch, hw_range)
        pred1 = self.tail_conv_s(x5)

        pred1_down = F.interpolate(pred1, scale_factor=0.25, mode='bilinear', align_corners=False)
        pred1_proj = self.feedback_proj(pred1_down)
        fused1 = self.cross_attn(x3, pred1_proj)

        x4_fb1 = self.up1(fused1, x2)
        x4_fb1 = self.rb4(x4_fb1, epoch, hw_range)
        x5_fb1 = self.up2(x4_fb1, x1)
        x5_fb1 = self.rb5(x5_fb1, epoch, hw_range)
        pred2 = self.tail_conv_s(x5_fb1)
        
        pred2_down = F.interpolate(pred2, scale_factor=0.25, mode='bilinear', align_corners=False)
        pred2_proj = self.feedback_proj(pred2_down)
        fused2 = self.cross_attn(x3, pred2_proj)

        x4_fb2 = self.up1(fused2, x4_fb1)
        x4_fb2 = self.rb4(x4_fb2, epoch, hw_range)
        x5_fb2 = self.up2(x4_fb2, x5_fb1)
        x5_fb2 = self.rb5(x5_fb2, epoch, hw_range)
        pred3 = self.tail_conv_s(x5_fb2)
        return pred1, pred2, pred3,x5_fb2

    def forward_fd_branch(self, x, epoch, hw_range):
        x1 = self.fdrb1(x)
        x2 = self.fddown1(x1)
        x2 = self.fdrb2(x2)
        x3 = self.fddown2(x2)
        x3 = self.fdrb3(x3)
        x4 = self.fdup1(x3, x2)
        x4 = self.fdrb4(x4)
        x5 = self.fdup2(x4, x1)
        x5 = self.fdrb5(x5)
        pred1 = self.tail_conv_f(x5)
        pred1_down = F.interpolate(pred1, scale_factor=0.25, mode='bilinear', align_corners=False)
        pred1_proj = self.feedback_proj_fd(pred1_down)
        
        fused1 = self.cross_attn_fd(x3, pred1_proj)
        x4_fb1 = self.fdup1(fused1, x2)
        # 注意这里 FDConv_Block 的 forward 不接受 epoch 和 hw_range，所以不要传
        x4_fb1 = self.fdrb4(x4_fb1)
        x5_fb1 = self.fdup2(x4_fb1, x1)
        x5_fb1 = self.fdrb5(x5_fb1)
        pred2 = self.tail_conv_f(x5_fb1)
        pred2_down = F.interpolate(pred2, scale_factor=0.25, mode='bilinear', align_corners=False)
        pred2_proj = self.feedback_proj_fd(pred2_down)
        
        fused2 = self.cross_attn_fd(x3, pred2_proj)
        x4_fb2 = self.fdup1(fused2, x4_fb1)
        # 这里同上，不传额外参数
        x4_fb2 = self.fdrb4(x4_fb2)
        x5_fb2 = self.fdup2(x4_fb2, x5_fb1)
        x5_fb2 = self.fdrb5(x5_fb2)
        pred3 = self.tail_conv_f(x5_fb2)

        return pred1, pred2, pred3,x5_fb2

    def forward(self, pan, lms, epoch, hw_range):
        x = torch.cat([pan, lms], dim=1)
        x = self.head_conv(x)

        pred1, pred2, pred3, x5_ar = self.forward_ar_branch(x, epoch, hw_range)
        fd_pred1, fd_pred2, fd_pred3, x5_fd = self.forward_fd_branch(x, epoch, hw_range)

        if self.fusion_type == "implicit":
            x_refined = self.implicit_decoder(x5_ar, x5_fd)
        elif self.fusion_type == "explicit":
            gate = self.explicit_gate(torch.cat([x5_ar, x5_fd], dim=1))
            x_refined = gate * x5_ar + (1 - gate) * x5_fd
        elif self.fusion_type == "add":
            x_refined = x5_ar + x5_fd
        elif self.fusion_type == "concat":
            x_refined = self.concat_fuse(torch.cat([x5_ar, x5_fd], dim=1))
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        output = self.tail_conv(x_refined)

        return (
            lms + pred1,
            lms + pred2,
            lms + pred3,
            lms + fd_pred1,
            lms + fd_pred2,
            lms + fd_pred3,
            lms + output,
        )
