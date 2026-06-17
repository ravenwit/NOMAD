import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, 2, dtype=torch.float32)
        )

    def forward(self, x):
        orig_dtype = x.dtype
        with autocast(x.device.type, enabled=False):
            x = x.to(torch.float32)
            batchsize = x.shape[0]

            x_ft = torch.fft.rfft2(x)

            x_ft_1 = x_ft[:, :, :self.modes1, :self.modes2]
            x_ft_2 = x_ft[:, :, -self.modes1:, :self.modes2]

            xr1, xi1 = x_ft_1.real, x_ft_1.imag
            xr2, xi2 = x_ft_2.real, x_ft_2.imag

            wr1, wi1 = self.weights1[..., 0], self.weights1[..., 1]
            wr2, wi2 = self.weights2[..., 0], self.weights2[..., 1]

            out1_r = torch.einsum("bixy,ioxy->boxy", xr1, wr1) - torch.einsum("bixy,ioxy->boxy", xi1, wi1)
            out1_i = torch.einsum("bixy,ioxy->boxy", xi1, wr1) + torch.einsum("bixy,ioxy->boxy", xr1, wi1)

            out2_r = torch.einsum("bixy,ioxy->boxy", xr2, wr2) - torch.einsum("bixy,ioxy->boxy", xi2, wi2)
            out2_i = torch.einsum("bixy,ioxy->boxy", xi2, wr2) + torch.einsum("bixy,ioxy->boxy", xr2, wi2)

            out_ft_real = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.float32, device=x.device)
            out_ft_imag = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.float32, device=x.device)

            out_ft_real[:, :, :self.modes1, :self.modes2] = out1_r
            out_ft_real[:, :, -self.modes1:, :self.modes2] = out2_r

            out_ft_imag[:, :, :self.modes1, :self.modes2] = out1_i
            out_ft_imag[:, :, -self.modes1:, :self.modes2] = out2_i

            out_ft = torch.complex(out_ft_real, out_ft_imag)
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

        return x.to(orig_dtype)


class BaseFNO2d(nn.Module):
    def __init__(self, modes=16, width=64, in_channels=12, out_channels=10, n_layers=4):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_channels, width)
        self.convs = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)          # (B, H, W, C)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)          # (B, C, H, W)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)          # (B, out_channels, H, W)
        return x.float()

class FNO2d(nn.Module):
    def __init__(self, modes=12, width=32, in_channels=4, out_channels=1):
        super(FNO2d, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x
