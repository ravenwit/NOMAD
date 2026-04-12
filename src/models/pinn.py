import torch
import torch.nn as nn
import torch.nn.functional as F

class TorusConvBlock(nn.Module):
    """
    A convolution block that enforces Toroidal periodicity using circular padding.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        self.norm = nn.GroupNorm(min(out_channels, 4), out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        # Circular padding handles the periodic boundary conditions of the Torus
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='circular')
        return self.act(self.norm(self.conv(x_padded)))

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Acoustic Wave Prediction on a Torus.
    Architecture: ResNet-style Convolutional Encoder-Decoder.
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        
        # Initial projection (2 channels for u_t and u_t-1)
        self.inc = TorusConvBlock(2, hidden_dim)
        
        # Residual blocks maintaining spatial resolution
        self.res1 = TorusConvBlock(hidden_dim, hidden_dim)
        self.res2 = TorusConvBlock(hidden_dim, hidden_dim)
        self.res3 = TorusConvBlock(hidden_dim, hidden_dim)
        
        # Final output projection
        self.outc = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        
    def forward(self, x):
        h = self.inc(x)
        
        # Residuality
        h1 = self.res1(h)
        h2 = self.res2(h1 + h)
        h3 = self.res3(h2 + h1)
        
        out = self.outc(h3)
        
        # Skip connection to u_t (the first channel of the input)
        # Assuming input shape is (batch, 2, H, W)
        # We predict the delta relative to the CURRENT state.
        return out + x[:, 0:1, :, :]

if __name__ == "__main__":
    model = PINN()
    # Test with 2 channels: (u_t, u_t-1)
    test_input = torch.randn(2, 2, 64, 64)
    output = model(test_input)
    print(f"Model Test: {test_input.shape} -> {output.shape}")
    
    # Verify periodicity (shift input, should shift output)
    shifted_input = torch.roll(test_input, shifts=(5, 5), dims=(2, 3))
    shifted_output = model(shifted_input)
    expected_shift = torch.roll(output, shifts=(5, 5), dims=(2, 3))
    diff = (shifted_output - expected_shift).abs().max().item()
    print(f"Equivariance Test (Diff): {diff:.6e} (Should be near 0)")
