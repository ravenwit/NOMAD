import pytest
import torch
from src.models.geofno import GeoFNO

def test_geofno_shapes():
    batch = 2
    t_in = 3
    t_out = 5
    h, w = 32, 32
    
    model = GeoFNO(modes=8, width=16, t_in=t_in, t_out=t_out, geom_channels=3, n_theta=h, n_phi=w)
    
    p_in = torch.randn(batch, t_in, h, w)
    s_in = torch.randn(batch, t_in + t_out, h, w)
    geom = torch.randn(batch, 3, h, w)
    
    out = model(p_in, s_in, geom)
    
    assert out.shape == (batch, t_out, h, w)
    assert torch.isfinite(out).all()
