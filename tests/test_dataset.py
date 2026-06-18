import pytest
import torch
import h5py
import os
from src.data.dataset import ChunkedTorusDataset

@pytest.fixture
def dummy_h5_file(tmp_path):
    file_path = tmp_path / "dummy_torus.h5"
    with h5py.File(file_path, 'w') as f:
        # P_save shape from solver is [Batch, Time, N_theta, N_phi, Channels]
        f.create_dataset('pressure', data=torch.rand(2, 50, 32, 32, 1).numpy())
        f.create_dataset('source', data=torch.rand(2, 50, 32, 32, 1).numpy())
        f.attrs['R'] = 3.0
        f.attrs['r'] = 1.0
        f.attrs['N_theta'] = 32
        f.attrs['N_phi'] = 32
        f.attrs['c'] = 343.0
        f.attrs['dt_macro'] = 0.05
    return str(file_path)

def test_chunked_dataset_shapes(dummy_h5_file):
    t_in = 3
    t_out = 5
    unroll = 2
    
    ds = ChunkedTorusDataset(dummy_h5_file, t_in=t_in, t_out=t_out, unroll_steps=unroll)
    
    assert ds.chunk_size == 3 + 2 * 5 # 13
    
    p_in, s_unrolled, geom, p_target = ds[0]
    
    assert p_in.shape == (t_in, 32, 32)
    assert s_unrolled.shape == (13, 32, 32)
    assert geom.shape == (3, 32, 32)
    assert p_target.shape == (10, 32, 32) # unroll * t_out

def test_chunked_dataset_lazy_loading(dummy_h5_file):
    ds = ChunkedTorusDataset(dummy_h5_file, t_in=2, t_out=3, unroll_steps=1)
    
    # ensure it hasn't opened the file handle persistently in memory during init
    assert ds._h5_file is None
    
    # grab an item, which initializes the handle
    _ = ds[0]
    assert ds._h5_file is not None
