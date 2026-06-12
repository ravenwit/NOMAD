#!/usr/bin/env python3
"""
ONNX Export Script for NOMAD Toroidal Operator Net

Inspects the trained PeriodicUNet checkpoint (toroidal_operator_net.pt),
loads the model weights, and exports to ONNX format for browser inference
via ONNX Runtime Web.

Usage:
    python scripts/export_onnx.py

Output:
    - web/public/toroidal_operator_net.onnx  (ONNX model)
    - web/public/norm_stats.json             (normalization statistics)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.periodic_unet import PeriodicUNet


def inspect_checkpoint(checkpoint_path: str) -> dict:
    """
    Inspect the checkpoint file to understand its structure.
    """
    print(f"\n{'='*60}")
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print(f"File size: {os.path.getsize(checkpoint_path) / 1e6:.1f} MB")
    print(f"{'='*60}\n")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        print("Checkpoint is a dictionary with keys:")
        for key in checkpoint.keys():
            val = checkpoint[key]
            if isinstance(val, dict):
                print(f"  '{key}': dict with {len(val)} entries")
                # Show first few keys of nested dicts
                for i, (k, v) in enumerate(val.items()):
                    if i < 5:
                        if isinstance(v, torch.Tensor):
                            print(f"    '{k}': Tensor {v.shape} ({v.dtype})")
                        else:
                            print(f"    '{k}': {type(v).__name__}")
                    elif i == 5:
                        print(f"    ... and {len(val) - 5} more entries")
                        break
            elif isinstance(val, torch.Tensor):
                print(f"  '{key}': Tensor {val.shape} ({val.dtype})")
            elif isinstance(val, (int, float)):
                print(f"  '{key}': {val}")
            elif isinstance(val, list):
                print(f"  '{key}': list of length {len(val)}")
            else:
                print(f"  '{key}': {type(val).__name__}")
    else:
        print(f"Checkpoint is a {type(checkpoint).__name__}")
    
    return checkpoint


def extract_norm_stats(checkpoint: dict) -> dict:
    """
    Extract normalization statistics from the checkpoint.
    """
    stats = {}
    for key in ['p_mean', 'p_std', 's_mean', 's_std']:
        if key in checkpoint:
            val = checkpoint[key]
            if isinstance(val, torch.Tensor):
                stats[key] = float(val.item())
            else:
                stats[key] = float(val)
            print(f"  {key}: {stats[key]:.8f}")
        else:
            print(f"  WARNING: '{key}' not found in checkpoint!")
            # Use safe defaults
            if 'mean' in key:
                stats[key] = 0.0
            else:
                stats[key] = 1.0
    
    return stats


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def validate_model(model: nn.Module, device: str = 'cpu'):
    """
    Run a quick forward pass to validate the model works.
    """
    model.eval()
    with torch.no_grad():
        # Input: [P_curr, P_prev, S_curr, M_static] = 4 channels, 64x64
        dummy_input = torch.randn(1, 4, 64, 64, device=device)
        output = model(dummy_input)
        print(f"\n  Validation forward pass:")
        print(f"    Input shape:  {dummy_input.shape}")
        print(f"    Output shape: {output.shape}")
        print(f"    Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
        
        # Sanity check: output should be finite
        assert torch.isfinite(output).all(), "ERROR: Model output contains NaN/Inf!"
        print(f"    ✓ Output is finite")
        
        return dummy_input


def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, output_path: str):
    """
    Export the PeriodicUNet to ONNX format.
    
    Note on circular padding: PyTorch's ONNX exporter handles Conv2d with 
    padding_mode='circular' by decomposing it into explicit Pad(mode='wrap') + Conv2d.
    This requires ONNX opset >= 16 which supports the 'wrap' pad mode.
    If that fails, we fall back to opset 11 with 'reflect' mode approximation.
    """
    model.eval()
    
    # Try opset 17 first (full FFT + wrap pad support)
    opset_versions = [17, 16, 14, 11]
    
    for opset in opset_versions:
        try:
            print(f"\n  Attempting ONNX export with opset {opset}...")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=opset,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=None,  # Fixed batch size = 1 for browser
            )
            
            # Verify the export
            file_size = os.path.getsize(output_path) / 1e6
            print(f"  ✓ ONNX export successful!")
            print(f"    Output: {output_path}")
            print(f"    Size: {file_size:.1f} MB")
            print(f"    Opset: {opset}")
            
            # Validate with onnxruntime if available
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(output_path)
                input_name = sess.get_inputs()[0].name
                result = sess.run(None, {input_name: dummy_input.numpy()})
                
                # Compare with PyTorch output
                with torch.no_grad():
                    pt_output = model(dummy_input).numpy()
                
                max_diff = np.max(np.abs(result[0] - pt_output))
                print(f"\n  ONNX vs PyTorch validation:")
                print(f"    Max absolute difference: {max_diff:.8e}")
                
                # Mathematical sanity check: difference should be negligible
                if max_diff < 1e-4:
                    print(f"    ✓ Numerical match (diff < 1e-4)")
                elif max_diff < 1e-2:
                    print(f"    ⚠ Minor numerical differences (diff < 1e-2)")
                else:
                    print(f"    ✗ WARNING: Significant numerical differences!")
                    
            except ImportError:
                print("  (onnxruntime not installed, skipping validation)")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Opset {opset} failed: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            continue
    
    return False


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(project_root, "toroidal_operator_net.pt")
    onnx_output_dir = os.path.join(project_root, "web", "public")
    onnx_output_path = os.path.join(onnx_output_dir, "toroidal_operator_net.onnx")
    stats_output_path = os.path.join(onnx_output_dir, "norm_stats.json")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    os.makedirs(onnx_output_dir, exist_ok=True)
    
    # Step 1: Inspect checkpoint
    checkpoint = inspect_checkpoint(checkpoint_path)
    
    # Step 2: Extract normalization statistics
    print("\nExtracting normalization statistics...")
    norm_stats = extract_norm_stats(checkpoint)
    
    # Save norm stats as JSON for the browser
    with open(stats_output_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved to: {stats_output_path}")
    
    # Step 3: Load model
    print("\nLoading PeriodicUNet model...")
    model = PeriodicUNet(n_channels=4, n_classes=1, bilinear=True)
    
    # Load state dict from checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    total_params = count_parameters(model)
    param_size_mb = total_params * 4 / 1e6  # float32
    print(f"  Total parameters: {total_params:,}")
    print(f"  Parameter memory: {param_size_mb:.1f} MB")
    
    # Mathematical sanity check on parameter count:
    # PeriodicUNet(4, 1) with bilinear=False should have ~31M params
    # That's 4 encoder stages (64,128,256,512,1024) + 4 decoder stages + output
    expected_min = 10_000_000  # At least 10M for a UNet of this depth
    expected_max = 50_000_000  # At most 50M
    if expected_min <= total_params <= expected_max:
        print(f"  ✓ Parameter count is in expected range [{expected_min/1e6:.0f}M, {expected_max/1e6:.0f}M]")
    else:
        print(f"  ⚠ Parameter count {total_params/1e6:.1f}M is outside expected range!")
    
    # Step 4: Validate forward pass
    dummy_input = validate_model(model)
    
    # Step 5: Export to ONNX
    print("\nExporting to ONNX...")
    success = export_to_onnx(model, dummy_input, onnx_output_path)
    
    if success:
        print(f"\n{'='*60}")
        print("EXPORT COMPLETE")
        print(f"{'='*60}")
        print(f"  ONNX model: {onnx_output_path}")
        print(f"  Norm stats: {stats_output_path}")
        print(f"\nThe model is ready for browser inference via ONNX Runtime Web.")
        print(f"Input: (1, 4, 64, 64) — [P_curr, P_prev, S_curr, M_static]")
        print(f"Output: (1, 1, 64, 64) — P_next (normalized)")
    else:
        print("\n✗ ONNX export failed with all opset versions!")
        print("You may need to modify the model to remove unsupported operations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
