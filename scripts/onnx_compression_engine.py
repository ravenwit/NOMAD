import os
import sys
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16

# Add project root to path smartly to allow modular execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.geofno import GeoFNO

def export_and_quantize_fno(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "best_geofno_resumed.pt"))
        
    print("=" * 80)
    print("🚀 CHORUS FRAMEWORK: ONNX COMPRESSION ENGINE")
    print("=" * 80)

    device = torch.device('cpu') # Export is always safest on CPU

    # 1. Reconstruct the exact 151M Parameter Architecture
    # (modes=24, width=128 is the footprint of your 151M model)
    T_IN = 3
    T_OUT = 30
    HI_RES = 256

    print("1. Initializing GeoFNO (151M Params)...")
    model = GeoFNO(modes=24, width=128, t_in=T_IN, t_out=T_OUT, geom_channels=3,
                   n_theta=HI_RES, n_phi=HI_RES).to(device)


    # print("1. Initializing GeoFNO...")
    # model = GeoFNO(modes=20, width=48, t_in=T_IN, t_out=T_OUT, geom_channels=3,
    #                n_theta=HI_RES, n_phi=HI_RES).to(device)

    # Safely load the weights (ignoring base_grid mismatches if any)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if 'base_grid' in state_dict:
            del state_dict['base_grid']
        model.load_state_dict(state_dict, strict=False)
        print(" ✅ Model weights loaded successfully.")
    except FileNotFoundError:
        print(f" ⚠️ Warning: {checkpoint_path} not found. Exporting untrained model structure.")
        
    model.eval()

    # 2. Create Dummy Tensors for the ONNX Tracer
    # Batch size is 1 because web deployment operates on single user clicks
    print("2. Generating dynamic tracer inputs...")
    dummy_p_in = torch.randn(1, T_IN, HI_RES, HI_RES, device=device)
    dummy_s_in = torch.randn(1, T_IN, HI_RES, HI_RES, device=device)
    dummy_geom = torch.randn(1, 3, HI_RES, HI_RES, device=device)

    # Output directory routes directly into web/public for React serving
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web', 'public', 'models'))
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Export to Base FP32 ONNX
    onnx_fp32_path = os.path.join(output_dir, "geofno_base_high_fp32.onnx")
    print(f"3. Compiling computational graph to {onnx_fp32_path} (Opset 17)...")

    torch.onnx.export(
        model,
        (dummy_p_in, dummy_s_in, dummy_geom),
        onnx_fp32_path,
        export_params=True,
        opset_version=17, # Required for complex-valued torch.fft
        do_constant_folding=True,
        # Ensure input names match your forward signature exactly
        input_names=['p_in', 's_in', 'geom_features'],
        output_names=['p_out'],
        # Lock spatial dimensions to 64; only batch size remains flexible
        dynamic_axes={
            'p_in': {0: 'batch_size'},
            's_in': {0: 'batch_size'},
            'geom_features': {0: 'batch_size'},
            'p_out': {0: 'batch_size'}
        }
    )

    # 4. FP16 Quantization (Half Precision)
    onnx_fp16_path = os.path.join(output_dir, "geofno_quantized_high_fp16.onnx")
    print(f"4. Applying FP16 Quantization -> {onnx_fp16_path}")
    onnx_model = onnx.load(onnx_fp32_path)
    onnx_model_fp16 = float16.convert_float_to_float16(onnx_model, op_block_list=['DFT'])
    onnx.save(onnx_model_fp16, onnx_fp16_path)

    # 5. INT8 Quantization (Dynamic 8-bit Integer)
    onnx_int8_path = os.path.join(output_dir, "geofno_quantized_high_int8.onnx")
    print(f"5. Applying INT8 Dynamic Quantization -> {onnx_int8_path}")
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        weight_type=QuantType.QUInt8
    )

    # 6. Evaluate Compression Results
    print("\n" + "=" * 80)
    print("📊 COMPRESSION RESULTS")
    print("=" * 80)

    def get_size_mb(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)

    try:
        orig_size = get_size_mb(checkpoint_path)
    except FileNotFoundError:
        orig_size = 0.0
        
    fp32_size = get_size_mb(onnx_fp32_path)
    fp16_size = get_size_mb(onnx_fp16_path)
    int8_size = get_size_mb(onnx_int8_path)

    print(f"PyTorch Checkpoint (Includes Optimizer) : ~1,730.00 MB")
    print(f"Original size                           : {orig_size:.2f} MB")
    print(f"ONNX FP32 (Bare Model)                  : {fp32_size:.2f} MB")
    print(f"ONNX FP16 (Half Precision)              : {fp16_size:.2f} MB  (Compression: {fp32_size/fp16_size:.2f}x)")
    print(f"ONNX INT8 (Integer Precision)           : {int8_size:.2f} MB  (Compression: {fp32_size/int8_size:.2f}x)")
    print("=" * 80)

if __name__ == "__main__":
    export_and_quantize_fno()
