import json
import numpy as np
import os

def convert(json_path, bin_path):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Converting to float32...")
    arr = np.array(data, dtype=np.float32)
    print(f"Shape: {arr.shape}, Min: {arr.min()}, Max: {arr.max()}")
    
    print(f"Saving to {bin_path}...")
    arr.tofile(bin_path)
    
    orig_size = os.path.getsize(json_path) / (1024*1024)
    new_size = os.path.getsize(bin_path) / (1024*1024)
    print(f"Done. Size reduced from {orig_size:.2f} MB to {new_size:.2f} MB")

if __name__ == '__main__':
    convert('complex_chaos_gt.json', 'web/public/data/complex_chaos_gt.bin')
    convert('complex_chaos_pred.json', 'web/public/data/complex_chaos_pred.bin')
