import os
import sys
import torch
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.hybrid_model import HybridWeaponDetector

def export_to_onnx():
    load_dotenv(PROJECT_ROOT / ".env")
    
    weights_path = PROJECT_ROOT / "models" / "weights" / "best.pt"
    onnx_path = PROJECT_ROOT / "models" / "weights" / "best.onnx"
    
    if not weights_path.exists():
        print(f"❌ Cannot find weights at {weights_path}")
        print("Please download best.pt from Google Colab and place it there first.")
        sys.exit(1)
        
    print(f"🔄 Loading PyTorch model from {weights_path}...")
    device = "cpu"  # Export on CPU to avoid GPU tensor tracing issues
    
    backbone_variant = os.getenv("BACKBONE_VARIANT", "yolo11s.pt")
    
    model = HybridWeaponDetector(
        backbone_variant=backbone_variant,
        pretrained=False,
        nc=3,
        device=device
    )
    
    # Load state dict
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        sys.exit(1)
        
    model.eval()
    
    # Create a dummy input (Batch Size = 1, Channels = 3, Height = 640, Width = 640)
    print("🔄 Tracing forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    
    # Export
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,          # Opset 14 supports most modern transformer ops
            do_constant_folding=True,
            input_names=['images'],
            output_names=['cls_logits', 'reg_offsets', 'objectness'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'cls_logits': {0: 'batch_size'},
                'reg_offsets': {0: 'batch_size'},
                'objectness': {0: 'batch_size'}
            }
        )
        print(f"✅ Successfully exported to {onnx_path}!")
        print("You can now run the backend; it will automatically detect and load the ONNX model.")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")

if __name__ == "__main__":
    export_to_onnx()
