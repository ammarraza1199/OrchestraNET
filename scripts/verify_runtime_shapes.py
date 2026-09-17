import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from orchestranet.backbone import MobileNetV4Backbone, LightweightFPN
from orchestranet.models import M1PrimaryDetector

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Runtime Verification on device: {device}")

    # 1. Backbone
    backbone = MobileNetV4Backbone(pretrained=False).to(device)
    channels = backbone.get_out_channels()
    print(f"Backbone Channels: {channels}")

    x = torch.randn(2, 3, 640, 640, device=device)
    bb_feats = backbone(x)
    print("Backbone Output Shapes:")
    for i, f in enumerate(bb_feats):
        print(f"  P{i+3}: {list(f.shape)} (stride {640 // f.shape[2]})")

    # 2. FPN
    fpn = LightweightFPN(in_channels=channels, out_channels=128).to(device)
    fpn_feats = fpn(bb_feats)
    print("FPN Output Shapes:")
    for i, f in enumerate(fpn_feats):
        print(f"  FP{i+3}: {list(f.shape)}")

    # 3. M1
    m1 = M1PrimaryDetector(in_channels=128).to(device)
    out = m1(fpn_feats)
    print(f"M1 Decoded Boxes Shape: {list(out['decoded_boxes'].shape)}")
    print(f"M1 Objectness Shape:    {list(out['objectness'].shape)}")
    print(f"M1 Class Logits Shape:  {list(out['class_logits'].shape)}")

    total_preds_per_image = out['decoded_boxes'].shape[1]
    print(f"TOTAL predictions/image: {total_preds_per_image}")
    assert total_preds_per_image == 25200, f"Expected 25200, got {total_preds_per_image}"
    print("RUNTIME VERIFICATION: SUCCESS!")

if __name__ == "__main__":
    main()
