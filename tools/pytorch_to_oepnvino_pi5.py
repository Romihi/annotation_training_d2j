#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TIMMBasedModel (EdgeNeXt + regressor)
→ OpenVINO IR (FP16, Pi 5 safe)

完全に学習時と同一構造
"""

import argparse
import torch
import torch.nn as nn
import timm

from openvino.tools import mo
from openvino.runtime import serialize


# --------------------------------------------------
# 学習時と完全一致するモデル定義
# --------------------------------------------------
class TIMMBasedModelForOV(nn.Module):
    def __init__(self, timm_model_name, num_outputs, input_size):
        super().__init__()

        self.num_outputs = num_outputs

        # backbone
        self.base_model = timm.create_model(
            timm_model_name,
            pretrained=False,
            num_classes=0
        )

        # feature dim を取得
        dummy = torch.zeros(1, 3, input_size, input_size)
        self.base_model.eval()
        with torch.no_grad():
            feat = self.base_model(dummy)
        self.base_model.train()

        feature_dim = feat.shape[1]

        # regressor
        self.regressor = nn.Linear(feature_dim, num_outputs)

    def forward(self, x):
        x = self.base_model(x)
        x = self.regressor(x)
        return x


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="trained .pth")
    parser.add_argument("--output", required=True, help="output path without extension")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_outputs", type=int, default=2)
    args = parser.parse_args()

    S = args.input_size

    print("=== TIMMBasedModel → OpenVINO (FULL MODEL) ===")
    print(f"Model        : edgenext_xx_small")
    print(f"Input size   : {S} x {S}")
    print(f"Num outputs  : {args.num_outputs}")
    print(f"Precision    : FP16")
    print("---------------------------------------------")

    # --------------------------------------------------
    # 1. モデル作成
    # --------------------------------------------------
    print("[1] Creating model")

    model = TIMMBasedModelForOV(
        timm_model_name="edgenext_xx_small",
        num_outputs=args.num_outputs,
        input_size=S
    )

    # --------------------------------------------------
    # 2. checkpoint load（完全一致）
    # --------------------------------------------------
    print("[2] Loading checkpoint")

    checkpoint = torch.load(args.model_path, map_location="cpu")

    state_dict = checkpoint["model_state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=True)

    print(f"Missing keys   : {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    model.eval()

    # --------------------------------------------------
    # 3. TorchScript trace
    # --------------------------------------------------
    print("[3] TorchScript tracing")

    dummy_input = torch.randn(1, 3, S, S)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
        traced.eval()

    # --------------------------------------------------
    # 4. OpenVINO 변환
    # --------------------------------------------------
    print("[4] Converting to OpenVINO IR")

    ov_model = mo.convert_model(
        traced,
        example_input=dummy_input,
        compress_to_fp16=True
    )

    serialize(
        ov_model,
        f"{args.output}.xml",
        f"{args.output}.bin"
    )

    print("---------------------------------------------")
    print("Conversion completed successfully")
    print(f"  - {args.output}.xml")
    print(f"  - {args.output}.bin")
    print("=============================================")


if __name__ == "__main__":
    main()
