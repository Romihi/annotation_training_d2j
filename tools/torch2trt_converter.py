#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
非推奨: tools/torch2trt_converter.py を使用してください。
"""
import warnings
warnings.warn(
    "annotation_training_d2j/tools/torch2trt_converter.py は非推奨です。"
    " tools/torch2trt_converter.py を使用してください。",
    DeprecationWarning,
    stacklevel=2
)

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
from torch2trt_converter import (
    convert_pytorch_to_tensorrt,
    convert_sequence_to_tensorrt,
    benchmark_inference,
    load_model_weights,
    find_pytorch_models,
    get_available_model_types,
    infer_model_type_from_filename,
)
