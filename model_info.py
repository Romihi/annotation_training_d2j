"""
モデル情報 - 各モデルの精度情報などのメタデータ
"""

# ImageNetデータセットでの精度情報
MODEL_ACCURACY_INFO = {
    # MobileViT variants
    'mobilevit_xxs': {'top1': 69.0, 'top5': 89.5},
    'mobilevit_xs': {'top1': 74.8, 'top5': 92.4},
    'mobilevit_s': {'top1': 78.4, 'top5': 94.2},
    
    # MobileNetV3 variants
    'mobilenetv3_small_100': {'top1': 67.4, 'top5': 87.4},
    'mobilenetv3_large_100': {'top1': 75.2, 'top5': 92.2},
    
    # EfficientNet variants
    'efficientnet_lite0': {'top1': 75.1, 'top5': 92.5},
    'efficientnet_b0': {'top1': 77.1, 'top5': 93.3},
    
    # ResNet variants
    'resnet18': {'top1': 69.8, 'top5': 89.1},
    'resnet34': {'top1': 73.3, 'top5': 91.4},
    
    # ConvNeXt variants
    'convnext_nano': {'top1': 80.5, 'top5': 95.1},
    'convnext_tiny': {'top1': 82.1, 'top5': 95.9},
    
    # EfficientFormer variants
    'efficientformer_l1': {'top1': 79.2, 'top5': 94.3},
    
    # EdgeNeXt variants
    'edgenext_xx_small': {'top1': 71.2, 'top5': 89.9},
    'edgenext_x_small': {'top1': 74.9, 'top5': 92.3},
    
    # MobileOne variants
    'mobileone_s0': {'top1': 71.4, 'top5': 90.2},
    
    # MobileViT v2 (if available)
    'mobilevitv2_050': {'top1': 70.2, 'top5': 89.7},
    
    # GhostNet
    'ghostnet_050': {'top1': 66.2, 'top5': 86.6},
    
    # ShuffleNetV2
    'shufflenetv2_x0_5': {'top1': 60.6, 'top5': 81.8},
    
    # Swin Transformer variants
    'swin_tiny_patch4_window7_224': {'top1': 81.3, 'top5': 95.5},
    'swin_s3_tiny_224': {'top1': 81.3, 'top5': 95.7},
    'swinv2_cr_tiny_ns_224': {'top1': 81.5, 'top5': 95.8},
    'swin_moe_tiny_patch4_window7_224': {'top1': 82.2, 'top5': 96.0},
}

# モデルのデフォルト入力サイズ
MODEL_INPUT_SIZE = {
    # デフォルトは224x224
    'default': (224, 224),
    
    # 一部のモデルは異なる入力サイズを持つ場合がある
    'mobilevit_xxs': (256, 256),
    'mobilevit_xs': (256, 256),
    'mobilevit_s': (256, 256),
}

# モデルの計算量情報 (GFLOPs)
MODEL_COMPUTE_INFO = {
    'mobilevit_xxs': 0.4,
    'mobilevit_xs': 0.8,
    'mobilevit_s': 1.8,
    'mobilenetv3_small_100': 0.06,
    'mobilenetv3_large_100': 0.22,
    'efficientnet_lite0': 0.4,
    'efficientnet_b0': 0.4,
    'resnet18': 1.8,
    'resnet34': 3.6,
    'convnext_nano': 0.6,
    'convnext_tiny': 4.5,
    'edgenext_xx_small': 0.26,
    'edgenext_x_small': 0.54,
    'ghostnet_050': 0.05,
    'shufflenetv2_x0_5': 0.04,
    'swin_tiny_patch4_window7_224': 4.5,
}

# モデルの論文情報
MODEL_PAPER_INFO = {
    'mobilevit': {
        'title': 'MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer',
        'authors': 'Sachin Mehta, Mohammad Rastegari',
        'year': 2021,
        'url': 'https://arxiv.org/abs/2110.02178'
    },
    'mobilenetv3': {
        'title': 'Searching for MobileNetV3',
        'authors': 'Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam',
        'year': 2019,
        'url': 'https://arxiv.org/abs/1905.02244'
    },
    'swin': {
        'title': 'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows',
        'authors': 'Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo',
        'year': 2021,
        'url': 'https://arxiv.org/abs/2103.14030'
    },
    'edgenext': {
        'title': 'EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications',
        'authors': 'Muhammad Maaz, Abdelrahman Shaker, Hisham Cholakkal, Salman Khan, Syed Waqas Zamir, Fahad Shahbaz Khan',
        'year': 2022,
        'url': 'https://arxiv.org/abs/2206.10589'
    },
    'resnet': {
        'title': 'Deep Residual Learning for Image Recognition',
        'authors': 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun',
        'year': 2015,
        'url': 'https://arxiv.org/abs/1512.03385'
    },
    'efficientnet': {
        'title': 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks',
        'authors': 'Mingxing Tan, Quoc V. Le',
        'year': 2019,
        'url': 'https://arxiv.org/abs/1905.11946'
    },
}

# モデルのパラメータ数（百万単位）
MODEL_PARAM_COUNTS = {
    'mobilevit_xxs': 1.3,
    'mobilevit_xs': 2.3,
    'mobilevit_s': 5.6,
    'mobilenetv3_small_100': 2.5,
    'mobilenetv3_large_100': 5.5,
    'efficientnet_lite0': 4.7,
    'efficientnet_b0': 5.3,
    'resnet18': 11.7,
    'resnet34': 21.8,
    'convnext_nano': 15.6,
    'convnext_tiny': 28.6,
    'efficientformer_l1': 12.3,
    'edgenext_xx_small': 1.3,
    'edgenext_x_small': 2.3,
    'mobileone_s0': 5.3,
    'ghostnet_050': 2.6,
    'shufflenetv2_x0_5': 1.4,
    'swin_tiny_patch4_window7_224': 28.0,
}

def get_model_input_size(model_name):
    """モデルの入力サイズを取得する"""
    return MODEL_INPUT_SIZE.get(model_name, MODEL_INPUT_SIZE['default'])

def get_model_compute(model_name):
    """モデルの計算量を取得する"""
    return MODEL_COMPUTE_INFO.get(model_name, None)

def get_model_accuracy(model_name):
    """モデルの精度情報を取得する"""
    return MODEL_ACCURACY_INFO.get(model_name, {'top1': 0, 'top5': 0})

def get_paper_info(model_family):
    """モデルファミリーの論文情報を取得する"""
    return MODEL_PAPER_INFO.get(model_family, None)

def get_param_count(model_name):
    """モデルのパラメータ数を取得する"""
    return MODEL_PARAM_COUNTS.get(model_name, None)
"""
モデル情報 - 各モデルの精度情報などのメタデータ
"""

# ImageNetデータセットでの精度情報
MODEL_ACCURACY_INFO = {
    # MobileViT variants
    'mobilevit_xxs': {'top1': 69.0, 'top5': 89.5},
    'mobilevit_xs': {'top1': 74.8, 'top5': 92.4},
    'mobilevit_s': {'top1': 78.4, 'top5': 94.2},
    
    # MobileNetV3 variants
    'mobilenetv3_small_100': {'top1': 67.4, 'top5': 87.4},
    'mobilenetv3_large_100': {'top1': 75.2, 'top5': 92.2},
    
    # EfficientNet variants
    'efficientnet_lite0': {'top1': 75.1, 'top5': 92.5},
    'efficientnet_b0': {'top1': 77.1, 'top5': 93.3},  # 追加
    
    # ResNet variants
    'resnet18': {'top1': 69.8, 'top5': 89.1},  # 追加
    'resnet34': {'top1': 73.3, 'top5': 91.4},  # 追加
    
    # ConvNeXt variants
    'convnext_nano': {'top1': 80.5, 'top5': 95.1},
    'convnext_tiny': {'top1': 82.1, 'top5': 95.9},
    
    # EfficientFormer variants
    'efficientformer_l1': {'top1': 79.2, 'top5': 94.3},
    
    # EdgeNeXt variants
    'edgenext_xx_small': {'top1': 71.2, 'top5': 89.9},
    'edgenext_x_small': {'top1': 74.9, 'top5': 92.3},
    
    # MobileOne variants
    'mobileone_s0': {'top1': 71.4, 'top5': 90.2},
    
    # MobileViT v2 (if available)
    'mobilevitv2_050': {'top1': 70.2, 'top5': 89.7},
    
    # GhostNet
    'ghostnet_050': {'top1': 66.2, 'top5': 86.6},
    
    # ShuffleNetV2
    'shufflenetv2_x0_5': {'top1': 60.6, 'top5': 81.8},
    
    # Swin Transformer variants
    'swin_tiny_patch4_window7_224': {'top1': 81.3, 'top5': 95.5},
    'swin_s3_tiny_224': {'top1': 81.3, 'top5': 95.7},
    'swinv2_cr_tiny_ns_224': {'top1': 81.5, 'top5': 95.8},
    'swin_moe_tiny_patch4_window7_224': {'top1': 82.2, 'top5': 96.0},
}

# モデルの計算量情報にも欠けている情報を追加
MODEL_COMPUTE_INFO = {
    'mobilevit_xxs': 0.4,
    'mobilevit_xs': 0.8,
    'mobilevit_s': 1.8,
    'mobilenetv3_small_100': 0.06,
    'mobilenetv3_large_100': 0.22,
    'efficientnet_lite0': 0.4,
    'efficientnet_b0': 0.4,  # 追加
    'resnet18': 1.8,  # 追加
    'resnet34': 3.6,  # 追加
    'convnext_nano': 0.6,
    'convnext_tiny': 4.5,
    'edgenext_xx_small': 0.26,
    'edgenext_x_small': 0.54,
    'ghostnet_050': 0.05,
    'shufflenetv2_x0_5': 0.04,
    'swin_tiny_patch4_window7_224': 4.5,
}

# 論文情報にも欠けている情報を追加
MODEL_PAPER_INFO = {
    'mobilevit': {
        'title': 'MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer',
        'authors': 'Sachin Mehta, Mohammad Rastegari',
        'year': 2021,
        'url': 'https://arxiv.org/abs/2110.02178'
    },
    'mobilenetv3': {
        'title': 'Searching for MobileNetV3',
        'authors': 'Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam',
        'year': 2019,
        'url': 'https://arxiv.org/abs/1905.02244'
    },
    'swin': {
        'title': 'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows',
        'authors': 'Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo',
        'year': 2021,
        'url': 'https://arxiv.org/abs/2103.14030'
    },
    'edgenext': {
        'title': 'EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications',
        'authors': 'Muhammad Maaz, Abdelrahman Shaker, Hisham Cholakkal, Salman Khan, Syed Waqas Zamir, Fahad Shahbaz Khan',
        'year': 2022,
        'url': 'https://arxiv.org/abs/2206.10589'
    },
    'resnet': {  # 追加
        'title': 'Deep Residual Learning for Image Recognition',
        'authors': 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun',
        'year': 2015,
        'url': 'https://arxiv.org/abs/1512.03385'
    },
    'efficientnet': {  # 追加
        'title': 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks',
        'authors': 'Mingxing Tan, Quoc V. Le',
        'year': 2019,
        'url': 'https://arxiv.org/abs/1905.11946'
    },
}

# パラメータ数情報も追加しておく（二つ目のファイルには存在しない）
MODEL_PARAM_COUNTS = {
    'mobilevit_xxs': 1.3,
    'mobilevit_xs': 2.3,
    'mobilevit_s': 5.6,
    'mobilenetv3_small_100': 2.5,
    'mobilenetv3_large_100': 5.5,
    'efficientnet_lite0': 4.7,
    'efficientnet_b0': 5.3,
    'resnet18': 11.7,
    'resnet34': 21.8,
    'convnext_nano': 15.6,
    'convnext_tiny': 28.6,
    'efficientformer_l1': 12.3,
    'edgenext_xx_small': 1.3,
    'edgenext_x_small': 2.3,
    'mobileone_s0': 5.3,
    'ghostnet_050': 2.6,
    'shufflenetv2_x0_5': 1.4,
    'swin_tiny_patch4_window7_224': 28.0,
}

def get_model_input_size(model_name):
    """モデルの入力サイズを取得する"""
    return MODEL_INPUT_SIZE.get(model_name, MODEL_INPUT_SIZE['default'])

def get_model_compute(model_name):
    """モデルの計算量を取得する"""
    return MODEL_COMPUTE_INFO.get(model_name, None)

def get_model_accuracy(model_name):
    """モデルの精度情報を取得する"""
    return MODEL_ACCURACY_INFO.get(model_name, {'top1': 0, 'top5': 0})

def get_paper_info(model_family):
    """モデルファミリーの論文情報を取得する"""
    return MODEL_PAPER_INFO.get(model_family, None)