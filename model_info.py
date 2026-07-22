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

    # MobileNetV4 variants
    'mobilenetv4_conv_small': {'top1': 74.6, 'top5': 92.0},

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
    'mobilenetv4_conv_small': 0.95,
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
    'mobilenetv4': {
        'title': 'MobileNetV4 - Universal Models for the Mobile Ecosystem',
        'authors': 'Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal, Tenghui Zhu, Daniele Moro, Andrew Howard',
        'year': 2024,
        'url': 'https://arxiv.org/abs/2404.10518'
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
    'mobilenetv4_conv_small': 3.8,
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


# =========================================================================
# 時系列シーケンスモデル情報
# =========================================================================

SEQUENCE_MODEL_INFO = {
    'gru': {
        'name': 'GRU Sequence',
        'description': 'GRUベース時系列シーケンスモデル',
        'backbone': 'MobileNetV3-Small',
        'temporal': 'GRU (Gated Recurrent Unit)',
        'param_count': 3.2,  # 百万単位 (backbone ~2.5M + GRU + head)
        'gflops': 0.12,  # per-frame backbone + GRU overhead
    },
    'tcn': {
        'name': 'TCN Sequence',
        'description': 'Temporal Convolutional Networkベース時系列シーケンスモデル',
        'backbone': 'MobileNetV3-Small',
        'temporal': 'TCN (Dilated Causal Conv1D + Residual)',
        'param_count': 3.5,
        'gflops': 0.15,
    },
    'causal_cnn': {
        'name': 'CausalCNN Sequence',
        'description': '軽量Causal CNNベース時系列シーケンスモデル (TinyLidarNet風)',
        'backbone': 'MobileNetV3-Small',
        'temporal': 'Causal Conv1D Stack',
        'param_count': 3.0,
        'gflops': 0.10,
    },
    'togivad': {
        'name': 'TogiVAD-Nano',
        'description': ('BEV+軌道語彙分類のE2E運転モデル (VADv2式)。'
                        'ラベルは実測自己位置 (pose既定/slam選択可) の将来軌道'),
        'backbone': 'TinyBackbone + IPM + BEVEncoder',
        'temporal': 'なし (単一フレーム + ego status)',
        'param_count': 1.5,
        'gflops': 0.9,
    },
}

SEQUENCE_PAPER_INFO = {
    'gru': {
        'title': 'Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation',
        'authors': 'Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio',
        'year': 2014,
        'url': 'https://arxiv.org/abs/1406.1078',
    },
    'tcn': {
        'title': 'An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling',
        'authors': 'Shaojie Bai, J. Zico Kolter, Vladlen Koltun',
        'year': 2018,
        'url': 'https://arxiv.org/abs/1803.01271',
        'repo': 'https://github.com/locuslab/TCN',
    },
    'causal_cnn': {
        'title': 'TinyLidarNet: 2D LiDAR-based End-to-End Deep Learning Model for F1TENTH Autonomous Racing',
        'authors': 'Mohammed Misbah Zarrar,Jaehyun Kim, Sang-Hyun Park, Jae-Han Park, Seung-Jun Han',
        'year': 2024,
        'url': 'https://arxiv.org/abs/2405.04436',
        'repo': 'https://github.com/CPR-D/TinyLidarNet',
    },
    'togivad': {
        'title': 'VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning',
        'authors': 'Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, Xinggang Wang',
        'year': 2024,
        'url': 'https://arxiv.org/abs/2402.13243',
    },
    'mobilenetv3_backbone': {
        'title': 'Searching for MobileNetV3',
        'authors': 'Andrew Howard et al.',
        'year': 2019,
        'url': 'https://arxiv.org/abs/1905.02244',
        'repo': 'https://github.com/huggingface/pytorch-image-models',
    },
}


def get_sequence_model_info(arch_name):
    """時系列モデルの情報を取得する"""
    return SEQUENCE_MODEL_INFO.get(arch_name, None)

def get_sequence_paper_info(arch_name):
    """時系列モデルの論文情報を取得する"""
    return SEQUENCE_PAPER_INFO.get(arch_name, None)

def list_sequence_architectures():
    """利用可能な時系列モデルアーキテクチャ一覧を返す"""
    return list(SEQUENCE_MODEL_INFO.keys())