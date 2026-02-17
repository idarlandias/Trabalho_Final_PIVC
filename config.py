"""
Configurações e hiperparâmetros do projeto.
Classificação de Pneumonia em Raio-X de Tórax usando Transfer Learning.
"""

import os

# =============================================================================
# SEED PARA REPRODUTIBILIDADE
# =============================================================================
SEED = 42

# =============================================================================
# CAMINHOS DO PROJETO
# =============================================================================
# Caminho raiz do dataset (Chest X-Ray Images - Pneumonia)
# Fonte: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "chest_xray")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Diretório de resultados
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# =============================================================================
# CLASSES
# =============================================================================
CLASSES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 1  # Saída binária (sigmoid)

# =============================================================================
# PARÂMETROS DE IMAGEM
# =============================================================================
IMG_SIZE = 224  # Tamanho padrão para modelos pré-treinados no ImageNet
NUM_CHANNELS = 3

# Normalização ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# DATA AUGMENTATION (apenas para treino)
# =============================================================================
AUGMENTATION = {
    "horizontal_flip_p": 0.5,
    "rotation_degrees": 15,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.0,
        "hue": 0.0,
    },
}

# =============================================================================
# TREINAMENTO - PARÂMETROS GERAIS
# =============================================================================
NUM_WORKERS = 0  # 0 para Windows (evita problemas com multiprocessing)
DEVICE = "cuda"  # "cuda" ou "cpu" (fallback automático para cpu)

# Early Stopping
EARLY_STOPPING_PATIENCE = 10

# Learning Rate Scheduler (ReduceLROnPlateau)
LR_SCHEDULER = {
    "mode": "min",
    "factor": 0.1,
    "patience": 5,
    "min_lr": 1e-7,
}

# =============================================================================
# DEFINIÇÃO DOS 3 EXPERIMENTOS
# =============================================================================
EXPERIMENTS = {
    "exp1": {
        "name": "Experimento 1 - ResNet50 (Feature Extraction)",
        "model_name": "resnet50",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 30,
        "unfreeze_layers": 0,  # Todas congeladas (feature extraction puro)
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "description": (
            "ResNet50 com todas as camadas congeladas. "
            "Apenas o classificador final é treinado (Feature Extraction)."
        ),
    },
    "exp2": {
        "name": "Experimento 2 - ResNet50 (Fine-Tuning Parcial)",
        "model_name": "resnet50",
        "learning_rate": 0.0001,
        "batch_size": 32,
        "epochs": 50,
        "unfreeze_layers": 10,  # Últimas 10 camadas descongeladas
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "description": (
            "ResNet50 com as últimas 10 camadas descongeladas. "
            "Fine-tuning parcial com learning rate menor."
        ),
    },
    "exp3": {
        "name": "Experimento 3 - DenseNet121 (Fine-Tuning Parcial)",
        "model_name": "densenet121",
        "learning_rate": 0.0001,
        "batch_size": 16,
        "epochs": 50,
        "unfreeze_layers": 20,  # Últimas 20 camadas descongeladas
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "description": (
            "DenseNet121 com as últimas 20 camadas descongeladas. "
            "Fine-tuning parcial com batch size menor."
        ),
    },
}
