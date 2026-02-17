"""
Funções auxiliares do projeto.
Inclui: dataset, transforms, early stopping, visualizações, seed, class weights.
"""

import os
import random
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import config


# =============================================================================
# REPRODUTIBILIDADE
# =============================================================================
def set_seed(seed=config.SEED):
    """Define seed para reprodutibilidade completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# TRANSFORMS (DATA AUGMENTATION)
# =============================================================================
def get_train_transforms():
    """Transforms para treino com data augmentation."""
    aug = config.AUGMENTATION
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
        transforms.RandomRotation(degrees=aug["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=aug["color_jitter"]["brightness"],
            contrast=aug["color_jitter"]["contrast"],
            saturation=aug["color_jitter"]["saturation"],
            hue=aug["color_jitter"]["hue"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_val_test_transforms():
    """Transforms para validação e teste (sem augmentation)."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# =============================================================================
# DATALOADERS
# =============================================================================
def get_dataloaders(batch_size, num_workers=config.NUM_WORKERS):
    """
    Cria DataLoaders para treino, validação e teste.

    O dataset original possui um conjunto de validação muito pequeno (16 imagens).
    Mantemos a estrutura original de pastas.

    Retorna:
        dict com chaves 'train', 'val', 'test' contendo os DataLoaders.
        dict com chaves 'train', 'val', 'test' contendo os datasets.
    """
    train_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR, transform=get_train_transforms()
    )
    val_dataset = datasets.ImageFolder(
        root=config.VAL_DIR, transform=get_val_test_transforms()
    )
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR, transform=get_val_test_transforms()
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataset_info = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    # Informações do dataset
    print("=" * 60)
    print("INFORMAÇÕES DO DATASET")
    print("=" * 60)
    for split_name, ds in dataset_info.items():
        class_counts = {}
        for _, label in ds.samples:
            class_name = config.CLASSES[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        total = len(ds)
        print(f"  {split_name:>6}: {total:>5} imagens | ", end="")
        for cls, count in sorted(class_counts.items()):
            print(f"{cls}: {count} ({100*count/total:.1f}%) ", end="")
        print()
    print("=" * 60)

    return loaders, dataset_info


# =============================================================================
# CLASS WEIGHTS (para lidar com desbalanceamento)
# =============================================================================
def compute_class_weights(dataset):
    """
    Calcula pesos das classes inversamente proporcionais à frequência.
    Retorna tensor com peso para a classe positiva (PNEUMONIA) no BCEWithLogitsLoss.
    """
    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels)
    total = sum(class_counts)
    # Peso para classe positiva: n_negativas / n_positivas
    # Isso equilibra a contribuição de ambas as classes no loss
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32)
    print(f"  Class weights - NORMAL: {class_counts[0]}, PNEUMONIA: {class_counts[1]}")
    print(f"  pos_weight para BCEWithLogitsLoss: {pos_weight.item():.4f}")
    return pos_weight


# =============================================================================
# MODELO
# =============================================================================
def build_model(model_name, unfreeze_layers=0):
    """
    Constrói modelo com transfer learning.

    Args:
        model_name: 'resnet50' ou 'densenet121'
        unfreeze_layers: número de camadas finais para descongelar (0 = todas congeladas)

    Returns:
        modelo PyTorch pronto para treinamento
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features

        # Congelar todas as camadas
        for param in model.parameters():
            param.requires_grad = False

        # Descongelar últimas N camadas (se especificado)
        if unfreeze_layers > 0:
            all_params = list(model.named_parameters())
            for name, param in all_params[-unfreeze_layers:]:
                param.requires_grad = True

        # Substituir classificador final
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # Saída binária (sem sigmoid - usamos BCEWithLogitsLoss)
        )

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features

        # Congelar todas as camadas
        for param in model.parameters():
            param.requires_grad = False

        # Descongelar últimas N camadas
        if unfreeze_layers > 0:
            all_params = list(model.named_parameters())
            for name, param in all_params[-unfreeze_layers:]:
                param.requires_grad = True

        # Substituir classificador final
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    else:
        raise ValueError(f"Modelo '{model_name}' não suportado. Use 'resnet50' ou 'densenet121'.")

    # Contar parâmetros treináveis vs totais
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Modelo: {model_name}")
    print(f"  Parâmetros totais:     {total_params:>12,}")
    print(f"  Parâmetros treináveis: {trainable_params:>12,}")
    print(f"  Parâmetros congelados: {total_params - trainable_params:>12,}")
    print(f"  Camadas descongeladas: {unfreeze_layers}")

    return model


# =============================================================================
# EARLY STOPPING
# =============================================================================
class EarlyStopping:
    """
    Interrompe o treinamento quando a métrica de validação para de melhorar.

    Args:
        patience: número de épocas sem melhora antes de parar
        min_delta: melhora mínima para considerar como progresso
        path: caminho para salvar o melhor modelo
    """

    def __init__(self, patience=config.EARLY_STOPPING_PATIENCE, min_delta=0.0, path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Salva o modelo com melhor val_loss."""
        torch.save(model.state_dict(), self.path)


# =============================================================================
# VISUALIZAÇÕES
# =============================================================================
def plot_training_curves(history, save_path):
    """Plota curvas de loss e accuracy durante o treinamento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Treino", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validação", linewidth=2)
    axes[0].set_title("Loss por Época", fontsize=14)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss (BCE)")
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Treino", linewidth=2)
    axes[1].plot(history["val_acc"], label="Validação", linewidth=2)
    axes[1].set_title("Accuracy por Época", fontsize=14)
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico salvo: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plota matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        annot_kws={"size": 16},
    )
    plt.title("Matriz de Confusão", fontsize=14)
    plt.ylabel("Rótulo Verdadeiro", fontsize=12)
    plt.xlabel("Rótulo Predito", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Matriz de confusão salva: {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    """Plota curva ROC com AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Aleatório")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontsize=12)
    plt.ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontsize=12)
    plt.title("Curva ROC", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curva ROC salva: {save_path}")
    return roc_auc


def plot_prediction_examples(model, dataset, device, save_path, num_examples=8):
    """
    Plota exemplos de predições corretas e incorretas.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    correct_examples = []
    incorrect_examples = []

    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)

    with torch.no_grad():
        for images, labels in loader:
            if len(correct_examples) >= num_examples and len(incorrect_examples) >= num_examples:
                break

            images_dev = images.to(device)
            outputs = model(images_dev)
            probs = torch.sigmoid(outputs).cpu()
            preds = (probs >= 0.5).long().squeeze()
            label = labels.item()
            pred = preds.item()
            prob = probs.item()

            # Desnormalizar imagem para visualização
            img = images[0].cpu() * std + mean
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()

            entry = (img, label, pred, prob)
            if pred == label and len(correct_examples) < num_examples:
                correct_examples.append(entry)
            elif pred != label and len(incorrect_examples) < num_examples:
                incorrect_examples.append(entry)

    # Plotar
    n_correct = min(len(correct_examples), 4)
    n_incorrect = min(len(incorrect_examples), 4)
    total = n_correct + n_incorrect

    if total == 0:
        return

    fig, axes = plt.subplots(2, max(n_correct, n_incorrect, 1), figsize=(4 * max(n_correct, n_incorrect, 1), 8))
    if max(n_correct, n_incorrect, 1) == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Exemplos de Predições", fontsize=16, fontweight="bold")

    # Predições corretas
    for i in range(axes.shape[1]):
        if i < n_correct:
            img, label, pred, prob = correct_examples[i]
            axes[0, i].imshow(img)
            axes[0, i].set_title(
                f"Real: {config.CLASSES[label]}\n"
                f"Pred: {config.CLASSES[pred]} ({prob:.2f})",
                color="green", fontsize=10,
            )
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("CORRETAS", fontsize=12, fontweight="bold", rotation=0, labelpad=70)

    # Predições incorretas
    for i in range(axes.shape[1]):
        if i < n_incorrect:
            img, label, pred, prob = incorrect_examples[i]
            axes[1, i].imshow(img)
            axes[1, i].set_title(
                f"Real: {config.CLASSES[label]}\n"
                f"Pred: {config.CLASSES[pred]} ({prob:.2f})",
                color="red", fontsize=10,
            )
        axes[1, i].axis("off")
    axes[1, 0].set_ylabel("INCORRETAS", fontsize=12, fontweight="bold", rotation=0, labelpad=70)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Exemplos de predições salvos: {save_path}")


# =============================================================================
# MÉTRICAS
# =============================================================================
def compute_metrics(y_true, y_pred, y_probs):
    """
    Calcula todas as métricas médicas relevantes.

    Métricas:
        - Accuracy:    (TP + TN) / (TP + TN + FP + FN)
        - Precision:   TP / (TP + FP)
        - Recall/Sensitivity: TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1-Score:    2 * (Precision * Recall) / (Precision + Recall)
        - AUC-ROC:     Área sob a curva ROC

    Onde:
        - Positivo = PNEUMONIA (label 1)
        - Negativo = NORMAL (label 0)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_roc = auc(fpr, tpr)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,  # Recall
        "specificity": specificity,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
    return metrics


def print_metrics(metrics):
    """Exibe métricas formatadas no console."""
    print("\n" + "=" * 50)
    print("MÉTRICAS DE AVALIAÇÃO")
    print("=" * 50)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}  (Recall)")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print("-" * 50)
    print(f"  TP: {metrics['true_positives']:>4}  |  FP: {metrics['false_positives']:>4}")
    print(f"  FN: {metrics['false_negatives']:>4}  |  TN: {metrics['true_negatives']:>4}")
    print("=" * 50)


def save_metrics(metrics, history, exp_config, save_dir):
    """Salva métricas e configurações em JSON."""
    results = {
        "experiment": exp_config,
        "metrics": metrics,
        "training_history": {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
            "epochs_trained": len(history["train_loss"]),
        },
    }
    path = os.path.join(save_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Métricas salvas: {path}")
