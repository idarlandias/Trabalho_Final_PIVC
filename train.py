"""
Script de treinamento para classificação de Pneumonia em Raio-X de Tórax.

Uso:
    python train.py                  # Treina todos os 3 experimentos
    python train.py --exp exp1       # Treina apenas o experimento 1
    python train.py --exp exp2 exp3  # Treina experimentos 2 e 3
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from utils import (
    set_seed,
    get_dataloaders,
    compute_class_weights,
    build_model,
    EarlyStopping,
    plot_training_curves,
    save_metrics,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Treina o modelo por uma época."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Valida o modelo."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def run_experiment(exp_name, exp_config, device):
    """
    Executa um experimento completo de treinamento.

    Args:
        exp_name: identificador do experimento ('exp1', 'exp2', 'exp3')
        exp_config: dicionário com configurações do experimento
        device: dispositivo (cuda/cpu)
    """
    print("\n" + "#" * 70)
    print(f"# {exp_config['name']}")
    print(f"# {exp_config['description']}")
    print("#" * 70)

    # Criar diretório de resultados
    exp_dir = os.path.join(config.RESULTS_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Seed para reprodutibilidade
    set_seed(config.SEED)

    # DataLoaders
    print("\n[1/5] Carregando dados...")
    loaders, datasets_info = get_dataloaders(
        batch_size=exp_config["batch_size"],
        num_workers=config.NUM_WORKERS,
    )

    # Class weights
    print("\n[2/5] Calculando class weights...")
    pos_weight = compute_class_weights(datasets_info["train"]).to(device)

    # Modelo
    print("\n[3/5] Construindo modelo...")
    model = build_model(
        model_name=exp_config["model_name"],
        unfreeze_layers=exp_config["unfreeze_layers"],
    )
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=exp_config["learning_rate"],
        weight_decay=exp_config["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config.LR_SCHEDULER["mode"],
        factor=config.LR_SCHEDULER["factor"],
        patience=config.LR_SCHEDULER["patience"],
        min_lr=config.LR_SCHEDULER["min_lr"],
    )

    # Early Stopping
    best_model_path = os.path.join(exp_dir, "best_model.pth")
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        path=best_model_path,
    )

    # Histórico de treinamento
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    # =================================================================
    # LOOP DE TREINAMENTO
    # =================================================================
    print(f"\n[4/5] Iniciando treinamento ({exp_config['epochs']} épocas máx)...")
    print(f"  Learning Rate: {exp_config['learning_rate']}")
    print(f"  Batch Size:    {exp_config['batch_size']}")
    print(f"  Optimizer:     {exp_config['optimizer']}")
    print("-" * 70)

    start_time = time.time()

    for epoch in range(1, exp_config["epochs"] + 1):
        epoch_start = time.time()

        # Treino
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )

        # Validação
        val_loss, val_acc = validate(model, loaders["val"], criterion, device)

        # Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Histórico
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start

        print(
            f"  Época {epoch:>3}/{exp_config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Tempo: {epoch_time:.1f}s"
        )

        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n  >>> Early Stopping na época {epoch}!")
            break

    total_time = time.time() - start_time
    print(f"\n  Treinamento concluído em {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Melhor val_loss: {early_stopping.best_loss:.4f}")

    # =================================================================
    # SALVAR RESULTADOS
    # =================================================================
    print("\n[5/5] Salvando resultados...")

    # Gráficos de treinamento
    plot_training_curves(history, os.path.join(exp_dir, "training_curves.png"))

    # Salvar histórico
    history_path = os.path.join(exp_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Histórico salvo: {history_path}")

    # Salvar configuração do experimento
    exp_info = {
        "experiment": exp_name,
        "config": exp_config,
        "total_training_time_seconds": total_time,
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": early_stopping.best_loss,
        "seed": config.SEED,
    }
    info_path = os.path.join(exp_dir, "experiment_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(exp_info, f, indent=2, ensure_ascii=False)

    print(f"\n  Resultados salvos em: {exp_dir}")
    print(f"  Melhor modelo salvo em: {best_model_path}")

    return history, best_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Treinamento - Classificação de Pneumonia em Raio-X"
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        default=list(config.EXPERIMENTS.keys()),
        choices=list(config.EXPERIMENTS.keys()),
        help="Experimentos para executar (default: todos)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        help="Dispositivo: cuda ou cpu",
    )
    args = parser.parse_args()

    # Verificar dispositivo
    if args.device == "cuda" and not torch.cuda.is_available():
        print("AVISO: CUDA não disponível. Usando CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"\nDispositivo: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memória GPU: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Criar diretório de resultados
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Executar experimentos
    for exp_name in args.exp:
        exp_config = config.EXPERIMENTS[exp_name]
        run_experiment(exp_name, exp_config, device)

    print("\n" + "=" * 70)
    print("TODOS OS EXPERIMENTOS CONCLUÍDOS!")
    print(f"Resultados em: {config.RESULTS_DIR}")
    print("Execute 'python evaluate.py' para avaliação completa no conjunto de teste.")
    print("=" * 70)


if __name__ == "__main__":
    main()
