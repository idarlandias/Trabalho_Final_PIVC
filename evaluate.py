"""
Script de avaliação para classificação de Pneumonia em Raio-X de Tórax.

Avalia os modelos treinados no conjunto de teste e gera:
- Métricas completas (Accuracy, Precision, Sensitivity, Specificity, F1, AUC-ROC)
- Matriz de confusão
- Curva ROC
- Exemplos de predições (corretas e incorretas)
- Tabela comparativa dos experimentos (comparison.csv)

Uso:
    python evaluate.py                  # Avalia todos os experimentos
    python evaluate.py --exp exp1       # Avalia apenas o experimento 1
    python evaluate.py --exp exp1 exp2  # Avalia experimentos 1 e 2
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

import config
from utils import (
    set_seed,
    get_val_test_transforms,
    build_model,
    compute_metrics,
    print_metrics,
    save_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    plot_prediction_examples,
)


def evaluate_model(model, loader, device):
    """
    Avalia o modelo no conjunto de teste.

    Retorna:
        y_true: rótulos reais
        y_pred: predições binárias
        y_probs: probabilidades (para curva ROC)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_experiment(exp_name, exp_config, device):
    """
    Avalia um experimento completo.

    Args:
        exp_name: identificador do experimento
        exp_config: configurações do experimento
        device: dispositivo (cuda/cpu)

    Returns:
        dict com métricas do experimento
    """
    exp_dir = os.path.join(config.RESULTS_DIR, exp_name)
    model_path = os.path.join(exp_dir, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"\n  ERRO: Modelo não encontrado: {model_path}")
        print(f"  Execute 'python train.py --exp {exp_name}' primeiro.")
        return None

    print("\n" + "#" * 70)
    print(f"# AVALIAÇÃO: {exp_config['name']}")
    print("#" * 70)

    set_seed(config.SEED)

    # Carregar dataset de teste
    print("\n[1/5] Carregando dataset de teste...")
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=get_val_test_transforms(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=exp_config["batch_size"],
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    print(f"  Total de imagens de teste: {len(test_dataset)}")

    # Construir e carregar modelo
    print("\n[2/5] Carregando modelo treinado...")
    model = build_model(
        model_name=exp_config["model_name"],
        unfreeze_layers=exp_config["unfreeze_layers"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"  Modelo carregado de: {model_path}")

    # Avaliar no conjunto de teste
    print("\n[3/5] Avaliando no conjunto de teste...")
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)

    # Calcular métricas
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print_metrics(metrics)

    # Gerar visualizações
    print("\n[4/5] Gerando visualizações...")

    # Matriz de confusão
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(exp_dir, "confusion_matrix.png"),
    )

    # Curva ROC
    plot_roc_curve(
        y_true, y_probs,
        save_path=os.path.join(exp_dir, "roc_curve.png"),
    )

    # Exemplos de predições
    plot_prediction_examples(
        model, test_dataset, device,
        save_path=os.path.join(exp_dir, "prediction_examples.png"),
        num_examples=8,
    )

    # Replotar curvas de treinamento (se histórico existir)
    history_path = os.path.join(exp_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        plot_training_curves(history, os.path.join(exp_dir, "training_curves.png"))

    # Salvar métricas
    print("\n[5/5] Salvando métricas...")
    history = {}
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    save_metrics(metrics, history, exp_config, exp_dir)

    return metrics


def create_comparison_table(all_results):
    """
    Cria tabela comparativa dos experimentos e salva como CSV.

    Args:
        all_results: dict {exp_name: metrics}
    """
    print("\n" + "#" * 70)
    print("# TABELA COMPARATIVA DOS EXPERIMENTOS")
    print("#" * 70)

    rows = []
    for exp_name, metrics in all_results.items():
        exp_config = config.EXPERIMENTS[exp_name]
        row = {
            "Experimento": exp_config["name"],
            "Modelo": exp_config["model_name"],
            "Learning Rate": exp_config["learning_rate"],
            "Batch Size": exp_config["batch_size"],
            "Épocas": exp_config["epochs"],
            "Camadas Descongeladas": exp_config["unfreeze_layers"],
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Sensitivity": f"{metrics['sensitivity']:.4f}",
            "Specificity": f"{metrics['specificity']:.4f}",
            "F1-Score": f"{metrics['f1_score']:.4f}",
            "AUC-ROC": f"{metrics['auc_roc']:.4f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Salvar CSV
    csv_path = os.path.join(config.RESULTS_DIR, "comparison.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  Tabela salva em: {csv_path}")

    # Exibir no console
    print("\n" + df.to_string(index=False))

    # Identificar melhor experimento por métrica
    print("\n" + "-" * 70)
    print("MELHOR EXPERIMENTO POR MÉTRICA:")
    metric_cols = ["Accuracy", "Precision", "Sensitivity", "Specificity", "F1-Score", "AUC-ROC"]
    for col in metric_cols:
        df[col] = df[col].astype(float)
        best_idx = df[col].idxmax()
        best_exp = df.loc[best_idx, "Experimento"]
        best_val = df.loc[best_idx, col]
        print(f"  {col:>13}: {best_val:.4f} ({best_exp})")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação - Classificação de Pneumonia em Raio-X"
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        default=list(config.EXPERIMENTS.keys()),
        choices=list(config.EXPERIMENTS.keys()),
        help="Experimentos para avaliar (default: todos)",
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

    # Avaliar cada experimento
    all_results = {}
    for exp_name in args.exp:
        exp_config = config.EXPERIMENTS[exp_name]
        metrics = evaluate_experiment(exp_name, exp_config, device)
        if metrics is not None:
            all_results[exp_name] = metrics

    # Tabela comparativa (se houver mais de um experimento)
    if len(all_results) > 1:
        create_comparison_table(all_results)
    elif len(all_results) == 1:
        print("\n  Apenas 1 experimento avaliado. Treine mais para gerar a tabela comparativa.")

    print("\n" + "=" * 70)
    print("AVALIAÇÃO CONCLUÍDA!")
    print(f"Resultados em: {config.RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
