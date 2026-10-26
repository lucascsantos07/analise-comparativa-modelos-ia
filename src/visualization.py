"""
Módulo de visualização e geração de gráficos (reports/figures/).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

# Estilo padrão limpo
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def plotar_matriz_confusao(
    y_true,
    y_pred,
    classes=None,
    titulo="Matriz de Confusão",
    arquivo_saida=None,
):
    """Gera e opcionalmente salva a matriz de confusão."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(titulo, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    if arquivo_saida:
        Path(arquivo_saida).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(arquivo_saida, dpi=300)
    return fig, ax


def plotar_curvas_roc_pr(
    y_true,
    modelos_proba: dict[str, np.ndarray],
    classes: list,
    classe_foco: str = "High",
    arquivo_saida=None,
):
    """
    Plota curvas ROC e Precision-Recall lado a lado para múltiplos modelos
    focando na classe de maior interesse clínico ou binarizada.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y_bin = (np.array(y_true) == classe_foco).astype(int)
    idx_classe = classes.index(classe_foco) if classe_foco in classes else 0

    for nome, y_proba in modelos_proba.items():
        if y_proba is None:
            continue
        scores = y_proba[:, idx_classe] if y_proba.ndim > 1 else y_proba
        
        # ROC
        fpr, tpr, _ = roc_curve(y_bin, scores)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{nome} (AUC = {roc_auc:.3f})", linewidth=2)

        # PR: a área trapezoidal superestima o desempenho; usamos a
        # precisão média (average precision), coerente com src/metrics.py.
        prec, rec, _ = precision_recall_curve(y_bin, scores)
        pr_ap = average_precision_score(y_bin, scores)
        axes[1].plot(rec, prec, label=f"{nome} (AP = {pr_ap:.3f})", linewidth=2)

    # Configuração do gráfico ROC
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Aleatório")
    axes[0].set_title(f"Curva ROC — Classe '{classe_foco}'", fontweight="bold")
    axes[0].set_xlabel("Taxa de Falsos Positivos (FPR)")
    axes[0].set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    axes[0].legend(loc="lower right")

    # Configuração do gráfico PR
    prevalencia = float(np.mean(y_bin))
    axes[1].axhline(y=prevalencia, color="k", linestyle="--", alpha=0.6, label=f"Baseline ({prevalencia:.2f})")
    axes[1].set_title(f"Curva Precision-Recall — Classe '{classe_foco}'", fontweight="bold")
    axes[1].set_xlabel("Revocação (Recall)")
    axes[1].set_ylabel("Precisão")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    if arquivo_saida:
        Path(arquivo_saida).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(arquivo_saida, dpi=300)
    return fig, axes


def plotar_residuos(
    y_true,
    y_pred,
    titulo="Análise de Resíduos",
    arquivo_saida=None,
):
    """Plota valores reais vs previstos e dispersão de resíduos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    residuos = np.array(y_true) - np.array(y_pred)

    # 1. Real vs Previsto
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.35, color="steelblue", ax=axes[0])
    lim_min = min(np.min(y_true), np.min(y_pred))
    lim_max = max(np.max(y_true), np.max(y_pred))
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=2, label="Ideal")
    axes[0].set_title(f"{titulo} — Real vs Previsto", fontweight="bold")
    axes[0].set_xlabel("Valor Real")
    axes[0].set_ylabel("Valor Previsto")
    axes[0].legend()

    # 2. Resíduos vs Previstos
    sns.scatterplot(x=y_pred, y=residuos, alpha=0.35, color="teal", ax=axes[1])
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title(f"{titulo} — Resíduos vs Previstos", fontweight="bold")
    axes[1].set_xlabel("Valor Previsto")
    axes[1].set_ylabel("Resíduo (Real - Previsto)")

    plt.tight_layout()
    if arquivo_saida:
        Path(arquivo_saida).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(arquivo_saida, dpi=300)
    return fig, axes


def plotar_importancia_features(
    importancias: pd.Series,
    titulo="Importância das Features",
    top_n=10,
    arquivo_saida=None,
):
    """Plota gráfico de barras horizontais com as features mais importantes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    top_features = importancias.head(top_n).sort_values(ascending=True)

    top_features.plot(kind="barh", color="#1f77b4", ax=ax)
    ax.set_title(titulo, fontweight="bold", pad=12)
    ax.set_xlabel("Importância Relativa (Gini/Impureza)")
    plt.tight_layout()

    if arquivo_saida:
        Path(arquivo_saida).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(arquivo_saida, dpi=300)
    return fig, ax
