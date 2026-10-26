"""Métricas de avaliação (Aula 08).

Reúne as métricas obrigatórias para regressão e classificação. As
métricas multiclasse são calculadas com média macro (todas as classes
pesam igual) e weighted (proporcional ao suporte), para que o relatório
possa discutir o efeito do desbalanceamento.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def calcular_metricas_regressao(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """MAE, MSE, RMSE, MAPE (%), R² e MedAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAPE (%)": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
    }


def calcular_metricas_classificacao(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
    classes: list[Any] | None = None,
    classe_positiva: str | None = None,
) -> dict[str, float]:
    """Métricas de classificação multiclasse.

    Inclui accuracy, precision/recall/F1 macro e weighted e, quando há
    probabilidades, ROC-AUC e PR-AUC no esquema one-vs-rest.

    A PR-AUC usa ``average_precision_score`` (soma das precisões por
    limiar), que é a definição correta para curvas PR e não a área
    trapezoidal — esta superestima o desempenho em classes raras.

    Parameters
    ----------
    y_true, y_pred : array-like
        Rótulos verdadeiros e previstos.
    y_proba : ndarray, optional
        Matriz (n_amostras, n_classes) com probabilidades.
    classes : list, optional
        Ordem das classes (deve casar com as colunas de ``y_proba``).
    classe_positiva : str, optional
        Se informada, acrescenta precision/recall/F1 dessa classe.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))
    classes = list(classes)

    res: dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (Macro)": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Recall (Macro)": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "F1-Score (Macro)": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Precision (Weighted)": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall (Weighted)": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1-Score (Weighted)": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    if classe_positiva is not None:
        res[f"Precision ({classe_positiva})"] = float(
            precision_score(
                y_true, y_pred, labels=[classe_positiva],
                average="macro", zero_division=0,
            )
        )
        res[f"Recall ({classe_positiva})"] = float(
            recall_score(
                y_true, y_pred, labels=[classe_positiva],
                average="macro", zero_division=0,
            )
        )
        res[f"F1-Score ({classe_positiva})"] = float(
            f1_score(
                y_true, y_pred, labels=[classe_positiva],
                average="macro", zero_division=0,
            )
        )

    if y_proba is not None and len(classes) > 1:
        y_proba = np.asarray(y_proba)
        y_bin = label_binarize(y_true, classes=classes)

        # Em binário, label_binarize devolve uma única coluna.
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        roc_auc = roc_auc_score(
            y_bin, y_proba, average="macro", multi_class="ovr"
        )
        pr_auc = float(
            np.mean(
                [
                    average_precision_score(y_bin[:, i], y_proba[:, i])
                    for i in range(len(classes))
                ]
            )
        )
        res["ROC-AUC (Macro)"] = float(roc_auc)
        res["PR-AUC (Macro)"] = pr_auc

    return res


def matriz_confusao_ordenada(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    classes: list[Any] | None = None,
) -> pd.DataFrame:
    """Matriz de confusão como DataFrame com rótulos legíveis."""
    labels = classes if classes is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tabela = pd.DataFrame(cm, index=labels, columns=labels)
    tabela.index.name = "Real"
    tabela.columns.name = "Previsto"
    return tabela
