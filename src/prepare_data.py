"""Carregamento e divisão dos datasets processados."""

from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR,
    FEATURES_CLASSIFICACAO,
    FEATURES_REGRESSAO,
    TARGET_CLASSIFICACAO,
    TARGET_REGRESSAO,
)
from .preprocessing import separar_treino_teste

ARQUIVO_CLASSIFICACAO = DATA_DIR / "dataset_classification.csv"
ARQUIVO_REGRESSAO = DATA_DIR / "dataset_regression.csv"


def carregar_classificacao() -> pd.DataFrame:
    """Lê o dataset de classificação com as features e o target definidos."""
    df = pd.read_csv(ARQUIVO_CLASSIFICACAO)
    return df[FEATURES_CLASSIFICACAO + [TARGET_CLASSIFICACAO]]


def carregar_regressao() -> pd.DataFrame:
    """Lê o dataset de regressão com as features e o target definidos."""
    df = pd.read_csv(ARQUIVO_REGRESSAO)
    return df[FEATURES_REGRESSAO + [TARGET_REGRESSAO]]


def preparar_classificacao():
    """Retorna X_train, X_test, y_train, y_test (divisão estratificada)."""
    df = carregar_classificacao()
    X = df[FEATURES_CLASSIFICACAO]
    y = df[TARGET_CLASSIFICACAO]

    return separar_treino_teste(X, y, stratify=y)


def preparar_regressao():
    """Retorna X_train, X_test, y_train, y_test (divisão aleatória)."""
    df = carregar_regressao()
    X = df[FEATURES_REGRESSAO]
    y = df[TARGET_REGRESSAO]

    return separar_treino_teste(X, y)


if __name__ == "__main__":
    Xtr_c, Xte_c, ytr_c, yte_c = preparar_classificacao()
    Xtr_r, Xte_r, ytr_r, yte_r = preparar_regressao()

    print("=== CLASSIFICAÇÃO ===")
    print(f"Treino: {Xtr_c.shape} | Teste: {Xte_c.shape}")
    print(ytr_c.value_counts(normalize=True).round(4).to_string())

    print("\n=== REGRESSÃO ===")
    print(f"Treino: {Xtr_r.shape} | Teste: {Xte_r.shape}")
    print(f"y treino: {ytr_r.shape} | y teste: {yte_r.shape}")
