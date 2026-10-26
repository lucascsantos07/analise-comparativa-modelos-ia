"""Divisão treino/teste e pré-processamento.

Todo ajuste de transformação acontece dentro de um ``Pipeline`` do
scikit-learn, portanto apenas nos dados de treino (sem data leakage).
"""

from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import RANDOM_STATE, TEST_SIZE


def separar_treino_teste(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide X e y em treino/teste, com estratificação opcional."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def criar_preprocessador_arvore(
    categoricas: list[str],
) -> ColumnTransformer:
    """Pré-processador para árvores e ensembles.

    Codifica as categóricas em one-hot e mantém as numéricas como estão
    (árvores não exigem padronização).
    """
    return ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categoricas,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def criar_preprocessor(
    numeric_features: Optional[list[str]] = None,
    nominal_features: Optional[list[str]] = None,
    ordinal_features: Optional[list[str]] = None,
    ordinal_categories: Optional[list[list[str]]] = None,
) -> ColumnTransformer:
    """Pré-processador genérico, com padronização das variáveis numéricas."""
    transformers = []

    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))

    if nominal_features:
        transformers.append(
            (
                "nom",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                nominal_features,
            )
        )

    if ordinal_features and ordinal_categories:
        transformers.append(
            (
                "ord",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def criar_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Encapsula pré-processador + modelo em um único Pipeline."""
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("model", estimator)]
    )
