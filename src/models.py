"""Construção, ajuste e persistência dos modelos de árvore e ensemble.

Todos os pipelines aplicam o pré-processamento dentro do objeto, de modo
que o one-hot é ajustado apenas nos dados de treino de cada fold.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import (
    CATEGORICAS_CLASSIFICACAO,
    CATEGORICAS_REGRESSAO,
    CLASSES,
    MODELS_DIR,
    N_FOLDS,
    RANDOM_STATE,
)
from .preprocessing import criar_preprocessador_arvore


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def pipeline_classificacao(estimador) -> Pipeline:
    """Pré-processador (one-hot) + classificador."""
    return Pipeline(
        steps=[
            ("preprocessor", criar_preprocessador_arvore(CATEGORICAS_CLASSIFICACAO)),
            ("model", estimador),
        ]
    )


def pipeline_regressao(estimador) -> Pipeline:
    """Pré-processador (one-hot) + regressor."""
    return Pipeline(
        steps=[
            ("preprocessor", criar_preprocessador_arvore(CATEGORICAS_REGRESSAO)),
            ("model", estimador),
        ]
    )


def criar_arvore_classificacao(criterion: str = "gini", **kwargs: Any) -> Pipeline:
    return pipeline_classificacao(
        DecisionTreeClassifier(
            criterion=criterion, random_state=RANDOM_STATE, **kwargs
        )
    )


def criar_arvore_regressao(criterion: str = "squared_error", **kwargs: Any) -> Pipeline:
    return pipeline_regressao(
        DecisionTreeRegressor(
            criterion=criterion, random_state=RANDOM_STATE, **kwargs
        )
    )


def criar_random_forest_classificacao(**kwargs: Any) -> Pipeline:
    parametros = dict(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    parametros.update(kwargs)
    return pipeline_classificacao(RandomForestClassifier(**parametros))


def criar_random_forest_regressao(**kwargs: Any) -> Pipeline:
    parametros = dict(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    parametros.update(kwargs)
    return pipeline_regressao(RandomForestRegressor(**parametros))


def criar_gradient_boosting_classificacao(**kwargs: Any) -> Pipeline:
    parametros = dict(n_estimators=100, learning_rate=0.1, max_depth=3,
                      random_state=RANDOM_STATE)
    parametros.update(kwargs)
    return pipeline_classificacao(GradientBoostingClassifier(**parametros))


def criar_gradient_boosting_regressao(**kwargs: Any) -> Pipeline:
    parametros = dict(n_estimators=100, learning_rate=0.1, max_depth=3,
                      random_state=RANDOM_STATE)
    parametros.update(kwargs)
    return pipeline_regressao(GradientBoostingRegressor(**parametros))


# ---------------------------------------------------------------------------
# Grades de hiperparâmetros
# ---------------------------------------------------------------------------
GRADE_ARVORE_CLASSIFICACAO = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [4, 6, 8, 10, None],
    "model__min_samples_leaf": [1, 10, 50],
    "model__ccp_alpha": [0.0, 0.0005],
}

GRADE_ARVORE_REGRESSAO = {
    "model__criterion": ["squared_error", "absolute_error"],
    "model__max_depth": [4, 6, 8, 10, None],
    "model__min_samples_leaf": [5, 20, 50],
    "model__ccp_alpha": [0.0, 0.0005],
}

GRADE_RF_CLASSIFICACAO = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [8, None],
    "model__min_samples_leaf": [1, 5],
    "model__max_features": ["sqrt", 0.5],
}

GRADE_RF_REGRESSAO = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [8, None],
    "model__min_samples_leaf": [1, 5],
    "model__max_features": ["sqrt", 0.5],
}

GRADE_GB_CLASSIFICACAO = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3],
}

GRADE_GB_REGRESSAO = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3],
}


# ---------------------------------------------------------------------------
# Busca de hiperparâmetros
# ---------------------------------------------------------------------------
def validadores_cv(problema: str) -> Any:
    """StratifiedKFold na classificação e KFold na regressão."""
    if problema == "classificacao":
        return StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def buscar_hiperparametros(
    pipeline: Pipeline,
    grade: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    problema: str,
    scoring: str,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Executa GridSearchCV com o CV adequado ao problema."""
    busca = GridSearchCV(
        estimator=pipeline,
        param_grid=grade,
        scoring=scoring,
        cv=validadores_cv(problema),
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
    )
    busca.fit(X, y)
    return busca


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def nomes_features(pipeline: Pipeline) -> list[str]:
    """Nomes das colunas após o one-hot (usados em importância e árvore)."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def probabilidades_ordenadas(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probabilidades com as colunas na ordem de ``CLASSES``.

    ``predict_proba`` devolve as colunas na ordem de ``classes_`` (alfabética
    no sklearn). As métricas e os gráficos usam a ordem clínica Low,
    Moderate, High; esta função garante o alinhamento.
    """
    probabilidades = pipeline.predict_proba(X)
    ordem_modelo = list(pipeline.classes_)
    indices = [ordem_modelo.index(classe) for classe in CLASSES]
    return probabilidades[:, indices]


def importancias_features(pipeline: Pipeline) -> pd.Series:
    """Importância das features, somando as dummies de cada variável.

    As colunas one-hot recebem o nome ``<feature>_<categoria>``. Para não
    confundir, por exemplo, ``Fasting_Blood_Sugar`` com ``Family_History``,
    o agrupamento usa a lista de features de entrada do pipeline.
    """
    nomes = nomes_features(pipeline)
    valores = pipeline.named_steps["model"].feature_importances_
    features_originais = list(
        pipeline.named_steps["preprocessor"].feature_names_in_
    )

    def origem(nome: str) -> str:
        for feature in features_originais:
            if nome == feature or nome.startswith(f"{feature}_"):
                return feature
        return nome

    serie = pd.Series(valores, index=nomes, dtype=float)
    return serie.groupby(origem).sum().sort_values(ascending=False)


def salvar_modelo(modelo: Any, nome: str) -> Path:
    """Salva o modelo em artifacts/models/<nome>.joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = MODELS_DIR / f"{nome}.joblib"
    joblib.dump(modelo, caminho)
    return caminho


def carregar_modelo(nome: str) -> Any:
    """Carrega um modelo salvo em artifacts/models/."""
    caminho = MODELS_DIR / f"{nome}.joblib"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho}. Execute o notebook que o treina."
        )
    return joblib.load(caminho)
