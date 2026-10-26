"""Motor de inferência do sistema especialista.

A base de conhecimento (regras SE-ENTÃO) fica separada do motor: as
regras estão em ``rules/`` e este módulo apenas as avalia. Dois modos de
inferência são suportados:

* ``prioridade``: cada amostra recebe a conclusão de uma única regra
  (usado na classificação);
* ``pontuacao``: regras somam pontos a um escore base (usado na
  regressão, pois o alvo é contínuo).
"""

from math import isnan
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Avaliação de condições (registro único)
# ---------------------------------------------------------------------------
def valor_ausente(valor: Any) -> bool:
    """Indica se o valor está ausente (None ou NaN)."""
    if valor is None:
        return True
    if isinstance(valor, Real):
        return isnan(valor)
    return False


def avaliar_condicao(
    fatos: dict[str, Any],
    variavel: str,
    condicao: dict[str, Any],
) -> bool:
    """Avalia uma condição individual; ausência de valor resulta em False."""
    if variavel not in fatos or valor_ausente(fatos[variavel]):
        return False

    valor_fato = fatos[variavel]
    operador = condicao["operador"]
    valor_regra = condicao["valor"]

    if operador == ">=":
        return valor_fato >= valor_regra
    if operador == "<=":
        return valor_fato <= valor_regra
    if operador == ">":
        return valor_fato > valor_regra
    if operador == "<":
        return valor_fato < valor_regra
    if operador == "==":
        return valor_fato == valor_regra
    if operador == "entre":
        return valor_regra[0] <= valor_fato <= valor_regra[1]
    if operador == "entre_aberto_direita":
        return valor_regra[0] <= valor_fato < valor_regra[1]

    raise ValueError(f"Operador não suportado: {operador}")


def avaliar_regra(fatos: dict[str, Any], regra: dict[str, Any]) -> bool:
    """Uma regra é ativada quando todas as suas condições (E) são verdadeiras."""
    return all(
        avaliar_condicao(fatos, variavel, condicao)
        for variavel, condicao in regra["condicoes"].items()
    )


# ---------------------------------------------------------------------------
# Avaliação vetorizada (lote)
# ---------------------------------------------------------------------------
def _mascara_condicao(serie: pd.Series, condicao: dict[str, Any]) -> np.ndarray:
    """Versão vetorizada de ``avaliar_condicao`` para uma coluna do DataFrame."""
    valores = serie.to_numpy()
    operador = condicao["operador"]
    alvo = condicao["valor"]

    if operador == ">=":
        return valores >= alvo
    if operador == "<=":
        return valores <= alvo
    if operador == ">":
        return valores > alvo
    if operador == "<":
        return valores < alvo
    if operador == "==":
        return valores == alvo
    if operador == "entre":
        return (valores >= alvo[0]) & (valores <= alvo[1])
    if operador == "entre_aberto_direita":
        return (valores >= alvo[0]) & (valores < alvo[1])

    raise ValueError(f"Operador não suportado: {operador}")


def _matriz_ativacao(
    df: pd.DataFrame,
    regras: list[dict[str, Any]],
) -> np.ndarray:
    """Matriz booleana (n_regras x n_amostras) de ativação das regras."""
    matriz = np.zeros((len(regras), len(df)), dtype=bool)

    for i, regra in enumerate(regras):
        ativa = np.ones(len(df), dtype=bool)
        for variavel, condicao in regra["condicoes"].items():
            if variavel not in df.columns:
                ativa[:] = False
                break
            ativa &= _mascara_condicao(df[variavel], condicao)
        matriz[i] = ativa

    return matriz


def _lista_regras_ativadas(
    matriz: np.ndarray,
    regras: list[dict[str, Any]],
) -> list[list[str]]:
    """Converte a matriz de ativação em listas de ids por amostra."""
    ids = np.array([regra["id"] for regra in regras], dtype=object)
    return [ids[matriz[:, j]].tolist() for j in range(matriz.shape[1])]


def _sem_target(df: pd.DataFrame, coluna_target: str | None) -> pd.DataFrame:
    if coluna_target and coluna_target in df.columns:
        return df.drop(columns=[coluna_target])
    return df


# ---------------------------------------------------------------------------
# Modo 1: conclusão única com resolução de conflito por prioridade
# ---------------------------------------------------------------------------
def inferir(
    fatos: dict[str, Any],
    regras: list[dict[str, Any]],
) -> dict[str, Any]:
    """Inferência de um único registro, com resolução de conflito.

    Quando regras com conclusões diferentes são ativadas, vence a de
    menor valor de ``prioridade``.
    """
    ativadas = [regra for regra in regras if avaliar_regra(fatos, regra)]
    ordenadas = sorted(ativadas, key=lambda regra: regra["prioridade"])

    conclusoes = list(
        dict.fromkeys(regra["conclusao"] for regra in ordenadas)
    )

    if ordenadas:
        selecionada = ordenadas[0]
        conclusao = selecionada["conclusao"]
        id_selecionada = selecionada["id"]
    else:
        conclusao = "Indeterminado"
        id_selecionada = None

    return {
        "conclusao": conclusao,
        "regras_ativadas": [regra["id"] for regra in ordenadas],
        "conclusoes_encontradas": conclusoes,
        "houve_conflito": len(conclusoes) > 1,
        "regra_selecionada": id_selecionada,
    }


def aplicar_sistema_especialista(
    df: pd.DataFrame,
    regras: list[dict[str, Any]],
    coluna_target: str | None = None,
) -> pd.DataFrame:
    """Aplica a base de regras a todas as linhas (avaliação vetorizada).

    Mantém o índice original e devolve: conclusão, regra selecionada,
    regras ativadas e indicador de conflito.
    """
    dados = _sem_target(df, coluna_target)
    matriz = _matriz_ativacao(dados, regras)
    n_amostras = len(dados)

    prioridades = np.array([regra["prioridade"] for regra in regras])
    prioridades_ativas = np.where(matriz, prioridades[:, None], np.inf)
    indice = prioridades_ativas.argmin(axis=0)
    encontrou = matriz.any(axis=0)

    conclusoes = np.array([regra["conclusao"] for regra in regras], dtype=object)
    conclusao = np.where(encontrou, conclusoes[indice], "Indeterminado")

    # Conflito: alguma regra ativada aponta para conclusão diferente da vencedora.
    conflito = np.zeros(n_amostras, dtype=bool)
    for i, valor in enumerate(conclusoes):
        conflito |= matriz[i] & (valor != conclusao)

    ids = np.array([regra["id"] for regra in regras], dtype=object)

    regra_selecionada = pd.Series(
        np.where(encontrou, ids[indice], None), index=df.index, dtype=object
    )

    return pd.DataFrame(
        {
            "conclusao": conclusao,
            "regra_selecionada": regra_selecionada,
            "regras_ativadas": _lista_regras_ativadas(matriz, regras),
            "houve_conflito": conflito,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Modo 2: escore por soma de pontos (regressão)
# ---------------------------------------------------------------------------
def inferir_pontuacao(
    fatos: dict[str, Any],
    regras: list[dict[str, Any]],
    pontuacao_base: float,
    limites: tuple[float, float] = (0.0, 100.0),
) -> dict[str, Any]:
    """Inferência de um registro somando os pontos das regras ativadas."""
    ativadas = [regra for regra in regras if avaliar_regra(fatos, regra)]
    bruta = pontuacao_base + sum(regra["pontos"] for regra in ativadas)
    final = float(np.clip(bruta, *limites))

    return {
        "conclusao": final,
        "pontuacao_bruta": float(bruta),
        "regras_ativadas": [regra["id"] for regra in ativadas],
        "n_regras_ativadas": len(ativadas),
    }


def aplicar_sistema_especialista_pontuacao(
    df: pd.DataFrame,
    regras: list[dict[str, Any]],
    pontuacao_base: float,
    limites: tuple[float, float] = (0.0, 100.0),
    coluna_target: str | None = None,
) -> pd.DataFrame:
    """Aplica a base de pontuação a todas as linhas (avaliação vetorizada)."""
    dados = _sem_target(df, coluna_target)
    matriz = _matriz_ativacao(dados, regras)

    pontos = np.array([regra["pontos"] for regra in regras], dtype=float)
    bruta = pontuacao_base + matriz.T @ pontos
    final = np.clip(bruta, *limites)

    return pd.DataFrame(
        {
            "conclusao": final,
            "pontuacao_bruta": bruta,
            "regras_ativadas": _lista_regras_ativadas(matriz, regras),
            "n_regras_ativadas": matriz.sum(axis=0),
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Diagnóstico da base
# ---------------------------------------------------------------------------
def resumo_regras(regras: list[dict[str, Any]]) -> pd.DataFrame:
    """Tabela descritiva das regras (id, tipo, condições e conclusão)."""
    tem_pontos = all("pontos" in regra for regra in regras)

    linhas = []
    for regra in regras:
        linha = {
            "Regra": regra["id"],
            "Tipo": regra["tipo"],
            "Condições": " E ".join(
                f"{variavel} {condicao['operador']} {condicao['valor']}"
                for variavel, condicao in regra["condicoes"].items()
            ),
            "Descrição": regra["descricao"],
        }

        if tem_pontos:
            linha["Pontos"] = regra["pontos"]
        else:
            linha["Conclusão"] = regra["conclusao"]
            if "prioridade" in regra:
                linha["Prioridade"] = regra["prioridade"]

        linhas.append(linha)

    return pd.DataFrame(linhas)
