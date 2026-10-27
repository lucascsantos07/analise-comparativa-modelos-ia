"""Leitura do cronômetro de elaboração das regras e exportação da base.

A base de conhecimento é mantida em Python (``rules/*.py``) e exportada
para texto (``rules/*.txt``) para servir como documentação legível do
sistema especialista.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOGS_DIR


def formatar_condicao(variavel: str, condicao: dict) -> str:
    """Converte uma condição em texto legível."""
    operador = condicao["operador"]
    valor = condicao["valor"]

    if operador == "entre":
        texto = f"entre {valor[0]} e {valor[1]}"
    elif operador == "entre_aberto_direita":
        texto = f"maior ou igual a {valor[0]} e menor que {valor[1]}"
    else:
        simbolos = {">=": ">=", "<=": "<=", ">": ">", "<": "<", "==": "="}
        texto = f"{simbolos.get(operador, operador)} {valor}"

    return f"{variavel} {texto}"


def regra_para_texto(regra: dict) -> str:
    """Formata uma regra SE-ENTÃO com o tipo e a conclusão."""
    condicoes = "\n    E ".join(
        formatar_condicao(variavel, condicao)
        for variavel, condicao in regra["condicoes"].items()
    )

    if "pontos" in regra:
        consequente = f"PONTOS = {regra['pontos']:+d}"
    else:
        consequente = f"Diabetes_Risk = {regra['conclusao']}"

    prioridade = regra.get("prioridade")
    linha_prioridade = f"\n  Prioridade: {prioridade}" if prioridade else ""

    return (
        f"[{regra['id']}] {regra['descricao']}\n"
        f"  Tipo: {regra['tipo']}{linha_prioridade}\n"
        f"  SE {condicoes}\n"
        f"  ENTÃO {consequente}"
    )


def exportar_regras_txt(
    regras: list[dict],
    caminho: str | Path,
    titulo: str,
    pontuacao_base: float | None = None,
    limites: tuple[float, float] | None = None,
) -> Path:
    """Grava a base de regras em texto, uma regra por bloco."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    partes = [titulo, "=" * len(titulo), ""]
    if pontuacao_base is not None:
        partes.append(f"Pontuação base: {pontuacao_base}")
    if limites is not None:
        partes.append(f"Limites do escore: {limites[0]} a {limites[1]}")
    if pontuacao_base is not None or limites is not None:
        partes.append("")

    partes.extend(regra_para_texto(regra) + "\n" for regra in regras)
    caminho.write_text("\n".join(partes), encoding="utf-8")
    return caminho


# ---------------------------------------------------------------------------
# Cronômetro (esforço humano)
# ---------------------------------------------------------------------------
COLUNAS_CRONOMETRO = [
    "data",
    "horario_inicio",
    "horario_fim",
    "interrupcoes",
    "sessao",
    "problema",
    "tempo_total",
    "numero_regras_criadas",
]


def ler_cronometro(caminho: str | Path | None = None) -> pd.DataFrame:
    """Lê o CSV do cronômetro e converte tempos para segundos.

    O arquivo é separado por ponto e vírgula e usa datas dd/mm/aaaa.
    """
    caminho = Path(caminho) if caminho is not None else LOGS_DIR / "cronometro_regras.csv"
    df = pd.read_csv(caminho, sep=";", dtype=str)

    faltantes = [coluna for coluna in COLUNAS_CRONOMETRO if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes no cronômetro: {faltantes}")

    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["sessao"] = df["sessao"].astype(int)
    df["interrupcoes"] = df["interrupcoes"].astype(int)
    df["numero_regras_criadas"] = df["numero_regras_criadas"].astype(int)
    df["tempo_total_s"] = (
        pd.to_timedelta(df["tempo_total"]).dt.total_seconds().astype(float)
    )
    df["tempo_total_min"] = df["tempo_total_s"] / 60

    return df.sort_values(["problema", "sessao"]).reset_index(drop=True)


def resumo_cronometro(df: pd.DataFrame) -> pd.DataFrame:
    """Tempo total, médio, desvio padrão e tempo médio por regra."""
    linhas = []

    for problema, grupo in df.groupby("problema", sort=True):
        tempos = grupo["tempo_total_min"].to_numpy()
        total_regras = int(grupo["numero_regras_criadas"].sum())

        linhas.append(
            {
                "Problema": problema.capitalize(),
                "Sessões": len(grupo),
                "Regras criadas": total_regras,
                "Tempo total (min)": float(tempos.sum()),
                "Tempo médio por sessão (min)": float(np.mean(tempos)),
                "Desvio padrão (min)": float(np.std(tempos, ddof=1))
                if len(tempos) > 1
                else np.nan,
                "Tempo médio por regra (min)": float(tempos.sum() / total_regras)
                if total_regras
                else np.nan,
            }
        )

    return pd.DataFrame(linhas)

