
"""Ferramentas para medir a eficiência computacional dos modelos."""

import pickle
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np


def medir_inferencia(
    funcao_predicao: Callable,
    X: Any,
    n_rep: int = 30,
    aquecer: bool = True,
) -> dict[str, float]:
    """Mede tempo e pico de memória apenas da etapa de inferência.

    Uma chamada de aquecimento (fora da medição) evita contabilizar o
    custo de inicialização de bibliotecas na primeira execução.
    """
    if tracemalloc.is_tracing():
        raise RuntimeError("Já existe uma sessão tracemalloc ativa.")

    if aquecer:
        funcao_predicao(X)

    tempos = []
    tracemalloc.start()
    try:
        for _ in range(n_rep):
            inicio = time.perf_counter()
            funcao_predicao(X)
            tempos.append(time.perf_counter() - inicio)
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    media = float(np.mean(tempos))
    desvio = float(np.std(tempos, ddof=1)) if n_rep > 1 else 0.0
    n_amostras = len(X)

    return {
        "n_repeticoes": n_rep,
        "tempo_medio_s": media,
        "tempo_desvio_s": desvio,
        "tempo_por_amostra_ms": 1000.0 * media / n_amostras,
        "vazao_amostras_s": n_amostras / media,
        "pico_memoria_mib": pico / (1024 ** 2),
    }


def medir_treino(
    funcao_treino: Callable,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Mede tempo wall-clock e pico de memória de um treinamento.

    Retorna também o objeto ajustado em ``resultado``.
    """
    if tracemalloc.is_tracing():
        raise RuntimeError("Já existe uma sessão tracemalloc ativa.")

    tracemalloc.start()
    try:
        inicio = time.perf_counter()
        resultado = funcao_treino(*args, **kwargs)
        tempo = time.perf_counter() - inicio
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return {
        "tempo_treino_s": float(tempo),
        "pico_memoria_treino_mib": pico / (1024 ** 2),
        "resultado": resultado,
    }


def medir_tamanho_arquivo(caminho: str | Path) -> dict[str, float]:
    """Tamanho do modelo salvo em disco."""
    tamanho = Path(caminho).stat().st_size
    return {"tamanho_bytes": tamanho, "tamanho_kib": tamanho / 1024}



def medir_tempo_execucao(
    funcao: Callable,
    *args: Any,
    n_execucoes: int = 30,
    **kwargs: Any,
) -> dict[str, float | int]:
    """Mede o tempo de execução de uma função em múltiplas execuções.

    Retorna média, desvio padrão, mediana, mínimo e máximo.
    """
    if n_execucoes < 2:
        raise ValueError("Use pelo menos duas execuções para calcular o desvio padrão.")

    tempos = []

    for _ in range(n_execucoes):
        inicio = time.perf_counter()
        funcao(*args, **kwargs)
        fim = time.perf_counter()

        tempos.append(fim - inicio)

    return {
        "n_execucoes": n_execucoes,
        "tempo_medio_s": float(np.mean(tempos)),
        "tempo_desvio_padrao_s": float(np.std(tempos, ddof=1)),
        "tempo_mediano_s": float(np.median(tempos)),
        "tempo_minimo_s": float(np.min(tempos)),
        "tempo_maximo_s": float(np.max(tempos)),
    }



def medir_pico_memoria(
    funcao: Callable,
    *args: Any,
    **kwargs: Any,
) -> dict[str, float]:
    """Mede o pico de memória rastreado durante a execução de uma função.

    A medição considera as alocações Python rastreadas pelo tracemalloc.
    Ela não representa necessariamente toda a RAM utilizada pelo processo.

    Args:
        funcao: Função que será avaliada.
        *args: Argumentos posicionais da função.
        **kwargs: Argumentos nomeados da função.

    Returns:
        Dicionário com memória atual e pico em MiB.
    """
    if tracemalloc.is_tracing():
        raise RuntimeError(
            "Já existe uma sessão tracemalloc ativa. "
            "Finalize-a antes de iniciar esta medição."
        )

    tracemalloc.start()

    try:
        funcao(*args, **kwargs)
        memoria_atual, pico_memoria = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return {
        "memoria_atual_mib": memoria_atual / (1024 ** 2),
        "pico_memoria_mib": pico_memoria / (1024 ** 2),
    }


def medir_tamanho_serializado(objeto: Any) -> dict[str, int | float]:
    """Estima o tamanho serializado de um objeto usando pickle.

    O resultado representa o tamanho da representação serializada,
    não a memória total ocupada pelo objeto em execução.

    Args:
        objeto: Modelo, regras ou outro objeto serializável.

    Returns:
        Tamanho serializado em bytes e KiB.
    """
    dados = pickle.dumps(objeto, protocol=pickle.HIGHEST_PROTOCOL)
    tamanho_bytes = len(dados)

    return {
        "tamanho_bytes": tamanho_bytes,
        "tamanho_kib": tamanho_bytes / 1024,
    }


def estimar_energia_co2(
    tempo_segundos: float,
    potencia_watts: float = 65.0,
    fator_emissao_g_kwh: float = 45.8,
) -> dict[str, float]:
    """
    Estima a energia consumida e a emissão de CO2 equivalente.

    Fórmula:
        Energia (kWh) = (Potência em Watts * Tempo em segundos) / (3600 * 1000)
        Emissão de CO2 (gCO2) = Energia (kWh) * Fator de emissão (gCO2/kWh)

    Parâmetros:
        tempo_segundos: Tempo de execução (treino ou inferência).
        potencia_watts: TDP / potência média estimada do processador (padrão: 65W).
        fator_emissao_g_kwh: Fator de emissão da matriz elétrica do Brasil (45.8 gCO2/kWh).

    Retorna:
        Dicionário com energia_kwh e emissao_co2_g.
    """
    energia_kwh = (potencia_watts * tempo_segundos) / (3600.0 * 1000.0)
    emissao_co2_g = energia_kwh * fator_emissao_g_kwh

    return {
        "potencia_watts": potencia_watts,
        "energia_kwh": energia_kwh,
        "emissao_co2_g": emissao_co2_g,
        "fator_emissao_utilizado": fator_emissao_g_kwh,
    }

