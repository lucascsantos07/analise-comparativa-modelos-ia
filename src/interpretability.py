"""Extração de regras legíveis a partir de uma árvore de decisão.

As árvores são treinadas sobre colunas transformadas (numéricas + dummies
one-hot). Este módulo percorre os caminhos até as folhas e traduz cada
split de volta para uma condição na variável original, permitindo
comparar as regras da árvore com as do sistema especialista.
"""

from typing import Any

import pandas as pd


def _condicao_estruturada(
    nome_transformado: str,
    limiar: float,
    vai_para_esquerda: bool,
    features_originais: list[str],
) -> dict[str, Any]:
    """Traduz um split da árvore em uma condição sobre a variável original."""
    for feature in features_originais:
        prefixo = f"{feature}_"
        if nome_transformado.startswith(prefixo):
            categoria = nome_transformado[len(prefixo):]
            # One-hot: <= 0.5 significa dummy ausente; > 0.5, presente.
            return {
                "feature": feature,
                "op": "!=" if vai_para_esquerda else "==",
                "valor": categoria,
            }

    return {
        "feature": nome_transformado,
        "op": "<=" if vai_para_esquerda else ">",
        "valor": round(float(limiar), 2),
    }


def _simplificar_condicoes(condicoes: list[dict]) -> list[dict]:
    """Remove condições redundantes do mesmo caminho.

    - Para variáveis numéricas, mantém apenas o limite inferior mais alto e o
      superior mais baixo (os demais são implicados por eles).
    - Para categóricas, descarta ``!=`` quando o caminho já afirma ``==`` de
      outra categoria da mesma variável (one-hot).
    """
    if not condicoes:
        return []

    ordem = {}
    for posicao, condicao in enumerate(condicoes):
        ordem.setdefault(condicao["feature"], posicao)

    numericas: dict[str, list[dict]] = {}
    categoricas: list[dict] = []

    for condicao in condicoes:
        if condicao["op"] in ("<=", ">"):
            numericas.setdefault(condicao["feature"], []).append(condicao)
        else:
            categoricas.append(condicao)

    simplificadas: list[dict] = []

    for lista in numericas.values():
        inferiores = [c for c in lista if c["op"] == ">"]
        superiores = [c for c in lista if c["op"] == "<="]
        if inferiores:
            simplificadas.append(max(inferiores, key=lambda c: c["valor"]))
        if superiores:
            simplificadas.append(min(superiores, key=lambda c: c["valor"]))

    features_com_igualdade = {c["feature"] for c in categoricas if c["op"] == "=="}
    for condicao in categoricas:
        if condicao["op"] == "!=" and condicao["feature"] in features_com_igualdade:
            continue
        simplificadas.append(condicao)

    return sorted(simplificadas, key=lambda c: ordem[c["feature"]])


def _formatar_condicao(condicao: dict) -> str:
    valor = condicao["valor"]
    if isinstance(valor, float):
        valor = f"{valor:g}"
    return f"{condicao['feature']} {condicao['op']} {valor}"


def _caminhos_folhas(arvore: Any, no: int = 0, condicoes: tuple = ()):
    """Gera (folha, condições) para todos os caminhos da árvore."""
    esquerda = arvore.children_left[no]
    direita = arvore.children_right[no]

    if esquerda == direita == -1:  # folha
        yield no, condicoes
        return

    atributo = arvore.feature[no]
    limiar = arvore.threshold[no]

    yield from _caminhos_folhas(
        arvore, esquerda, condicoes + ((atributo, limiar, True),)
    )
    yield from _caminhos_folhas(
        arvore, direita, condicoes + ((atributo, limiar, False),)
    )


def extrair_regras_arvore(
    pipeline: Any,
    n_regras: int = 3,
    diversificar: bool = True,
) -> pd.DataFrame:
    """Extrai as maiores folhas da árvore como regras SE-ENTÃO.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline já ajustado (preprocessor + model).
    n_regras : int
        Quantidade de folhas a retornar.
    diversificar : bool
        Se True, garante ao menos uma folha por classe prevista antes de
        completar pelas folhas de maior cobertura.
    """
    preprocessador = pipeline.named_steps["preprocessor"]
    modelo = pipeline.named_steps["model"]
    arvore = modelo.tree_

    nomes = list(preprocessador.get_feature_names_out())
    features_originais = list(preprocessador.feature_names_in_)
    classes = list(modelo.classes_)

    folhas = []
    for folha, condicoes in _caminhos_folhas(arvore):
        contagens = arvore.value[folha][0]
        total = float(contagens.sum())
        indice_classe = int(contagens.argmax())
        # Em versões recentes do sklearn, tree_.value guarda proporções;
        # n_node_samples mantém a contagem real de amostras da folha.
        amostras = int(arvore.n_node_samples[folha])

        estruturadas = [
            _condicao_estruturada(
                nomes[atributo], limiar, esquerda, features_originais
            )
            for atributo, limiar, esquerda in condicoes
        ]
        estruturadas = _simplificar_condicoes(estruturadas)

        folhas.append(
            {
                "Folha": folha,
                "Classe": classes[indice_classe],
                "Probabilidade": float(contagens[indice_classe] / total),
                "Amostras": amostras,
                "Condições": [_formatar_condicao(c) for c in estruturadas],
            }
        )

    tabela = pd.DataFrame(folhas).sort_values("Amostras", ascending=False)
    tabela = tabela.set_index("Folha", drop=False)

    if diversificar:
        escolhidas = []
        for classe in classes:
            candidatas = tabela[tabela["Classe"] == classe]
            if not candidatas.empty and len(escolhidas) < n_regras:
                escolhidas.append(candidatas.index[0])

        for indice in tabela.index:
            if len(escolhidas) >= n_regras:
                break
            if indice not in escolhidas:
                escolhidas.append(indice)

        tabela = tabela.loc[escolhidas]

    tabela = tabela.head(n_regras).reset_index(drop=True)
    tabela.insert(0, "Regra", [f"ARV{i + 1}" for i in range(len(tabela))])
    tabela["Condições"] = tabela["Condições"].apply(lambda itens: " E ".join(itens))
    return tabela
