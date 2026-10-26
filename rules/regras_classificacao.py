"""Base de conhecimento do sistema especialista — classificação.

Risco de diabetes em três classes (Low, Moderate, High). As regras usam
apenas limiares clínicos (ADA) e relações simples observadas no conjunto
de treino. Cada regra tem condições unidas por E (sem misturar OU) e um
tipo conforme a taxonomia da Aula 05.

A base tem quatro regras: as duas antigas regras de risco alto (com e sem
histórico familiar) foram fundidas, pois tinham a mesma conclusão e se
distinguiam apenas por uma condição que não alterava o resultado.
"""

REGRAS_CLASSIFICACAO = [
    {
        "id": "R1",
        "descricao": (
            "Risco alto — critérios diagnósticos de diabetes "
            "(glicemia de jejum e HbA1c elevadas)"
        ),
        "condicoes": {
            "Fasting_Blood_Sugar": {"operador": ">=", "valor": 126},
            "HbA1c": {"operador": ">=", "valor": 6.5},
        },
        "conclusao": "High",
        "prioridade": 1,
        "tipo": "heuristica",
    },
    {
        "id": "R2",
        "descricao": (
            "Risco moderado — faixa intermediária de glicemia e HbA1c "
            "em adultos"
        ),
        "condicoes": {
            "Fasting_Blood_Sugar": {"operador": "entre", "valor": (80, 125)},
            "HbA1c": {"operador": "entre", "valor": (5.3, 7.0)},
            "Age": {"operador": "<", "valor": 65},
        },
        "conclusao": "Moderate",
        "prioridade": 2,
        "tipo": "heuristica",
    },
    {
        "id": "R3",
        "descricao": (
            "Risco baixo — glicemia e HbA1c abaixo dos limites normais"
        ),
        "condicoes": {
            "Fasting_Blood_Sugar": {"operador": "<", "valor": 100},
            "HbA1c": {"operador": "<", "valor": 5.7},
        },
        "conclusao": "Low",
        "prioridade": 3,
        "tipo": "heuristica",
    },
    {
        "id": "R4",
        "descricao": (
            "Risco moderado — glicemia normal com HbA1c na faixa de "
            "pré-diabetes (marcadores discordantes)"
        ),
        "condicoes": {
            "Fasting_Blood_Sugar": {
                "operador": "entre_aberto_direita",
                "valor": (80, 100),
            },
            "HbA1c": {
                "operador": "entre_aberto_direita",
                "valor": (5.7, 6.5),
            },
        },
        "conclusao": "Moderate",
        "prioridade": 4,
        "tipo": "heuristica",
    },
]
