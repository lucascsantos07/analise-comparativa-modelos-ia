"""Base de conhecimento do sistema especialista — regressão.

O alvo (``Diabetes_Risk_Score``) é contínuo, portanto uma base finita de
regras não pode cobri-lo por enumeração. A solução adotada é um motor de
pontuação aditiva: cada regra representa uma faixa clinicamente
interpretável e contribui com pontos; o escore final é a soma à
pontuação base, truncada em [0, 100].

Os pesos foram estimados por mínimos quadrados
(OLS) sobre os indicadores das faixas no conjunto de TREINO e arredondados
para inteiros — o teste não participa da calibração.

``REGRAS_REGRESSAO_V1`` preserva a base anterior (quatro células definidas
pelas medianas do treino) para comparação no notebook 03.
"""

# ---------------------------------------------------------------------------
# Versão 2: motor de pontuação (19 regras)
# ---------------------------------------------------------------------------
REGRAS_REGRESSAO = [
    {
        "id": "RG_G_BAIXA",
        "descricao": "Glicemia de jejum normal (< 100 mg/dL)",
        "condicoes": {
            "Fasting_Blood_Sugar": {"operador": "<", "valor": 100},
        },
        "pontos": -10,
        "tipo": "heuristica",
    },
    {
        "id": "RG_G_PRE",
        "descricao": "Glicemia de jejum na faixa de pré-diabetes (100–125 mg/dL)",
        "condicoes": {
            "Fasting_Blood_Sugar": {
                "operador": "entre_aberto_direita",
                "valor": (100, 126),
            },
        },
        "pontos": 0,
        "tipo": "heuristica",
    },
    {
        "id": "RG_G_ALTA",
        "descricao": "Glicemia de jejum em faixa diagnóstica (>= 126 mg/dL)",
        "condicoes": {
            "Fasting_Blood_Sugar": {"operador": ">=", "valor": 126},
        },
        "pontos": 10,
        "tipo": "heuristica",
    },
    {
        "id": "RG_H_BAIXA",
        "descricao": "HbA1c normal (< 5,7%)",
        "condicoes": {
            "HbA1c": {"operador": "<", "valor": 5.7},
        },
        "pontos": -9,
        "tipo": "heuristica",
    },
    {
        "id": "RG_H_PRE",
        "descricao": "HbA1c na faixa de pré-diabetes (5,7–6,4%)",
        "condicoes": {
            "HbA1c": {
                "operador": "entre_aberto_direita",
                "valor": (5.7, 6.5),
            },
        },
        "pontos": 0,
        "tipo": "heuristica",
    },
    {
        "id": "RG_H_ALTA",
        "descricao": "HbA1c em faixa diagnóstica (>= 6,5%)",
        "condicoes": {
            "HbA1c": {"operador": ">=", "valor": 6.5},
        },
        "pontos": 9,
        "tipo": "heuristica",
    },
    # IMC — faixas clássicas de obesidade
    {
        "id": "RG_IMC_BAIXO",
        "descricao": "IMC abaixo de 25 kg/m²",
        "condicoes": {
            "BMI": {"operador": "<", "valor": 25},
        },
        "pontos": -5,
        "tipo": "heuristica",
    },
    {
        "id": "RG_IMC_SOBREPESO",
        "descricao": "IMC de sobrepeso (25–29,9 kg/m²)",
        "condicoes": {
            "BMI": {
                "operador": "entre_aberto_direita",
                "valor": (25, 30),
            },
        },
        "pontos": -2,
        "tipo": "heuristica",
    },
    {
        "id": "RG_IMC_OBESO1",
        "descricao": "Obesidade grau I (30–34,9 kg/m²)",
        "condicoes": {
            "BMI": {
                "operador": "entre_aberto_direita",
                "valor": (30, 35),
            },
        },
        "pontos": 2,
        "tipo": "heuristica",
    },
    {
        "id": "RG_IMC_OBESO2",
        "descricao": "Obesidade grau II ou superior (>= 35 kg/m²)",
        "condicoes": {
            "BMI": {"operador": ">=", "valor": 35},
        },
        "pontos": 5,
        "tipo": "heuristica",
    },
    # Idade — faixas heurísticas
    {
        "id": "RG_IDADE_JOVEM",
        "descricao": "Idade inferior a 40 anos",
        "condicoes": {
            "Age": {"operador": "<", "valor": 40},
        },
        "pontos": -6,
        "tipo": "heuristica",
    },
    {
        "id": "RG_IDADE_ADULTO",
        "descricao": "Idade entre 40 e 54 anos",
        "condicoes": {
            "Age": {
                "operador": "entre_aberto_direita",
                "valor": (40, 55),
            },
        },
        "pontos": -1,
        "tipo": "heuristica",
    },
    {
        "id": "RG_IDADE_MEIA",
        "descricao": "Idade entre 55 e 69 anos",
        "condicoes": {
            "Age": {
                "operador": "entre_aberto_direita",
                "valor": (55, 70),
            },
        },
        "pontos": 3,
        "tipo": "heuristica",
    },
    {
        "id": "RG_IDADE_IDOSO",
        "descricao": "Idade igual ou superior a 70 anos",
        "condicoes": {
            "Age": {"operador": ">=", "valor": 70},
        },
        "pontos": 4,
        "tipo": "heuristica",
    },
    # Histórico e comorbidades
    {
        "id": "RG_HIST_FAMILIAR",
        "descricao": "Histórico familiar de diabetes",
        "condicoes": {
            "Family_History_Diabetes": {"operador": "==", "valor": "Yes"},
        },
        "pontos": 10,
        "tipo": "heuristica",
    },
    {
        "id": "RG_HIPERTENSAO",
        "descricao": "Hipertensão arterial",
        "condicoes": {
            "Hypertension": {"operador": "==", "valor": "Yes"},
        },
        "pontos": 6,
        "tipo": "heuristica",
    },
    {
        "id": "RG_DIETA_RUIM",
        "descricao": "Qualidade da dieta ruim",
        "condicoes": {
            "Diet_Quality": {"operador": "==", "valor": "Poor"},
        },
        "pontos": 6,
        "tipo": "heuristica",
    },
    {
        "id": "RG_ATIVIDADE_BAIXA",
        "descricao": "Nível de atividade física baixo",
        "condicoes": {
            "Physical_Activity_Level": {"operador": "==", "valor": "Low"},
        },
        "pontos": 6,
        "tipo": "heuristica",
    },
    {
        "id": "RG_ATIVIDADE_IGNORADA",
        "descricao": "Nível de atividade física não informado",
        "condicoes": {
            "Physical_Activity_Level": {
                "operador": "==",
                "valor": "Não_informado",
            },
        },
        "pontos": 2,
        "tipo": "heuristica",
    },
]

PONTUACAO_BASE = 53
LIMITES_PONTUACAO = (0.0, 100.0)


# ---------------------------------------------------------------------------
# Versão 1 (linha de base): quatro células definidas pelas medianas
# ---------------------------------------------------------------------------
REGRAS_REGRESSAO_V1 = [
    {
        "id": "REG1",
        "descricao": (
            "Risco baixo a moderado — glicemia e HbA1c abaixo das medianas "
            "de treino"
        ),
        "condicoes": {
            "Fasting_Blood_Sugar": {"operador": "<", "valor": 142.9},
            "HbA1c": {"operador": "<", "valor": 8.5},
        },
        "conclusao": 64.08649388959124,
        "prioridade": 1,
        "tipo": "heuristica",
    },
    {
        "id": "REG2",
        "descricao": "Risco intermediário — glicemia baixa e HbA1c elevada",
        "condicoes": {
            "Fasting_Blood_Sugar": {"operador": "<", "valor": 142.9},
            "HbA1c": {"operador": ">=", "valor": 8.5},
        },
        "conclusao": 71.40114285714286,
        "prioridade": 2,
        "tipo": "heuristica",
    },
    {
        "id": "REG3",
        "descricao": "Risco intermediário — glicemia elevada e HbA1c controlada",
        "condicoes": {
            "Fasting_Blood_Sugar": {"operador": ">=", "valor": 142.9},
            "HbA1c": {"operador": "<", "valor": 8.5},
        },
        "conclusao": 76.67333472978635,
        "prioridade": 3,
        "tipo": "heuristica",
    },
    {
        "id": "REG4",
        "descricao": "Risco alto — glicemia e HbA1c acima das medianas de treino",
        "condicoes": {
            "Fasting_Blood_Sugar": {"operador": ">=", "valor": 142.9},
            "HbA1c": {"operador": ">=", "valor": 8.5},
        },
        "conclusao": 83.43078393881453,
        "prioridade": 4,
        "tipo": "heuristica",
    },
]
