"""
Funções auxiliares da aplicação: carregamento do dataset, construção de
tabelas de frequência, detecção de outliers (regra do IQR) e geração de
interpretações textuais automáticas.

Importante: pandas/numpy são usados aqui apenas para *carregar e manipular*
os dados (ler CSV, filtrar colunas, ordenar). Toda medida estatística
exibida ao usuário nas páginas do app vem das funções de `minhastats`.
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple

import pandas as pd
import streamlit as st

import minhastats as ms

CAMINHO_DADOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vgsales.csv")


@st.cache_data
def carregar_dados() -> pd.DataFrame:
    """Carrega o dataset Video Game Sales (vgsales.csv) do disco."""
    df = pd.read_csv(CAMINHO_DADOS)
    return df


def colunas_numericas(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def colunas_categoricas(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def numero_de_classes_sturges(n: int) -> int:
    """Regra de Sturges: k = 1 + log2(n), arredondado para cima."""
    if n <= 0:
        return 1
    return max(1, math.ceil(1 + math.log2(n)))


def tabela_frequencias_numerica(dados: List[float], n_classes: int | None = None) -> pd.DataFrame:
    """
    Monta uma tabela de frequências em classes de igual amplitude para uma
    variável numérica contínua, usando amplitude/quartis calculados pela
    biblioteca própria `minhastats` para definir min/max e o número de
    classes pela regra de Sturges.
    """
    valores = sorted(float(v) for v in dados)
    n = len(valores)
    if n_classes is None:
        n_classes = numero_de_classes_sturges(n)

    minimo = valores[0]
    maximo = valores[-1]
    amplitude_total = ms.amplitude(valores)
    largura = amplitude_total / n_classes if amplitude_total > 0 else 1.0

    limites = [minimo + i * largura for i in range(n_classes + 1)]
    limites[-1] = maximo  # garante que o último valor entre na última classe

    contagens = [0] * n_classes
    for v in valores:
        if v == maximo:
            idx = n_classes - 1
        else:
            idx = min(int((v - minimo) / largura), n_classes - 1)
        contagens[idx] += 1

    linhas = []
    acumulada = 0
    for i in range(n_classes):
        acumulada += contagens[i]
        linhas.append(
            {
                "Classe": f"[{limites[i]:.2f} – {limites[i + 1]:.2f})"
                if i < n_classes - 1
                else f"[{limites[i]:.2f} – {limites[i + 1]:.2f}]",
                "Frequência": contagens[i],
                "Frequência relativa (%)": round(100 * contagens[i] / n, 2),
                "Frequência acumulada": acumulada,
            }
        )
    return pd.DataFrame(linhas)


def tabela_frequencias_categorica(dados: pd.Series) -> pd.DataFrame:
    """Tabela de frequências absolutas e relativas para uma variável categórica."""
    contagem = dados.value_counts(dropna=True)
    total = int(contagem.sum())
    df = pd.DataFrame(
        {
            "Categoria": contagem.index.astype(str),
            "Frequência": contagem.values,
            "Frequência relativa (%)": (contagem.values / total * 100).round(2),
        }
    )
    return df.reset_index(drop=True)


def detectar_outliers_iqr(dados: List[float]) -> Tuple[List[float], float, float]:
    """
    Detecta outliers pela regra do IQR (intervalo interquartil), usando os
    quartis calculados pela biblioteca própria:
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
    """
    q1, _, q3 = ms.quartis(dados)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = [v for v in dados if v < limite_inferior or v > limite_superior]
    return outliers, limite_inferior, limite_superior


def interpretar_assimetria(media: float, mediana: float, desvio_padrao: float) -> str:
    """Gera uma leitura textual simples da forma da distribuição comparando média e mediana."""
    if desvio_padrao == 0:
        return "A variável não apresenta dispersão (todos os valores são iguais)."
    diferenca_padronizada = (media - mediana) / desvio_padrao
    if abs(diferenca_padronizada) < 0.05:
        return "Média e mediana são próximas — a distribuição parece aproximadamente simétrica."
    elif diferenca_padronizada > 0:
        return (
            "A média é maior que a mediana — indício de assimetria positiva (cauda longa à "
            "direita), comum quando há poucos valores muito altos puxando a média para cima."
        )
    else:
        return (
            "A média é menor que a mediana — indício de assimetria negativa (cauda longa à "
            "esquerda), com poucos valores muito baixos puxando a média para baixo."
        )


def interpretar_cv(cv: float) -> str:
    """Classifica o coeficiente de variação em faixas de dispersão relativa."""
    if cv < 15:
        return f"Coeficiente de variação de {cv:.2f}% — dispersão **baixa**: os dados são relativamente homogêneos."
    elif cv < 30:
        return f"Coeficiente de variação de {cv:.2f}% — dispersão **moderada**."
    else:
        return f"Coeficiente de variação de {cv:.2f}% — dispersão **alta**: os dados são bastante heterogêneos."
