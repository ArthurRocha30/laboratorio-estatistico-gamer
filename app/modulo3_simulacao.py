"""Módulo 3 — Probabilidade e Simulação de Monte Carlo (LGN e TCL)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import minhastats as ms
from app.utils import carregar_dados, colunas_numericas


def _lei_dos_grandes_numeros():
    st.subheader("(a) Lei dos Grandes Números")
    st.markdown(
        "Simulamos lançamentos de uma moeda (ou dado) e acompanhamos a "
        "**frequência relativa** de um resultado específico conforme o número "
        "de lançamentos cresce. A Lei dos Grandes Números prevê que essa "
        "frequência converge para a probabilidade teórica."
    )

    experimento = st.selectbox("Experimento", ["Moeda (cara/coroa)", "Dado (tirar o número 6)"])
    n_lancamentos = st.slider("Número de lançamentos (repetições)", 100, 50_000, 5_000, step=100)
    semente = st.number_input("Semente aleatória (para reprodutibilidade)", value=42, step=1)

    rng = np.random.default_rng(int(semente))

    if experimento == "Moeda (cara/coroa)":
        p_teorica = 0.5
        resultados = rng.integers(0, 2, size=n_lancamentos)  # 1 = cara
        rotulo = "cara"
    else:
        p_teorica = 1 / 6
        resultados = (rng.integers(1, 7, size=n_lancamentos) == 6).astype(int)
        rotulo = "número 6"

    frequencias_relativas = np.cumsum(resultados) / np.arange(1, n_lancamentos + 1)
    frequencia_final = float(frequencias_relativas[-1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=frequencias_relativas, mode="lines", name="Frequência relativa acumulada"))
    fig.add_hline(y=p_teorica, line_dash="dash", line_color="red", annotation_text=f"P teórica = {p_teorica:.4f}")
    fig.update_layout(
        title=f"Convergência da frequência relativa de '{rotulo}'",
        xaxis_title="Número de lançamentos",
        yaxis_title="Frequência relativa acumulada",
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        f"Após {n_lancamentos:,} lançamentos, a frequência relativa observada foi "
        f"**{frequencia_final:.4f}**, contra a probabilidade teórica de **{p_teorica:.4f}** "
        f"— diferença absoluta de {abs(frequencia_final - p_teorica):.4f}. Quanto maior o número "
        "de repetições, menor tende a ser essa diferença."
    )


def _teorema_central_do_limite(df):
    st.subheader("(b) Teorema Central do Limite (TCL)")
    st.markdown(
        "Sorteamos repetidamente amostras (com reposição) de uma variável numérica do "
        "dataset e calculamos a **média de cada amostra** (usando `minhastats.media`). "
        "O TCL prevê que a distribuição dessas médias amostrais se aproxima de uma "
        "Normal conforme o tamanho da amostra cresce — mesmo que a variável original "
        "não seja normal."
    )

    numericas = colunas_numericas(df)
    coluna = st.selectbox(
        "Variável do dataset a amostrar", numericas,
        index=numericas.index("Global_Sales") if "Global_Sales" in numericas else 0,
        key="tcl_coluna",
    )
    populacao = df[coluna].dropna().astype(float).tolist()

    col1, col2 = st.columns(2)
    n_amostras = col1.slider("Número de amostras (repetições)", 100, 10_000, 2_000, step=100)
    tamanho_amostra = col2.slider("Tamanho de cada amostra (n)", 2, 500, 30, step=1)
    semente = st.number_input("Semente aleatória", value=7, step=1, key="tcl_semente")

    rng = np.random.default_rng(int(semente))
    populacao_arr = np.array(populacao)

    medias_amostrais = []
    for _ in range(n_amostras):
        amostra = rng.choice(populacao_arr, size=tamanho_amostra, replace=True)
        medias_amostrais.append(ms.media(amostra))

    media_pop = ms.media(populacao)
    desvio_pop = ms.desvio_padrao(populacao, "populacional")
    erro_padrao_teorico = desvio_pop / (tamanho_amostra ** 0.5)

    media_das_medias = ms.media(medias_amostrais)
    desvio_das_medias = ms.desvio_padrao(medias_amostrais, "amostral")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=medias_amostrais,
            histnorm="probability density",
            name="Médias amostrais simuladas",
            nbinsx=40,
        )
    )

    xs = np.linspace(min(medias_amostrais), max(medias_amostrais), 300)
    ys = [ms.normal_pdf(x, media_pop, erro_padrao_teorico) for x in xs]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Normal teórica (μ, erro padrão)", line=dict(color="red")))

    fig.update_layout(
        title=f"Distribuição das médias amostrais (n = {tamanho_amostra}, {n_amostras} amostras)",
        xaxis_title=f"Média amostral de {coluna}",
        yaxis_title="Densidade",
    )
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Média da população", f"{media_pop:,.4f}")
    c2.metric("Média das médias amostrais", f"{media_das_medias:,.4f}")
    c3.metric("Desvio padrão das médias (empírico vs. teórico)", f"{desvio_das_medias:,.4f} / {erro_padrao_teorico:,.4f}")

    st.info(
        "Observe que a média das médias amostrais se aproxima da média populacional, e o "
        "desvio padrão das médias se aproxima do **erro padrão teórico** (σ/√n). Aumente o "
        "tamanho da amostra (n) e note como a distribuição das médias fica cada vez mais "
        "próxima da curva Normal vermelha, mesmo que a variável original seja bastante "
        "assimétrica — essa é a essência do Teorema Central do Limite."
    )


def render():
    st.header("🎲 Módulo 3 — Probabilidade e Simulação (Monte Carlo)")
    df = carregar_dados()
    _lei_dos_grandes_numeros()
    st.divider()
    _teorema_central_do_limite(df)
