"""Módulo 4 — Sobreposição de distribuições teóricas ao histograma dos dados."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import minhastats as ms
from app.utils import carregar_dados, colunas_numericas


def render():
    st.header("📐 Módulo 4 — Distribuições Teóricas")
    st.caption(
        "O histograma é normalizado (densidade) e comparado a curvas teóricas cujos "
        "parâmetros são estimados a partir dos próprios dados (método dos momentos), "
        "usando as medidas da biblioteca `minhastats`."
    )

    df = carregar_dados()
    numericas = colunas_numericas(df)
    coluna = st.selectbox(
        "Variável numérica",
        numericas,
        index=numericas.index("Global_Sales") if "Global_Sales" in numericas else 0,
    )
    dados = df[coluna].dropna().astype(float)
    dados = dados[dados >= 0] if (dados < 0).any() else dados  # evita valores negativos para dist. não-negativas
    dados_lista = dados.tolist()

    media = ms.media(dados_lista)
    desvio = ms.desvio_padrao(dados_lista, "populacional")

    st.write(f"**Média estimada (μ̂):** {media:,.4f}  ·  **Desvio padrão estimado (σ̂):** {desvio:,.4f}")

    opcoes_dist = ["Normal", "Exponencial", "Uniforme", "Poisson", "Binomial"]
    distribuicoes = st.multiselect(
        "Distribuições teóricas a sobrepor (compare ao menos a Normal com mais uma)",
        opcoes_dist,
        default=["Normal", "Exponencial"],
    )

    n_bins = st.slider("Número de classes do histograma", 5, 60, 30)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=dados_lista, histnorm="probability density", nbinsx=n_bins, name=f"{coluna} (empírico)", opacity=0.6)
    )

    minimo, maximo = float(dados.min()), float(dados.max())
    xs = np.linspace(minimo, maximo, 400)

    if "Normal" in distribuicoes:
        ys = [ms.normal_pdf(x, media, desvio) if desvio > 0 else 0 for x in xs]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Normal(μ={media:.2f}, σ={desvio:.2f})"))

    if "Exponencial" in distribuicoes:
        if media > 0:
            lam = 1 / media  # estimador de momentos: E[X] = 1/λ
            ys = [ms.exponencial_pdf(x, lam) for x in xs]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Exponencial(λ={lam:.3f})"))
        else:
            st.warning("Não é possível ajustar Exponencial: média não positiva.")

    if "Uniforme" in distribuicoes:
        ys = [ms.uniforme_pdf(x, minimo, maximo) for x in xs]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Uniforme(a={minimo:.2f}, b={maximo:.2f})"))

    if "Poisson" in distribuicoes:
        # Poisson é discreta: plotamos como pontos/hastes sobre a escala inteira.
        lam = media
        k_max = int(min(maximo, media + 6 * (desvio + 1)))
        ks = list(range(0, k_max + 1))
        ys = [ms.poisson_pmf(k, lam) if lam > 0 else 0 for k in ks]
        fig.add_trace(go.Scatter(x=ks, y=ys, mode="markers+lines", name=f"Poisson(λ={lam:.2f})", line=dict(dash="dot")))

    if "Binomial" in distribuicoes:
        # Estimador de momentos: p = 1 - σ²/μ (aprox.), n = μ/p, arredondados.
        variancia = desvio ** 2
        if media > 0 and variancia > 0 and variancia < media:
            p_est = 1 - variancia / media
            n_est = max(1, round(media / p_est)) if p_est > 0 else 1
            ks = list(range(0, n_est + 1))
            ys = [ms.binomial_pmf(k, n_est, p_est) for k in ks]
            fig.add_trace(go.Scatter(x=ks, y=ys, mode="markers+lines", name=f"Binomial(n={n_est}, p={p_est:.3f})", line=dict(dash="dot")))
        else:
            st.warning("Não foi possível estimar parâmetros Binomiais razoáveis para esta variável (requer variância < média).")

    fig.update_layout(title=f"Ajuste de distribuições teóricas — {coluna}", xaxis_title=coluna, yaxis_title="Densidade")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Discussão sobre a qualidade do ajuste")
    assimetria_aprox = (media - ms.mediana(dados_lista)) / desvio if desvio > 0 else 0
    if abs(assimetria_aprox) < 0.1:
        st.info(
            "A distribuição empírica é aproximadamente simétrica em torno da média, o que "
            "favorece um bom ajuste pela curva **Normal**."
        )
    else:
        st.info(
            "A distribuição empírica é visivelmente assimétrica (cauda longa de um dos lados), "
            "então a curva **Normal** tende a se ajustar mal nas extremidades — repare como uma "
            "distribuição **Exponencial** (para variáveis não-negativas e concentradas perto de "
            "zero, como vendas) costuma acompanhar melhor o formato do histograma."
        )
    st.caption(
        "Este é um diagnóstico visual, como pedido pelo enunciado — compare a altura e a forma "
        "da curva teórica com as barras do histograma: bom ajuste = a curva acompanha o contorno "
        "das barras; ajuste ruim = a curva sub ou superestima sistematicamente regiões do histograma."
    )
