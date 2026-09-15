"""Módulo 2 — Estatística Descritiva Interativa."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import minhastats as ms
from app.utils import (
    carregar_dados,
    colunas_categoricas,
    colunas_numericas,
    detectar_outliers_iqr,
    interpretar_assimetria,
    interpretar_cv,
    tabela_frequencias_categorica,
    tabela_frequencias_numerica,
)


def _render_numerica(df, coluna: str):
    dados = df[coluna].dropna().astype(float).tolist()
    st.caption(f"{len(dados)} valores válidos (de {len(df)} registros) após remover ausentes.")

    tipo = st.radio(
        "Tipo de variância/desvio/CV",
        ["amostral", "populacional"],
        horizontal=True,
        key=f"tipo_{coluna}",
    )

    media = ms.media(dados)
    mediana = ms.mediana(dados)
    moda = ms.moda(dados)
    amplitude = ms.amplitude(dados)
    variancia = ms.variancia(dados, tipo)
    desvio = ms.desvio_padrao(dados, tipo)
    q1, q2, q3 = ms.quartis(dados)
    cv = ms.coeficiente_variacao(dados, tipo)

    st.subheader("Medidas de tendência central")
    c1, c2, c3 = st.columns(3)
    c1.metric("Média", f"{media:,.3f}")
    c2.metric("Mediana", f"{mediana:,.3f}")
    c3.metric("Moda", ", ".join(f"{m:g}" for m in moda) if moda else "amodal")

    st.subheader("Medidas de dispersão")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Amplitude", f"{amplitude:,.3f}")
    d2.metric(f"Variância ({tipo})", f"{variancia:,.3f}")
    d3.metric(f"Desvio padrão ({tipo})", f"{desvio:,.3f}")
    d4.metric(f"Coef. variação ({tipo})", f"{cv:,.2f}%")

    st.subheader("Medidas de posição (quartis)")
    q1c, q2c, q3c = st.columns(3)
    q1c.metric("Q1 (25%)", f"{q1:,.3f}")
    q2c.metric("Q2 / Mediana (50%)", f"{q2:,.3f}")
    q3c.metric("Q3 (75%)", f"{q3:,.3f}")

    st.subheader("Interpretação automática")
    st.info(interpretar_assimetria(media, mediana, desvio))
    st.info(interpretar_cv(cv))

    st.subheader("Tabela de frequências (em classes)")
    n_classes = st.slider("Número de classes", min_value=3, max_value=30, value=10, key=f"classes_{coluna}")
    tabela = tabela_frequencias_numerica(dados, n_classes)
    st.dataframe(tabela, use_container_width=True)

    st.subheader("Gráficos")
    col_hist, col_box = st.columns(2)
    with col_hist:
        fig_hist = px.histogram(x=dados, nbins=n_classes, title=f"Histograma — {coluna}")
        fig_hist.update_layout(xaxis_title=coluna, yaxis_title="Frequência")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_box:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(x=dados, name=coluna, boxpoints="outliers"))
        fig_box.update_layout(title=f"Boxplot — {coluna}")
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Detecção de outliers (regra do IQR)")
    outliers, limite_inf, limite_sup = detectar_outliers_iqr(dados)
    st.write(
        f"Limite inferior = Q1 − 1.5·IQR = **{limite_inf:,.3f}** · "
        f"Limite superior = Q3 + 1.5·IQR = **{limite_sup:,.3f}**"
    )
    if outliers:
        st.warning(f"{len(outliers)} outlier(s) detectado(s) ({100 * len(outliers) / len(dados):.1f}% dos dados).")
        with st.expander("Ver valores"):
            st.write(sorted(outliers))
    else:
        st.success("Nenhum outlier detectado pela regra do IQR.")


def _render_categorica(df, coluna: str):
    dados = df[coluna].dropna()
    st.caption(f"{len(dados)} valores válidos (de {len(df)} registros) após remover ausentes.")

    tabela = tabela_frequencias_categorica(dados)
    top_n = st.slider("Mostrar top N categorias", min_value=5, max_value=min(40, len(tabela)), value=min(15, len(tabela)))
    tabela_top = tabela.head(top_n)

    st.subheader("Tabela de frequências")
    st.dataframe(tabela, use_container_width=True)

    st.subheader("Gráficos")
    col_bar, col_pie = st.columns(2)
    with col_bar:
        fig_bar = px.bar(tabela_top, x="Categoria", y="Frequência", title=f"Barras — {coluna} (top {top_n})")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_pie:
        fig_pie = px.pie(tabela_top, names="Categoria", values="Frequência", title=f"Pizza — {coluna} (top {top_n})")
        st.plotly_chart(fig_pie, use_container_width=True)

    moda_texto = tabela.iloc[0]["Categoria"]
    st.info(
        f"A categoria mais frequente (moda) é **{moda_texto}**, presente em "
        f"{tabela.iloc[0]['Frequência']} registros ({tabela.iloc[0]['Frequência relativa (%)']}% do total)."
    )


def render():
    st.header("📊 Módulo 2 — Estatística Descritiva Interativa")
    st.caption(
        "Todas as medidas abaixo (média, mediana, moda, variância, desvio padrão, "
        "quartis e coeficiente de variação) são calculadas pela biblioteca própria "
        "`minhastats`, validada por testes automatizados no Módulo 1."
    )

    df = carregar_dados()
    numericas = colunas_numericas(df)
    categoricas = colunas_categoricas(df)

    tipo_variavel = st.radio("Tipo de variável", ["Numérica", "Categórica"], horizontal=True)

    if tipo_variavel == "Numérica":
        coluna = st.selectbox("Selecione a variável numérica", numericas, index=numericas.index("Global_Sales") if "Global_Sales" in numericas else 0)
        _render_numerica(df, coluna)
    else:
        colunas_cat_uteis = [c for c in categoricas if c != "Name"]
        coluna = st.selectbox("Selecione a variável categórica", colunas_cat_uteis)
        _render_categorica(df, coluna)
