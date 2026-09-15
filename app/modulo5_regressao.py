"""Módulo 5 — Correlação e Regressão Linear Simples (mínimos quadrados)."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import minhastats as ms
from app.utils import carregar_dados, colunas_numericas


def render():
    st.header("📈 Módulo 5 — Correlação e Regressão Linear")
    st.caption(
        "Correlação e regressão calculadas pela biblioteca própria `minhastats` "
        "(mínimos quadrados ordinários implementados do zero)."
    )

    df = carregar_dados()
    numericas = colunas_numericas(df)

    col1, col2 = st.columns(2)
    var_x = col1.selectbox("Variável X (explicativa)", numericas, index=numericas.index("EU_Sales") if "EU_Sales" in numericas else 0)
    var_y = col2.selectbox("Variável Y (resposta)", numericas, index=numericas.index("NA_Sales") if "NA_Sales" in numericas else 1)

    if var_x == var_y:
        st.warning("Selecione duas variáveis diferentes para X e Y.")
        return

    subset = df[[var_x, var_y]].dropna()
    x = subset[var_x].astype(float).tolist()
    y = subset[var_y].astype(float).tolist()
    st.caption(f"{len(x)} pares válidos (de {len(df)} registros) após remover ausentes.")

    r = ms.correlacao_pearson(x, y)
    resultado = ms.regressao_linear(x, y)
    a, b, r2 = resultado["intercepto"], resultado["inclinacao"], resultado["r2"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Correlação de Pearson (r)", f"{r:,.4f}")
    c2.metric("R² da regressão", f"{r2:,.4f}")
    c3.metric("Coeficiente angular (b)", f"{b:,.4f}")

    st.subheader("Diagrama de dispersão e reta ajustada")
    xs_linha = np.linspace(min(x), max(x), 100)
    ys_linha = [resultado["prever"](xv) for xv in xs_linha]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observações", opacity=0.5))
    fig.add_trace(go.Scatter(x=xs_linha, y=ys_linha, mode="lines", name="Reta de mínimos quadrados", line=dict(color="red")))
    fig.update_layout(title=f"{var_y} em função de {var_x}", xaxis_title=var_x, yaxis_title=var_y)
    st.plotly_chart(fig, width="stretch")

    sinal = "+" if a >= 0 else "−"
    st.subheader("Equação da reta")
    st.latex(rf"\hat y = {b:.4f} \cdot x {sinal} {abs(a):.4f}")

    st.subheader("Predição interativa")
    valor_x = st.number_input(f"Digite um valor de {var_x} para prever {var_y}:", value=float(round(np.mean(x), 2)))
    valor_previsto = resultado["prever"](valor_x)
    st.success(f"Predição: {var_y} ≈ **{valor_previsto:,.4f}** quando {var_x} = {valor_x:g}")

    st.subheader("Interpretação dos coeficientes")
    st.markdown(
        f"""
        - **Inclinação (b = {b:.4f})**: a cada aumento de 1 unidade em **{var_x}**,
          o valor esperado de **{var_y}** varia em **{b:.4f}** unidades, mantendo o
          padrão observado nos dados.
        - **Intercepto (a = {a:.4f})**: valor esperado de **{var_y}** quando
          **{var_x} = 0** (nem sempre tem interpretação prática se x=0 estiver fora
          do intervalo observado dos dados).
        - **R² = {r2:.4f}**: cerca de **{r2 * 100:.1f}%** da variação de **{var_y}**
          é "explicada" linearmente por **{var_x}**; o restante se deve a outros
          fatores não incluídos no modelo.
        """
    )

    st.warning(
        "⚠️ **Correlação não implica causalidade!** Um valor alto de r (ou R²) mostra "
        "apenas que as duas variáveis se movem juntas de forma linear — não prova que "
        "uma causa a outra. Aqui, por exemplo, vendas de um jogo na Europa e na América "
        "do Norte tendem a crescer juntas porque ambas refletem a **popularidade geral "
        "do jogo**, e não porque vender mais em um continente *causa* vender mais no outro."
    )
