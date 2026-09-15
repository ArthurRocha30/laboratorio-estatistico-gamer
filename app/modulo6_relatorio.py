"""Módulo 6 — Relatório de Descobertas."""

import streamlit as st

import minhastats as ms
from app.utils import carregar_dados, detectar_outliers_iqr


def render():
    st.header("💡 Módulo 6 — Relatório de Descobertas")
    st.caption(
        "As três descobertas abaixo são recalculadas em tempo real pela biblioteca "
        "`minhastats`, a partir do dataset Video Game Sales. Uma versão em Markdown "
        "está disponível em `relatorio/RELATORIO.md`."
    )

    df = carregar_dados()
    vendas_globais = df["Global_Sales"].dropna().tolist()

    media = ms.media(vendas_globais)
    mediana = ms.mediana(vendas_globais)
    cv = ms.coeficiente_variacao(vendas_globais, "amostral")
    outliers, li, ls = detectar_outliers_iqr(vendas_globais)

    sub = df[["NA_Sales", "EU_Sales"]].dropna()
    r = ms.correlacao_pearson(sub["EU_Sales"].tolist(), sub["NA_Sales"].tolist())
    reg = ms.regressao_linear(sub["EU_Sales"].tolist(), sub["NA_Sales"].tolist())

    genero_top = df["Genre"].value_counts().idxmax()
    genero_top_n = int(df["Genre"].value_counts().max())

    st.subheader("Descoberta 1 — O mercado é dominado por poucos \"blockbusters\"")
    st.markdown(
        f"""
        A média de vendas globais (**{media:.3f} milhões de cópias**) é mais de
        **{media / mediana:.1f} vezes maior** que a mediana (**{mediana:.3f} milhões**),
        e o coeficiente de variação chega a **{cv:.0f}%**. A regra do IQR classifica
        **{len(outliers)} jogos ({100 * len(outliers) / len(vendas_globais):.1f}%)** como
        outliers de venda (acima de {ls:.3f} milhões). Ou seja: a maioria absoluta dos
        jogos vende pouco, e um pequeno grupo de sucessos (como *Wii Sports*, com 82,74
        milhões de cópias) puxa a média para cima — uma distribuição fortemente
        assimétrica à direita, melhor descrita por uma cauda longa (tipo exponencial)
        do que por uma Normal simétrica (ver Módulo 4).
        """
    )

    st.subheader("Descoberta 2 — Vendas em regiões diferentes crescem juntas, mas isso não é causalidade")
    st.markdown(
        f"""
        Existe uma correlação forte e positiva entre vendas na Europa e na América do
        Norte (**r = {r:.3f}**, R² = {reg['r2']:.3f}), ou seja, cerca de
        **{reg['r2'] * 100:.1f}%** da variação das vendas norte-americanas de um jogo
        pode ser "explicada" linearmente pelas vendas europeias. Isso **não** significa
        que vender bem na Europa *causa* vender bem nos EUA — o mais provável é que
        ambas sejam efeito de uma causa comum: a **qualidade/popularidade do próprio
        jogo** (franquias como Mario, GTA ou FIFA vendem bem nos dois mercados pelo
        mesmo motivo).
        """
    )

    st.subheader('Descoberta 3 — O TCL "funciona" mesmo com uma variável tão assimétrica')
    st.markdown(
        f"""
        O gênero mais comum no catálogo é **{genero_top}**, com {genero_top_n} jogos.
        Apesar de `Global_Sales` ser extremamente assimétrica (Descoberta 1), a
        simulação de Monte Carlo do Módulo 3 mostra que, ao sortear repetidamente
        amostras dessa variável e calcular suas médias, a distribuição dessas médias
        amostrais se aproxima de uma curva Normal conforme o tamanho da amostra cresce
        (visível já com n ≈ 30) — a confirmação prática do Teorema Central do Limite
        mesmo partindo de uma população que está longe de ser normal.
        """
    )

    st.divider()
    st.caption(
        "Relatório completo, com fórmulas e decisões de projeto, em "
        "`relatorio/RELATORIO.md`."
    )
