"""Módulo 0 — Apresentação do dataset real usado no laboratório."""

import streamlit as st

from app.utils import carregar_dados, colunas_categoricas, colunas_numericas


def render():
    st.header("🎮 Módulo 0 — Dados Reais")

    st.markdown(
        """
        **Dataset:** [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales)
        (vendas globais de jogos por plataforma, gênero e publicadora,
        resultado de uma raspagem do site VGChartz).

        **Tema:** jogos / e-sports.
        """
    )

    df = carregar_dados()
    numericas = colunas_numericas(df)
    categoricas = colunas_categoricas(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", f"{len(df):,}".replace(",", "."))
    col2.metric("Variáveis numéricas", len(numericas))
    col3.metric("Variáveis categóricas", len(categoricas))

    st.subheader("Amostra dos dados")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("Dicionário de variáveis")
    dicionario = {
        "Rank": ("numérica", "Posição no ranking de vendas globais."),
        "Name": ("categórica", "Nome do jogo."),
        "Platform": ("categórica", "Plataforma de lançamento (Wii, PS4, NES, ...)."),
        "Year": ("numérica", "Ano de lançamento."),
        "Genre": ("categórica", "Gênero do jogo (Ação, Esporte, RPG, ...)."),
        "Publisher": ("categórica", "Empresa publicadora do jogo."),
        "NA_Sales": ("numérica", "Vendas na América do Norte (milhões de cópias)."),
        "EU_Sales": ("numérica", "Vendas na Europa (milhões de cópias)."),
        "JP_Sales": ("numérica", "Vendas no Japão (milhões de cópias)."),
        "Other_Sales": ("numérica", "Vendas no resto do mundo (milhões de cópias)."),
        "Global_Sales": ("numérica", "Vendas globais totais (milhões de cópias)."),
    }
    st.table(
        {
            "Variável": list(dicionario.keys()),
            "Tipo": [v[0] for v in dicionario.values()],
            "Descrição": [v[1] for v in dicionario.values()],
        }
    )

    st.subheader("Valores ausentes por coluna")
    ausentes = df.isna().sum()
    ausentes = ausentes[ausentes > 0]
    if ausentes.empty:
        st.success("Não há valores ausentes no dataset.")
    else:
        st.dataframe(ausentes.rename("Valores ausentes"), width="stretch")
        st.caption(
            "As páginas seguintes removem automaticamente valores ausentes da "
            "variável selecionada antes de calcular qualquer medida."
        )
