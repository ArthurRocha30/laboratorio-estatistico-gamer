"""
Ponto de entrada da aplicação Streamlit.

Execução (a partir da raiz do projeto):
    streamlit run app/main.py
"""

import os
import sys

# Garante que a raiz do projeto esteja no sys.path, para que `import minhastats`
# e `import app.*` funcionem independentemente do diretório de onde o
# Streamlit foi iniciado.
_RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAIZ_PROJETO not in sys.path:
    sys.path.insert(0, _RAIZ_PROJETO)

import streamlit as st

from app import (
    modulo0_dados,
    modulo2_descritiva,
    modulo3_simulacao,
    modulo4_distribuicoes,
    modulo5_regressao,
    modulo6_relatorio,
)

st.set_page_config(
    page_title="Laboratório de Estatística — Video Game Sales",
    page_icon="🎮",
    layout="wide",
)

PAGINAS = {
    "Módulo 0 — Dados Reais": modulo0_dados.render,
    "Módulo 2 — Estatística Descritiva": modulo2_descritiva.render,
    "Módulo 3 — Probabilidade e Simulação": modulo3_simulacao.render,
    "Módulo 4 — Distribuições Teóricas": modulo4_distribuicoes.render,
    "Módulo 5 — Correlação e Regressão": modulo5_regressao.render,
    "Módulo 6 — Relatório de Descobertas": modulo6_relatorio.render,
}

st.sidebar.title("🎮 Laboratório de Estatística")
st.sidebar.markdown("**Dataset:** Video Game Sales")
pagina_escolhida = st.sidebar.radio("Navegação", list(PAGINAS.keys()))

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Módulo 1** (núcleo estatístico próprio `minhastats` + testes
    automatizados) não é uma página interativa — é a biblioteca que
    alimenta todas as páginas acima. Veja `minhastats/` e
    `tests/test_minhastats.py`.
    """
)

PAGINAS[pagina_escolhida]()
