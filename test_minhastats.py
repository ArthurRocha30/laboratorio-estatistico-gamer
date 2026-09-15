"""
test_minhastats.py
==================

Testes automatizados da nossa biblioteca (Modulo 1).

A ideia: para cada funcao que escrevemos "na mao" em minhastats.py,
comparamos o resultado dela com o resultado da MESMA conta feita pelo
NumPy ou pelo SciPy. Se bater, nossa formula esta certa.

Para rodar:  pytest -v

TOLERANCIA NUMERICA
-------------------
Nao da para comparar numeros com "==" porque o computador guarda numeros
com casas decimais de forma aproximada. Por isso conferimos se a diferenca
é menor que uma tolerancia:

    TOLERANCIA         = 1e-10  -> listas pequenas (poucos valores)
    TOLERANCIA_GRANDE  = 1e-9   -> dataset real (16.598 valores)

No dataset real usamos uma tolerancia um pouco maior porque quanto mais
numeros somamos, mais os pequenos erros de arredondamento se acumulam.
"""

import numpy as np
import pandas as pd
from scipy import stats

import minhastats as ms

TOLERANCIA = 1e-10
TOLERANCIA_GRANDE = 1e-9

# Listas usadas nos testes
dados_x = [4, 8, 15, 16, 23, 42, 8, 15, 15, 4, 30, 12, 7, 19, 21]
dados_y = [2, 6, 13, 14, 20, 40, 10, 15, 13, 5, 28, 11, 9, 17, 22]
dados_impar = [7, 1, 9, 3, 5]
dados_par = [7, 1, 9, 3, 5, 11]


# ==========================================================
# TENDENCIA CENTRAL
# ==========================================================

def test_media():
    assert abs(ms.media(dados_x) - np.mean(dados_x)) < TOLERANCIA


def test_mediana_com_quantidade_impar():
    assert abs(ms.mediana(dados_impar) - np.median(dados_impar)) < TOLERANCIA


def test_mediana_com_quantidade_par():
    assert abs(ms.mediana(dados_par) - np.median(dados_par)) < TOLERANCIA


def test_moda():
    # o 15 aparece 3 vezes, é o valor que mais se repete
    assert ms.moda(dados_x) == [15]


def test_moda_com_dois_valores_repetidos():
    # 2 e 3 aparecem 2 vezes cada -> duas modas
    assert ms.moda([1, 2, 2, 3, 3, 4]) == [2, 3]


def test_moda_quando_nada_se_repete():
    # nenhum valor se repete -> lista vazia (amodal)
    assert ms.moda([1, 2, 3, 4, 5]) == []


# ==========================================================
# DISPERSAO
# ==========================================================

def test_amplitude():
    assert ms.amplitude(dados_x) == max(dados_x) - min(dados_x)


def test_variancia_amostral():
    # ddof=1 faz o numpy dividir por (n - 1), igual a nossa amostral
    assert abs(ms.variancia(dados_x) - np.var(dados_x, ddof=1)) < TOLERANCIA


def test_variancia_populacional():
    # ddof=0 faz o numpy dividir por n
    assert abs(ms.variancia(dados_x, populacional=True) - np.var(dados_x, ddof=0)) < TOLERANCIA


def test_desvio_padrao_amostral():
    assert abs(ms.desvio_padrao(dados_x) - np.std(dados_x, ddof=1)) < TOLERANCIA


def test_desvio_padrao_populacional():
    assert abs(ms.desvio_padrao(dados_x, populacional=True) - np.std(dados_x, ddof=0)) < TOLERANCIA


def test_coeficiente_variacao():
    esperado = (np.std(dados_x, ddof=1) / np.mean(dados_x)) * 100
    assert abs(ms.coeficiente_variacao(dados_x) - esperado) < TOLERANCIA


# ==========================================================
# PERCENTIS E QUARTIS
# ==========================================================

def test_percentil():
    for p in [0, 10, 25, 50, 75, 90, 100]:
        assert abs(ms.percentil(dados_x, p) - np.percentile(dados_x, p)) < TOLERANCIA


def test_quartis():
    q1, q2, q3 = ms.quartis(dados_x)
    assert abs(q1 - np.percentile(dados_x, 25)) < TOLERANCIA
    assert abs(q2 - np.percentile(dados_x, 50)) < TOLERANCIA
    assert abs(q3 - np.percentile(dados_x, 75)) < TOLERANCIA


def test_quartil_do_meio_e_a_mediana():
    # Q2 tem que dar exatamente a mesma coisa que a mediana
    q1, q2, q3 = ms.quartis(dados_x)
    assert abs(q2 - ms.mediana(dados_x)) < TOLERANCIA


# ==========================================================
# COVARIANCIA E CORRELACAO
# ==========================================================

def test_covariancia_amostral():
    esperado = np.cov(dados_x, dados_y, ddof=1)[0][1]
    assert abs(ms.covariancia(dados_x, dados_y) - esperado) < TOLERANCIA


def test_covariancia_populacional():
    esperado = np.cov(dados_x, dados_y, ddof=0)[0][1]
    assert abs(ms.covariancia(dados_x, dados_y, populacional=True) - esperado) < TOLERANCIA


def test_correlacao_comparando_com_numpy():
    esperado = np.corrcoef(dados_x, dados_y)[0][1]
    assert abs(ms.correlacao(dados_x, dados_y) - esperado) < TOLERANCIA


def test_correlacao_comparando_com_scipy():
    esperado = stats.pearsonr(dados_x, dados_y)[0]
    assert abs(ms.correlacao(dados_x, dados_y) - esperado) < TOLERANCIA


def test_correlacao_positiva_perfeita():
    # y é exatamente o dobro de x -> r tem que dar 1
    assert abs(ms.correlacao([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) - 1) < TOLERANCIA


def test_correlacao_negativa_perfeita():
    # quando x cresce, y cai na mesma proporcao -> r tem que dar -1
    assert abs(ms.correlacao([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) + 1) < TOLERANCIA


# ==========================================================
# REGRESSAO LINEAR
# ==========================================================

def test_regressao_linear():
    a, b = ms.regressao_linear(dados_x, dados_y)
    referencia = stats.linregress(dados_x, dados_y)
    assert abs(b - referencia.slope) < TOLERANCIA       # inclinacao
    assert abs(a - referencia.intercept) < TOLERANCIA   # intercepto


def test_r_quadrado():
    referencia = stats.linregress(dados_x, dados_y)
    esperado = referencia.rvalue ** 2
    assert abs(ms.r_quadrado(dados_x, dados_y) - esperado) < TOLERANCIA


def test_regressao_com_reta_exata():
    # esses pontos estao TODOS na reta y = 1 + 2x
    x = [0, 1, 2, 3, 4]
    y = [1, 3, 5, 7, 9]
    a, b = ms.regressao_linear(x, y)
    assert abs(a - 1) < TOLERANCIA
    assert abs(b - 2) < TOLERANCIA
    assert abs(ms.r_quadrado(x, y) - 1) < TOLERANCIA     # acerta tudo
    assert abs(ms.prever(a, b, 10) - 21) < TOLERANCIA    # 1 + 2*10 = 21


# ==========================================================
# DISTRIBUICOES TEORICAS
# ==========================================================

def test_densidade_normal():
    for x in [-2, -1, 0, 0.5, 1, 3]:
        esperado = stats.norm.pdf(x, loc=2, scale=1.5)
        assert abs(ms.densidade_normal(x, 2, 1.5) - esperado) < TOLERANCIA


def test_probabilidade_binomial():
    for k in range(0, 11):
        esperado = stats.binom.pmf(k, 10, 0.3)
        assert abs(ms.probabilidade_binomial(k, 10, 0.3) - esperado) < TOLERANCIA


def test_probabilidade_poisson():
    for k in range(0, 15):
        esperado = stats.poisson.pmf(k, 4.2)
        assert abs(ms.probabilidade_poisson(k, 4.2) - esperado) < TOLERANCIA


def test_densidade_uniforme():
    for x in [1, 2.5, 4, 5, 7]:
        esperado = stats.uniform.pdf(x, loc=2, scale=3)   # intervalo de 2 a 5
        assert abs(ms.densidade_uniforme(x, 2, 5) - esperado) < TOLERANCIA


def test_densidade_exponencial():
    for x in [0, 0.5, 1, 3, 10]:
        esperado = stats.expon.pdf(x, scale=1 / 0.7)      # taxa = 0.7
        assert abs(ms.densidade_exponencial(x, 0.7) - esperado) < TOLERANCIA


def test_fatorial_e_combinacao():
    assert ms.fatorial(5) == 120          # 5*4*3*2*1
    assert ms.combinacao(5, 2) == 10      # C(5,2) = 10


# ==========================================================
# TESTES COM O DATASET REAL DO TRABALHO
# ==========================================================

def carregar_vendas():
    """Le a coluna de vendas globais do nosso dataset."""
    tabela = pd.read_csv("data/vgsales.csv")
    return tabela["Global_Sales"].dropna().tolist()


def test_media_no_dataset_real():
    vendas = carregar_vendas()
    assert abs(ms.media(vendas) - np.mean(vendas)) < TOLERANCIA_GRANDE


def test_desvio_padrao_no_dataset_real():
    vendas = carregar_vendas()
    assert abs(ms.desvio_padrao(vendas) - np.std(vendas, ddof=1)) < TOLERANCIA_GRANDE


def test_quartis_no_dataset_real():
    vendas = carregar_vendas()
    q1, q2, q3 = ms.quartis(vendas)
    assert abs(q1 - np.percentile(vendas, 25)) < TOLERANCIA_GRANDE
    assert abs(q2 - np.percentile(vendas, 50)) < TOLERANCIA_GRANDE
    assert abs(q3 - np.percentile(vendas, 75)) < TOLERANCIA_GRANDE


def test_correlacao_no_dataset_real():
    tabela = pd.read_csv("data/vgsales.csv")
    vendas_eu = tabela["EU_Sales"].tolist()
    vendas_na = tabela["NA_Sales"].tolist()
    esperado = np.corrcoef(vendas_eu, vendas_na)[0][1]
    assert abs(ms.correlacao(vendas_eu, vendas_na) - esperado) < TOLERANCIA_GRANDE
