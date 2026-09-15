"""
Testes automatizados que validam a biblioteca própria `minhastats`
comparando seus resultados com NumPy/SciPy (usados aqui APENAS como
referência de conferência, nunca dentro da biblioteca implementada).

Execução:  pytest -v
"""

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

import minhastats as ms

TOLERANCIA = 1e-9

# Conjuntos de dados usados nos testes -------------------------------------
DADOS_A = [4, 8, 15, 16, 23, 42, 8, 15, 15, 4, 30, 12, 7, 19, 21]
DADOS_B = [2, 6, 13, 14, 20, 40, 10, 15, 13, 5, 28, 11, 9, 17, 22]
DADOS_IMPAR = [7, 1, 9, 3, 5]
DADOS_PAR = [7, 1, 9, 3, 5, 11]
DADOS_MULTIMODAL = [1, 2, 2, 3, 3, 4]
DADOS_AMODAL = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Medidas de tendência central
# ---------------------------------------------------------------------------
class TestMedia:
    def test_media_bate_com_numpy(self):
        assert ms.media(DADOS_A) == pytest.approx(np.mean(DADOS_A), abs=TOLERANCIA)

    def test_media_valor_unico(self):
        assert ms.media([42]) == 42.0

    def test_media_lista_vazia_levanta_erro(self):
        with pytest.raises(ValueError):
            ms.media([])


class TestMediana:
    def test_mediana_impar_bate_com_numpy(self):
        assert ms.mediana(DADOS_IMPAR) == pytest.approx(np.median(DADOS_IMPAR), abs=TOLERANCIA)

    def test_mediana_par_bate_com_numpy(self):
        assert ms.mediana(DADOS_PAR) == pytest.approx(np.median(DADOS_PAR), abs=TOLERANCIA)

    def test_mediana_dataset_grande_bate_com_numpy(self):
        rng = np.random.default_rng(42)
        amostra = rng.normal(loc=10, scale=3, size=1000)
        assert ms.mediana(amostra) == pytest.approx(np.median(amostra), abs=TOLERANCIA)


class TestModa:
    def test_moda_multimodal(self):
        assert ms.moda(DADOS_MULTIMODAL) == [2, 3]

    def test_moda_amodal_retorna_lista_vazia(self):
        assert ms.moda(DADOS_AMODAL) == []

    def test_moda_unimodal_bate_com_scipy(self):
        dados = [1, 1, 1, 2, 3, 4]
        moda_scipy = scipy_stats.mode(dados, keepdims=False).mode
        assert ms.moda(dados) == [float(moda_scipy)]


# ---------------------------------------------------------------------------
# Medidas de dispersão
# ---------------------------------------------------------------------------
class TestAmplitude:
    def test_amplitude(self):
        assert ms.amplitude(DADOS_A) == max(DADOS_A) - min(DADOS_A)


class TestVariancia:
    def test_variancia_amostral_bate_com_numpy(self):
        assert ms.variancia(DADOS_A, "amostral") == pytest.approx(
            np.var(DADOS_A, ddof=1), abs=TOLERANCIA
        )

    def test_variancia_populacional_bate_com_numpy(self):
        assert ms.variancia(DADOS_A, "populacional") == pytest.approx(
            np.var(DADOS_A, ddof=0), abs=TOLERANCIA
        )

    def test_variancia_um_elemento_levanta_erro(self):
        with pytest.raises(ValueError):
            ms.variancia([5], "amostral")

    def test_tipo_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            ms.variancia(DADOS_A, "invalido")


class TestDesvioPadrao:
    def test_desvio_padrao_amostral_bate_com_numpy(self):
        assert ms.desvio_padrao(DADOS_A, "amostral") == pytest.approx(
            np.std(DADOS_A, ddof=1), abs=TOLERANCIA
        )

    def test_desvio_padrao_populacional_bate_com_numpy(self):
        assert ms.desvio_padrao(DADOS_A, "populacional") == pytest.approx(
            np.std(DADOS_A, ddof=0), abs=TOLERANCIA
        )


class TestPercentisEQuartis:
    @pytest.mark.parametrize("p", [0, 10, 25, 50, 75, 90, 100])
    def test_percentil_bate_com_numpy(self, p):
        assert ms.percentil(DADOS_A, p) == pytest.approx(
            np.percentile(DADOS_A, p), abs=TOLERANCIA
        )

    def test_quartis_bate_com_numpy(self):
        q1, q2, q3 = ms.quartis(DADOS_A)
        q1_np, q2_np, q3_np = np.percentile(DADOS_A, [25, 50, 75])
        assert q1 == pytest.approx(q1_np, abs=TOLERANCIA)
        assert q2 == pytest.approx(q2_np, abs=TOLERANCIA)
        assert q3 == pytest.approx(q3_np, abs=TOLERANCIA)

    def test_percentil_dataset_grande_bate_com_numpy(self):
        rng = np.random.default_rng(7)
        amostra = rng.exponential(scale=2.0, size=2000)
        for p in [5, 25, 50, 75, 95]:
            assert ms.percentil(amostra, p) == pytest.approx(
                np.percentile(amostra, p), abs=TOLERANCIA
            )

    def test_percentil_fora_do_intervalo_levanta_erro(self):
        with pytest.raises(ValueError):
            ms.percentil(DADOS_A, 150)


class TestCoeficienteVariacao:
    def test_coeficiente_variacao_formula(self):
        cv_esperado = (np.std(DADOS_A, ddof=1) / np.mean(DADOS_A)) * 100
        assert ms.coeficiente_variacao(DADOS_A, "amostral") == pytest.approx(
            cv_esperado, abs=TOLERANCIA
        )

    def test_coeficiente_variacao_media_zero_levanta_erro(self):
        with pytest.raises(ValueError):
            ms.coeficiente_variacao([-2, -1, 0, 1, 2], "amostral")


# ---------------------------------------------------------------------------
# Covariância, correlação e regressão
# ---------------------------------------------------------------------------
class TestCovariancia:
    def test_covariancia_amostral_bate_com_numpy(self):
        cov_np = np.cov(DADOS_A, DADOS_B, ddof=1)[0, 1]
        assert ms.covariancia(DADOS_A, DADOS_B, "amostral") == pytest.approx(cov_np, abs=TOLERANCIA)

    def test_covariancia_populacional_bate_com_numpy(self):
        cov_np = np.cov(DADOS_A, DADOS_B, ddof=0)[0, 1]
        assert ms.covariancia(DADOS_A, DADOS_B, "populacional") == pytest.approx(cov_np, abs=TOLERANCIA)

    def test_tamanhos_diferentes_levanta_erro(self):
        with pytest.raises(ValueError):
            ms.covariancia([1, 2, 3], [1, 2])


class TestCorrelacaoPearson:
    def test_correlacao_bate_com_numpy_corrcoef(self):
        r_np = np.corrcoef(DADOS_A, DADOS_B)[0, 1]
        assert ms.correlacao_pearson(DADOS_A, DADOS_B) == pytest.approx(r_np, abs=TOLERANCIA)

    def test_correlacao_bate_com_scipy_pearsonr(self):
        r_scipy, _ = scipy_stats.pearsonr(DADOS_A, DADOS_B)
        assert ms.correlacao_pearson(DADOS_A, DADOS_B) == pytest.approx(r_scipy, abs=TOLERANCIA)

    def test_correlacao_perfeita_positiva(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        assert ms.correlacao_pearson(x, y) == pytest.approx(1.0, abs=TOLERANCIA)

    def test_correlacao_perfeita_negativa(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        assert ms.correlacao_pearson(x, y) == pytest.approx(-1.0, abs=TOLERANCIA)

    def test_variavel_constante_levanta_erro(self):
        with pytest.raises(ValueError):
            ms.correlacao_pearson([1, 1, 1, 1], [1, 2, 3, 4])


class TestRegressaoLinear:
    def test_coeficientes_batem_com_scipy_linregress(self):
        resultado = ms.regressao_linear(DADOS_A, DADOS_B)
        referencia = scipy_stats.linregress(DADOS_A, DADOS_B)
        assert resultado["inclinacao"] == pytest.approx(referencia.slope, abs=TOLERANCIA)
        assert resultado["intercepto"] == pytest.approx(referencia.intercept, abs=TOLERANCIA)
        assert resultado["r2"] == pytest.approx(referencia.rvalue ** 2, abs=TOLERANCIA)

    def test_predicao_reta_perfeita(self):
        x = [0, 1, 2, 3, 4]
        y = [1, 3, 5, 7, 9]  # y = 1 + 2x
        resultado = ms.regressao_linear(x, y)
        assert resultado["inclinacao"] == pytest.approx(2.0, abs=TOLERANCIA)
        assert resultado["intercepto"] == pytest.approx(1.0, abs=TOLERANCIA)
        assert resultado["r2"] == pytest.approx(1.0, abs=TOLERANCIA)
        assert resultado["prever"](10) == pytest.approx(21.0, abs=TOLERANCIA)

    def test_x_constante_levanta_erro(self):
        with pytest.raises(ValueError):
            ms.regressao_linear([5, 5, 5], [1, 2, 3])

    def test_dataset_grande_bate_com_scipy(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 100, size=500)
        ruido = rng.normal(0, 5, size=500)
        y = 3.5 * x - 12 + ruido
        resultado = ms.regressao_linear(x, y)
        referencia = scipy_stats.linregress(x, y)
        assert resultado["inclinacao"] == pytest.approx(referencia.slope, abs=1e-6)
        assert resultado["intercepto"] == pytest.approx(referencia.intercept, abs=1e-6)
        assert resultado["r2"] == pytest.approx(referencia.rvalue ** 2, abs=1e-6)


# ---------------------------------------------------------------------------
# Distribuições teóricas
# ---------------------------------------------------------------------------
class TestDistribuicoes:
    def test_normal_pdf_bate_com_scipy(self):
        for x in [-2, -1, 0, 0.5, 1, 3]:
            esperado = scipy_stats.norm.pdf(x, loc=2, scale=1.5)
            assert ms.normal_pdf(x, 2, 1.5) == pytest.approx(esperado, abs=TOLERANCIA)

    def test_binomial_pmf_bate_com_scipy(self):
        for k in range(0, 11):
            esperado = scipy_stats.binom.pmf(k, 10, 0.3)
            assert ms.binomial_pmf(k, 10, 0.3) == pytest.approx(esperado, abs=TOLERANCIA)

    def test_poisson_pmf_bate_com_scipy(self):
        for k in range(0, 15):
            esperado = scipy_stats.poisson.pmf(k, 4.2)
            assert ms.poisson_pmf(k, 4.2) == pytest.approx(esperado, abs=TOLERANCIA)

    def test_uniforme_pdf_bate_com_scipy(self):
        for x in [1, 2.5, 4, 5, 7]:
            esperado = scipy_stats.uniform.pdf(x, loc=2, scale=3)  # intervalo [2, 5]
            assert ms.uniforme_pdf(x, 2, 5) == pytest.approx(esperado, abs=TOLERANCIA)

    def test_exponencial_pdf_bate_com_scipy(self):
        for x in [0, 0.5, 1, 3, 10]:
            esperado = scipy_stats.expon.pdf(x, scale=1 / 0.7)  # lambda = 0.7
            assert ms.exponencial_pdf(x, 0.7) == pytest.approx(esperado, abs=TOLERANCIA)

    def test_normal_pdf_desvio_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            ms.normal_pdf(0, 0, -1)


# ---------------------------------------------------------------------------
# Integração com o dataset real do laboratório
# ---------------------------------------------------------------------------
class TestComDatasetReal:
    """Confere a biblioteca própria contra NumPy/SciPy usando colunas reais
    do dataset do laboratório (data/vgsales.csv), incluindo valores ausentes."""

    @staticmethod
    @pytest.fixture(scope="class")
    def coluna_vendas():
        import pandas as pd

        df = pd.read_csv("data/vgsales.csv")
        return df["Global_Sales"].dropna().tolist()

    def test_media_no_dataset_real(self, coluna_vendas):
        assert ms.media(coluna_vendas) == pytest.approx(np.mean(coluna_vendas), abs=1e-9)

    def test_desvio_padrao_no_dataset_real(self, coluna_vendas):
        assert ms.desvio_padrao(coluna_vendas, "amostral") == pytest.approx(
            np.std(coluna_vendas, ddof=1), abs=1e-9
        )

    def test_quartis_no_dataset_real(self, coluna_vendas):
        q1, q2, q3 = ms.quartis(coluna_vendas)
        q1_np, q2_np, q3_np = np.percentile(coluna_vendas, [25, 50, 75])
        assert q1 == pytest.approx(q1_np, abs=1e-9)
        assert q2 == pytest.approx(q2_np, abs=1e-9)
        assert q3 == pytest.approx(q3_np, abs=1e-9)
