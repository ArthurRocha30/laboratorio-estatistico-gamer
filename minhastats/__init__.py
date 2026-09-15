"""
minhastats
==========

Biblioteca estatística implementada do zero (sem uso de funções prontas de
estatística de terceiros, como numpy.mean, numpy.std, numpy.percentile,
statistics.* ou scipy.stats.*) para o Laboratório de Estatística.

Submódulos:
    - descritiva:  medidas de tendência central, dispersão e posição.
    - correlacao:  covariância, correlação de Pearson e regressão linear
                    simples (mínimos quadrados).
    - distribuicoes: funções de densidade/probabilidade de distribuições
                    teóricas (Normal, Binomial, Poisson, Uniforme, Exponencial).

Todas as funções recebem sequências (list, tuple, pandas.Series, numpy.array)
de números e devolvem valores em float puro (ou estruturas simples), sem
depender de bibliotecas de estatística prontas para o cálculo em si.
"""

from .descritiva import (
    media,
    mediana,
    moda,
    amplitude,
    variancia,
    desvio_padrao,
    percentil,
    quartis,
    coeficiente_variacao,
)
from .correlacao import (
    covariancia,
    correlacao_pearson,
    regressao_linear,
)
from .distribuicoes import (
    normal_pdf,
    binomial_pmf,
    poisson_pmf,
    uniforme_pdf,
    exponencial_pdf,
)

__all__ = [
    "media",
    "mediana",
    "moda",
    "amplitude",
    "variancia",
    "desvio_padrao",
    "percentil",
    "quartis",
    "coeficiente_variacao",
    "covariancia",
    "correlacao_pearson",
    "regressao_linear",
    "normal_pdf",
    "binomial_pmf",
    "poisson_pmf",
    "uniforme_pdf",
    "exponencial_pdf",
]
