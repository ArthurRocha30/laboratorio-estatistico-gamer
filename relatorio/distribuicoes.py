"""
Funções de densidade/probabilidade de distribuições teóricas, implementadas
manualmente a partir das fórmulas fechadas (sem scipy.stats.*.pdf/pmf).

Usa apenas a biblioteca padrão `math` (constantes e, pi, exponenciação) —
nunca uma função de distribuição pronta.
"""

from __future__ import annotations

import math


def normal_pdf(x: float, media: float, desvio_padrao: float) -> float:
    """
    Densidade da distribuição Normal(média, desvio_padrão) no ponto x:

        f(x) = 1 / (σ * sqrt(2π)) * exp(-(x - μ)² / (2σ²))
    """
    if desvio_padrao <= 0:
        raise ValueError("O desvio padrão deve ser positivo.")
    coeficiente = 1.0 / (desvio_padrao * math.sqrt(2 * math.pi))
    expoente = -((x - media) ** 2) / (2 * desvio_padrao ** 2)
    return coeficiente * math.exp(expoente)


def _fatorial(n: int) -> int:
    if n < 0 or int(n) != n:
        raise ValueError("Fatorial definido apenas para inteiros não-negativos.")
    resultado = 1
    for i in range(2, int(n) + 1):
        resultado *= i
    return resultado


def _combinacao(n: int, k: int) -> int:
    """Combinação C(n, k) = n! / (k! * (n-k)!), calculada via fatorial próprio."""
    if k < 0 or k > n:
        return 0
    return _fatorial(n) // (_fatorial(k) * _fatorial(n - k))


def binomial_pmf(k: int, n: int, p: float) -> float:
    """
    Probabilidade de exatamente k sucessos em n tentativas de Bernoulli(p):

        P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
    """
    if not 0 <= p <= 1:
        raise ValueError("p deve estar entre 0 e 1.")
    k = int(k)
    return _combinacao(n, k) * (p ** k) * ((1 - p) ** (n - k))


def poisson_pmf(k: int, lam: float) -> float:
    """
    Probabilidade de observar exatamente k eventos em um intervalo, dado
    que a taxa média de ocorrência é lam (λ):

        P(X = k) = (λ^k * e^-λ) / k!
    """
    if lam <= 0:
        raise ValueError("λ (taxa média) deve ser positivo.")
    k = int(k)
    return (lam ** k) * math.exp(-lam) / _fatorial(k)


def uniforme_pdf(x: float, a: float, b: float) -> float:
    """
    Densidade da distribuição Uniforme contínua no intervalo [a, b]:

        f(x) = 1 / (b - a),  se a <= x <= b
        f(x) = 0,            caso contrário
    """
    if a >= b:
        raise ValueError("É necessário que a < b.")
    return 1.0 / (b - a) if a <= x <= b else 0.0


def exponencial_pdf(x: float, lam: float) -> float:
    """
    Densidade da distribuição Exponencial com taxa lam (λ):

        f(x) = λ * e^(-λx),  se x >= 0
        f(x) = 0,            caso contrário
    """
    if lam <= 0:
        raise ValueError("λ (taxa) deve ser positivo.")
    return lam * math.exp(-lam * x) if x >= 0 else 0.0
