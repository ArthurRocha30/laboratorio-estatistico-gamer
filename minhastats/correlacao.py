"""
Covariância, correlação de Pearson e regressão linear simples pelo método
dos mínimos quadrados — tudo implementado manualmente.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Union

from .descritiva import _para_lista, desvio_padrao, media

Numero = Union[int, float]


def _validar_pares(x: Iterable[Numero], y: Iterable[Numero]) -> tuple[List[float], List[float]]:
    lx = _para_lista(x)
    ly = _para_lista(y)
    if len(lx) != len(ly):
        raise ValueError("x e y devem ter o mesmo número de observações.")
    if len(lx) < 2:
        raise ValueError("São necessárias ao menos 2 observações pareadas.")
    return lx, ly


def covariancia(x: Iterable[Numero], y: Iterable[Numero], tipo: str = "amostral") -> float:
    """
    Covariância entre duas variáveis: média (ou média n-1) do produto dos
    desvios de x e y em relação às suas respectivas médias.
    """
    lx, ly = _validar_pares(x, y)
    n = len(lx)
    mx, my = media(lx), media(ly)
    soma = sum((xi - mx) * (yi - my) for xi, yi in zip(lx, ly))

    if tipo == "amostral":
        return soma / (n - 1)
    elif tipo == "populacional":
        return soma / n
    else:
        raise ValueError('tipo deve ser "amostral" ou "populacional".')


def correlacao_pearson(x: Iterable[Numero], y: Iterable[Numero]) -> float:
    """
    Coeficiente de correlação de Pearson (r): covariância normalizada pelo
    produto dos desvios padrão. Varia de -1 (correlação linear negativa
    perfeita) a +1 (correlação linear positiva perfeita); 0 indica ausência
    de correlação linear.
    """
    lx, ly = _validar_pares(x, y)
    dp_x = desvio_padrao(lx, "amostral")
    dp_y = desvio_padrao(ly, "amostral")
    if dp_x == 0 or dp_y == 0:
        raise ValueError("Correlação indefinida quando uma variável é constante.")
    cov = covariancia(lx, ly, "amostral")
    return cov / (dp_x * dp_y)


def regressao_linear(x: Iterable[Numero], y: Iterable[Numero]) -> Dict[str, object]:
    """
    Regressão linear simples y = a + b*x pelo método dos mínimos quadrados
    ordinários (OLS), implementado a partir das somas de desvios (sem usar
    numpy.polyfit / scipy.stats.linregress / sklearn).

        b (inclinação) = Sum((xi - x̄)(yi - ȳ)) / Sum((xi - x̄)²)
        a (intercepto) = ȳ - b * x̄

    Retorna um dicionário com:
        intercepto, inclinacao, r2 (coeficiente de determinação) e
        prever(x_novo) -> função que aplica a reta ajustada a um novo x.
    """
    lx, ly = _validar_pares(x, y)
    mx, my = media(lx), media(ly)

    s_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(lx, ly))
    s_xx = sum((xi - mx) ** 2 for xi in lx)
    if s_xx == 0:
        raise ValueError("Regressão indefinida quando x é constante.")

    b = s_xy / s_xx
    a = my - b * mx

    y_previstos = [a + b * xi for xi in lx]
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(ly, y_previstos))
    ss_tot = sum((yi - my) ** 2 for yi in ly)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    def prever(x_novo: Numero) -> float:
        return a + b * float(x_novo)

    return {
        "intercepto": a,
        "inclinacao": b,
        "r2": r2,
        "prever": prever,
    }
