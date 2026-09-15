"""
Medidas de tendência central, dispersão e posição — implementadas "na unha",
sem chamar funções estatísticas prontas de outras bibliotecas.

Convenções:
    - `dados`: qualquer sequência iterável de números (list, tuple,
      pandas.Series, numpy.array...). É sempre convertida para list(float)
      internamente.
    - `tipo`: "amostral" (divide por n-1, padrão) ou "populacional"
      (divide por n) para variância, desvio padrão e coeficiente de variação.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

Numero = Union[int, float]


def _para_lista(dados: Iterable[Numero]) -> List[float]:
    lista = [float(x) for x in dados]
    if len(lista) == 0:
        raise ValueError("A sequência de dados não pode estar vazia.")
    return lista


def media(dados: Iterable[Numero]) -> float:
    """Média aritmética: soma dos valores dividida pela quantidade de valores."""
    d = _para_lista(dados)
    return sum(d) / len(d)


def mediana(dados: Iterable[Numero]) -> float:
    """
    Mediana: valor central da sequência ordenada.
    Para n par, é a média aritmética dos dois valores centrais.
    """
    d = sorted(_para_lista(dados))
    n = len(d)
    meio = n // 2
    if n % 2 == 1:
        return float(d[meio])
    return (d[meio - 1] + d[meio]) / 2.0


def moda(dados: Iterable[Numero]) -> List[float]:
    """
    Moda: valor(es) mais frequente(s).
    Retorna uma lista (pode ser multimodal). Lista vazia = amodal
    (nenhum valor se repete).
    """
    d = _para_lista(dados)
    frequencias = {}
    for valor in d:
        frequencias[valor] = frequencias.get(valor, 0) + 1
    freq_maxima = max(frequencias.values())
    if freq_maxima == 1:
        return []
    modas = sorted(v for v, f in frequencias.items() if f == freq_maxima)
    return modas


def amplitude(dados: Iterable[Numero]) -> float:
    """Amplitude total: diferença entre o maior e o menor valor."""
    d = _para_lista(dados)
    return max(d) - min(d)


def variancia(dados: Iterable[Numero], tipo: str = "amostral") -> float:
    """
    Variância: média dos quadrados dos desvios em torno da média.
        - "amostral" (padrão): divide pela soma dos quadrados por (n - 1)
          (estimador não-viesado da variância populacional).
        - "populacional": divide por n.
    """
    d = _para_lista(dados)
    n = len(d)
    m = media(d)
    soma_dos_quadrados = sum((x - m) ** 2 for x in d)

    if tipo == "amostral":
        if n < 2:
            raise ValueError("Variância amostral exige ao menos 2 valores.")
        return soma_dos_quadrados / (n - 1)
    elif tipo == "populacional":
        return soma_dos_quadrados / n
    else:
        raise ValueError('tipo deve ser "amostral" ou "populacional".')


def desvio_padrao(dados: Iterable[Numero], tipo: str = "amostral") -> float:
    """Desvio padrão: raiz quadrada da variância (amostral ou populacional)."""
    return variancia(dados, tipo) ** 0.5


def percentil(dados: Iterable[Numero], p: float) -> float:
    """
    Percentil p (0 <= p <= 100) pelo método de interpolação linear entre os
    dois pontos de dados mais próximos (mesmo método usado por padrão em
    numpy.percentile / R type 7), o que permite comparação direta nos testes.
    """
    if not 0 <= p <= 100:
        raise ValueError("p deve estar entre 0 e 100.")
    d = sorted(_para_lista(dados))
    n = len(d)
    if n == 1:
        return d[0]

    posicao = (p / 100) * (n - 1)
    indice_inferior = int(posicao // 1)
    fracao = posicao - indice_inferior

    if indice_inferior + 1 < n:
        return d[indice_inferior] + fracao * (d[indice_inferior + 1] - d[indice_inferior])
    return d[indice_inferior]


def quartis(dados: Iterable[Numero]) -> Sequence[float]:
    """Retorna a tupla (Q1, Q2, Q3) — os percentis 25, 50 e 75."""
    d = _para_lista(dados)
    return percentil(d, 25), percentil(d, 50), percentil(d, 75)


def coeficiente_variacao(dados: Iterable[Numero], tipo: str = "amostral") -> float:
    """
    Coeficiente de variação (%): desvio padrão relativo à média,
    útil para comparar a dispersão de variáveis com escalas diferentes.
    """
    d = _para_lista(dados)
    m = media(d)
    if m == 0:
        raise ValueError("Coeficiente de variação indefinido quando a média é 0.")
    dp = desvio_padrao(d, tipo)
    return (dp / abs(m)) * 100
