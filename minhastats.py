"""
minhastats.py
=============

Biblioteca de estatistica feita pelo nosso grupo (Modulo 1).

Todas as funcoes foram escritas "na mao", a partir das formulas.
Nao usamos nenhuma funcao pronta de estatistica (nada de numpy.mean,
numpy.std, statistics.media, scipy.stats...). Usamos apenas o modulo
`math` da biblioteca padrao do Python (para raiz quadrada, exponencial e pi).

Em todas as funcoes, `dados` é uma lista de numeros.
"""

import math


# ==========================================================
# MEDIDAS DE TENDENCIA CENTRAL
# ==========================================================

def media(dados):
    """
    Media aritmetica.

    Formula:  x̄ = (x1 + x2 + ... + xn) / n
    """
    soma = 0
    for valor in dados:
        soma = soma + valor
    return soma / len(dados)


def mediana(dados):
    """
    Mediana: o valor que fica no meio dos dados ordenados.

    - Se a quantidade de dados (n) for IMPAR: é o valor do meio.
    - Se for PAR: é a media dos dois valores do meio.
    """
    ordenados = sorted(dados)
    n = len(ordenados)
    meio = n // 2

    if n % 2 == 1:
        return ordenados[meio]
    else:
        return (ordenados[meio - 1] + ordenados[meio]) / 2


def moda(dados):
    """
    Moda: o valor que mais se repete.

    Retorna uma LISTA, porque pode existir mais de uma moda (bimodal, etc).
    Se nenhum valor se repetir, retorna uma lista vazia (dados amodais).
    """
    # 1) conta quantas vezes cada valor aparece
    contagem = {}
    for valor in dados:
        if valor in contagem:
            contagem[valor] = contagem[valor] + 1
        else:
            contagem[valor] = 1

    # 2) descobre qual foi a maior contagem
    maior_frequencia = 0
    for valor in contagem:
        if contagem[valor] > maior_frequencia:
            maior_frequencia = contagem[valor]

    # se o valor mais repetido apareceu so 1 vez, nao existe moda
    if maior_frequencia == 1:
        return []

    # 3) junta todos os valores que tem essa maior contagem
    modas = []
    for valor in contagem:
        if contagem[valor] == maior_frequencia:
            modas.append(valor)

    return sorted(modas)


# ==========================================================
# MEDIDAS DE DISPERSAO
# ==========================================================

def amplitude(dados):
    """
    Amplitude: a distancia entre o maior e o menor valor.

    Formula:  A = maior valor - menor valor
    """
    return max(dados) - min(dados)


def variancia(dados, populacional=False):
    """
    Variancia: a media dos quadrados das diferencas ate a media.

    Formula amostral (padrao):   s² = Σ(xi - x̄)² / (n - 1)
    Formula populacional:        σ² = Σ(xi - x̄)² / n

    Usamos (n - 1) na amostral porque quando temos so uma AMOSTRA
    (e nao a populacao inteira) isso corrige a tendencia de subestimar
    a variancia real.
    """
    n = len(dados)
    m = media(dados)

    # soma dos quadrados das diferencas ate a media
    soma_dos_quadrados = 0
    for valor in dados:
        soma_dos_quadrados = soma_dos_quadrados + (valor - m) ** 2

    if populacional:
        return soma_dos_quadrados / n
    else:
        return soma_dos_quadrados / (n - 1)


def desvio_padrao(dados, populacional=False):
    """
    Desvio padrao: a raiz quadrada da variancia.

    Formula:  s = √(s²)

    Serve para voltar para a mesma unidade dos dados originais
    (a variancia esta "ao quadrado").
    """
    return math.sqrt(variancia(dados, populacional))


def coeficiente_variacao(dados, populacional=False):
    """
    Coeficiente de variacao (em %): o desvio padrao comparado com a media.

    Formula:  CV = (s / |x̄|) * 100

    Serve para comparar a dispersao de variaveis com escalas diferentes.
    """
    return (desvio_padrao(dados, populacional) / abs(media(dados))) * 100


# ==========================================================
# MEDIDAS DE POSICAO (PERCENTIS E QUARTIS)
# ==========================================================

def percentil(dados, p):
    """
    Percentil p (de 0 a 100): o valor que deixa p% dos dados abaixo dele.

    Calculamos a posicao com:  posicao = (p / 100) * (n - 1)

    Quando essa posicao cai entre dois valores, fazemos uma interpolacao
    linear entre eles (é o mesmo metodo que o numpy.percentile usa por
    padrao, por isso nossos testes batem com o dele).
    """
    ordenados = sorted(dados)
    n = len(ordenados)

    posicao = (p / 100) * (n - 1)

    indice_de_baixo = int(posicao)          # parte inteira da posicao
    quanto_falta = posicao - indice_de_baixo  # parte decimal da posicao

    # se a posicao caiu exatamente no ultimo valor, devolve ele
    if indice_de_baixo + 1 >= n:
        return ordenados[indice_de_baixo]

    valor_de_baixo = ordenados[indice_de_baixo]
    valor_de_cima = ordenados[indice_de_baixo + 1]

    # "anda" a fracao que falta entre um valor e o outro
    return valor_de_baixo + quanto_falta * (valor_de_cima - valor_de_baixo)


def quartis(dados):
    """
    Quartis: dividem os dados ordenados em 4 partes iguais.

    Q1 = percentil 25  (25% dos dados estao abaixo)
    Q2 = percentil 50  (é a propria mediana)
    Q3 = percentil 75  (75% dos dados estao abaixo)

    Retorna a tripla (Q1, Q2, Q3).
    """
    q1 = percentil(dados, 25)
    q2 = percentil(dados, 50)
    q3 = percentil(dados, 75)
    return q1, q2, q3


# ==========================================================
# RELACAO ENTRE DUAS VARIAVEIS
# ==========================================================

def covariancia(x, y, populacional=False):
    """
    Covariancia: mede se duas variaveis crescem juntas.

    Formula amostral:      cov = Σ(xi - x̄)(yi - ȳ) / (n - 1)
    Formula populacional:  cov = Σ(xi - x̄)(yi - ȳ) / n

    - Positiva: quando x cresce, y tende a crescer.
    - Negativa: quando x cresce, y tende a diminuir.

    O problema da covariancia é que ela depende da unidade das variaveis,
    por isso normalmente usamos a correlacao (abaixo).
    """
    n = len(x)
    media_x = media(x)
    media_y = media(y)

    soma = 0
    for i in range(n):
        soma = soma + (x[i] - media_x) * (y[i] - media_y)

    if populacional:
        return soma / n
    else:
        return soma / (n - 1)


def correlacao(x, y):
    """
    Coeficiente de correlacao de Pearson (r).

    Formula:  r = cov(x, y) / (desvio_padrao(x) * desvio_padrao(y))

    O resultado fica sempre entre -1 e 1:
        r =  1  -> relacao linear positiva perfeita
        r =  0  -> nao existe relacao linear
        r = -1  -> relacao linear negativa perfeita
    """
    return covariancia(x, y) / (desvio_padrao(x) * desvio_padrao(y))


# ==========================================================
# REGRESSAO LINEAR (METODO DOS MINIMOS QUADRADOS)
# ==========================================================

def regressao_linear(x, y):
    """
    Calcula a reta que melhor passa no meio dos pontos:  ŷ = a + b*x

    Metodo dos minimos quadrados (a reta que deixa a soma dos erros
    ao quadrado o menor possivel):

        b = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²      (inclinacao)
        a = ȳ - b * x̄                            (intercepto)

    Retorna a dupla (a, b).
    """
    media_x = media(x)
    media_y = media(y)

    numerador = 0
    denominador = 0
    for i in range(len(x)):
        numerador = numerador + (x[i] - media_x) * (y[i] - media_y)
        denominador = denominador + (x[i] - media_x) ** 2

    b = numerador / denominador
    a = media_y - b * media_x

    return a, b


def prever(a, b, valor_de_x):
    """
    Usa a reta ŷ = a + b*x para prever o y de um x qualquer.
    """
    return a + b * valor_de_x


def r_quadrado(x, y):
    """
    Coeficiente de determinacao (R²): o quanto da variacao de y
    a nossa reta consegue explicar.

    Formula:  R² = 1 - (Σ(yi - ŷi)²) / (Σ(yi - ȳ)²)

        Σ(yi - ŷi)²  = o que a reta ERROU
        Σ(yi - ȳ)²   = a variacao total de y

    R² = 1   -> a reta acerta tudo
    R² = 0   -> a reta nao explica nada
    """
    a, b = regressao_linear(x, y)
    media_y = media(y)

    soma_dos_erros = 0
    soma_total = 0
    for i in range(len(x)):
        y_previsto = prever(a, b, x[i])
        soma_dos_erros = soma_dos_erros + (y[i] - y_previsto) ** 2
        soma_total = soma_total + (y[i] - media_y) ** 2

    return 1 - (soma_dos_erros / soma_total)


# ==========================================================
# DISTRIBUICOES TEORICAS (usadas no Modulo 4)
# ==========================================================

def densidade_normal(x, m, dp):
    """
    Curva Normal (aquela em forma de sino), com media `m` e desvio padrao `dp`.

    Formula:  f(x) = 1 / (dp * √(2π)) * e^( -(x - m)² / (2 * dp²) )
    """
    parte_da_frente = 1 / (dp * math.sqrt(2 * math.pi))
    expoente = -((x - m) ** 2) / (2 * dp ** 2)
    return parte_da_frente * math.exp(expoente)


def fatorial(n):
    """
    Fatorial de n:  n! = n * (n-1) * (n-2) * ... * 1
    (usado nas formulas da Binomial e da Poisson)
    """
    resultado = 1
    for i in range(2, n + 1):
        resultado = resultado * i
    return resultado


def combinacao(n, k):
    """
    Combinacao: de quantos jeitos da para escolher k itens entre n.

    Formula:  C(n, k) = n! / (k! * (n - k)!)
    """
    return fatorial(n) // (fatorial(k) * fatorial(n - k))


def probabilidade_binomial(k, n, p):
    """
    Distribuicao Binomial: probabilidade de dar EXATAMENTE k sucessos
    em n tentativas, quando cada tentativa tem probabilidade p de sucesso.

    Formula:  P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)

    Exemplo: chance de tirar exatamente 3 caras em 10 jogadas de moeda.
    """
    return combinacao(n, k) * (p ** k) * ((1 - p) ** (n - k))


def probabilidade_poisson(k, media_de_eventos):
    """
    Distribuicao de Poisson: probabilidade de acontecerem exatamente k
    eventos num intervalo, sabendo que em media acontecem
    `media_de_eventos` (o lambda) por intervalo.

    Formula:  P(X = k) = (λ^k * e^(-λ)) / k!
    """
    lam = media_de_eventos
    return (lam ** k) * math.exp(-lam) / fatorial(k)


def densidade_uniforme(x, a, b):
    """
    Distribuicao Uniforme: todos os valores entre a e b tem a mesma chance.

    Formula:  f(x) = 1 / (b - a)   se a <= x <= b
              f(x) = 0             fora desse intervalo
    """
    if a <= x <= b:
        return 1 / (b - a)
    else:
        return 0


def densidade_exponencial(x, taxa):
    """
    Distribuicao Exponencial: muito usada quando a maioria dos valores é
    pequena e poucos valores sao bem grandes (cauda longa a direita).

    Formula:  f(x) = λ * e^(-λx)   para x >= 0

    A taxa λ é estimada como 1 / media dos dados.
    """
    if x < 0:
        return 0
    return taxa * math.exp(-taxa * x)
