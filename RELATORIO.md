# Relatorio - Laboratorio de Estatistica

**Disciplina:** Matematica e Estatistica
**Dataset escolhido:** Video Game Sales (vendas de jogos de video game)

---

## 1. O dataset

- **Fonte:** [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) - dataset publicado no Kaggle
  pelo usuario gregorut, com dados originados de uma raspagem do site VGChartz.
  Como o Kaggle exige login para baixar, pegamos o mesmo CSV num espelho
  publico no GitHub (os numeros sao identicos ao do Kaggle). Vale registrar
  que, por vir do VGChartz, sao **estimativas** de vendas e nao numeros
  oficiais divulgados pelas empresas.
- **Tema:** jogos
- **Quantidade de registros:** 16.598 jogos
- **Variaveis numericas (6):** `Year`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales`
  (as vendas estao em milhoes de copias)
- **Variaveis categoricas (3):** `Platform`, `Genre`, `Publisher`

**Por que escolhemos:** o tema jogos é do interesse do grupo e o dataset atende todos os
requisitos (mais de 1.000 registros, mais de 4 variaveis numericas e mais de 2 categoricas).

**Valores faltando:** 271 jogos estao sem o ano de lancamento e 58 estao sem a publicadora.
Decidimos **nao apagar essas linhas do dataset inteiro**: a gente tira os valores vazios
apenas da variavel que esta sendo analisada no momento (funcao `pegar_valores` no `app.py`).
Assim nao perdemos dados de vendas por causa de um ano faltando.

---

## 2. Como o projeto foi organizado

| Arquivo | Para que serve |
|---|---|
| `minhastats.py` | Todas as nossas funcoes de estatistica (Modulo 1) |
| `test_minhastats.py` | Testes que comparam nossas funcoes com NumPy/SciPy |
| `app.py` | A aplicacao interativa (Modulos 0, 2, 3, 4, 5 e 6) |

Separamos o **nucleo** (as contas, em `minhastats.py`) da **interface** (as telas, em `app.py`)
de proposito: assim da para testar as formulas sem precisar abrir a aplicacao, e a mesma
funcao é reaproveitada em varias telas.

**Regra de ouro:** o pandas so lê o CSV e filtra colunas. Nenhuma medida mostrada na tela
vem de funcao pronta - todas vem do `minhastats.py`.

---

## 3. As formulas que implementamos

### Tendencia central

| Medida | Formula |
|---|---|
| Media | x̄ = (x₁ + x₂ + ... + xₙ) / n |
| Mediana | valor do meio dos dados ordenados (se n é par, media dos dois do meio) |
| Moda | o valor que mais se repete (pode ter mais de um, ou nenhum) |

### Dispersao

| Medida | Formula |
|---|---|
| Amplitude | A = maior valor − menor valor |
| Variancia amostral | s² = Σ(xᵢ − x̄)² / (n − 1) |
| Variancia populacional | σ² = Σ(xᵢ − x̄)² / n |
| Desvio padrao | s = √(s²) |
| Coeficiente de variacao | CV = (s / \|x̄\|) × 100 |

**Por que (n − 1) na amostral?** Quando temos so uma amostra e nao a populacao inteira,
dividir por n subestima a variancia real. Dividir por (n − 1) corrige isso. Na aplicacao,
o usuario escolhe qual das duas quer usar (tem uma caixinha no Modulo 2).

### Posicao

| Medida | Formula |
|---|---|
| Percentil p | posicao = (p/100) × (n − 1), com interpolacao linear entre os dois valores vizinhos |
| Quartis | Q1 = percentil 25, Q2 = percentil 50 (= mediana), Q3 = percentil 75 |

**Decisao:** existem varios metodos de calcular percentil. Escolhemos a interpolacao linear
porque é o mesmo metodo que o `numpy.percentile` usa por padrao - assim nossos testes
conseguem comparar os dois resultados diretamente.

### Relacao entre duas variaveis

| Medida | Formula |
|---|---|
| Covariancia | cov(x,y) = Σ(xᵢ − x̄)(yᵢ − ȳ) / (n − 1) |
| Correlacao de Pearson | r = cov(x,y) / (sₓ · s_y) |
| Reta (minimos quadrados) | ŷ = a + b·x, onde b = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)² e a = ȳ − b·x̄ |
| R² | R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)² |

### Distribuicoes teoricas

| Distribuicao | Formula |
|---|---|
| Normal | f(x) = 1/(σ√(2π)) · e^(−(x−μ)²/(2σ²)) |
| Binomial | P(X=k) = C(n,k) · pᵏ · (1−p)^(n−k) |
| Poisson | P(X=k) = (λᵏ · e^(−λ)) / k! |
| Uniforme | f(x) = 1/(b−a), para a ≤ x ≤ b |
| Exponencial | f(x) = λ · e^(−λx), para x ≥ 0 |

### Regra do IQR (outliers)

```
IQR = Q3 − Q1
limite de baixo = Q1 − 1,5 × IQR
limite de cima  = Q3 + 1,5 × IQR
```
Todo valor fora desses limites é marcado como outlier.

---

## 4. Como validamos (tolerancia numerica)

Rodando `pytest -v` sao executados **38 testes**. Cada teste faz a mesma conta de dois
jeitos - com a nossa funcao e com o NumPy/SciPy - e confere se a diferenca é minuscula.

Nao da para usar `==` porque o computador guarda numeros com casas decimais de forma
aproximada. Entao usamos tolerancias:

- **1e-10 (0,0000000001)** para as listas pequenas dos testes;
- **1e-9** para os testes com o dataset real (16.598 valores), porque quanto mais numeros
  a gente soma, mais os errinhos de arredondamento se acumulam.

Exemplos do que é comparado: `media` x `numpy.mean`, `variancia` x `numpy.var(ddof=1)`,
`percentil` x `numpy.percentile`, `correlacao` x `scipy.stats.pearsonr`,
`regressao_linear` x `scipy.stats.linregress`, `densidade_normal` x `scipy.stats.norm.pdf`.

---

## 5. As 3 descobertas

### Descoberta 1 - Pouquissimos jogos vendem muito

| Medida (Global_Sales) | Valor |
|---|---|
| Media | 0,537 milhoes de copias |
| Mediana | 0,170 milhoes de copias |
| Desvio padrao | 1,555 |
| Coeficiente de variacao | 289% |
| Q1 / Q2 / Q3 | 0,06 / 0,17 / 0,47 |
| Outliers (regra do IQR) | 1.893 jogos (11,4%), acima de 1,085 milhao |
| Maior venda | 82,74 milhoes (Wii Sports) |

A media é **3,2 vezes maior** que a mediana. Isso acontece porque a distribuicao é
**assimetrica a direita**: a maioria esmagadora dos jogos vende pouquissimo e um grupo
pequeno de "arrasa-quarteirao" vende muito, puxando a media para cima.

**O que aprendemos:** para esse tipo de dado, dizer que "um jogo vende em media 537 mil
copias" engana - metade dos jogos nao chega nem a 170 mil. A **mediana** descreve muito
melhor o jogo tipico do que a media. E um CV de 289% confirma que os dados sao
extremamente espalhados.

### Descoberta 2 - As vendas das regioes andam juntas (mas nao é causa)

Comparando as vendas na Europa (X) com as vendas na America do Norte (Y):

| Medida | Valor |
|---|---|
| Correlacao de Pearson (r) | 0,768 (forte e positiva) |
| Reta de regressao | ŷ = 1,2407·x + 0,0827 |
| R² | 0,589 |

O R² de 0,589 quer dizer que **58,9%** da variacao das vendas na America do Norte é
explicada pelas vendas na Europa. A inclinacao 1,24 quer dizer que, a cada 1 milhao de
copias vendidas na Europa, espera-se cerca de 1,24 milhao na America do Norte.

**O alerta honesto:** correlacao **nao** é causalidade. Vender bem na Europa nao *faz* o
jogo vender bem nos EUA. As duas coisas acontecem juntas porque existe uma terceira causa
por tras: o jogo ser bom e famoso mundialmente (Mario, GTA, FIFA vendem bem em todo lugar).

### Descoberta 3 - O Teorema Central do Limite funciona mesmo com dados tortos

O genero com mais jogos lancados é **Action**, com 3.316 jogos (19,98% do catalogo).

A descoberta que mais nos surpreendeu veio da simulacao do Modulo 3: a variavel
`Global_Sales` é super assimetrica (Descoberta 1), nada parecida com uma curva Normal.
Mesmo assim, quando sorteamos varias amostras dela e calculamos a **media de cada amostra**,
o histograma dessas medias fica praticamente igual a curva Normal - ja com amostras de
tamanho 30. Alem disso, o desvio padrao dessas medias bate com o valor teorico σ/√n.

**O que aprendemos:** é por isso que a curva Normal aparece em todo lugar na estatistica.
Nao é que os dados precisem ser normais - as **medias** deles é que ficam normais quando a
amostra é grande o suficiente. Da para ver isso acontecendo ao vivo, mexendo no controle de
tamanho da amostra na aplicacao.

---

## 6. Conclusao

O trabalho mostrou na pratica que as formulas da estatistica nao sao caixas-pretas: quando
a gente escreve a variancia linha por linha e o resultado bate com o NumPy na decima casa
decimal, fica claro o que cada pedaco da formula faz. E aplicar isso em dados reais mostrou
coisas que nao dava para ver so olhando a tabela - como o fato da media de vendas ser
enganosa, ou a curva Normal aparecer do nada quando trabalhamos com medias.
