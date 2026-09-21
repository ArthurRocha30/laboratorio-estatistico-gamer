# Relatorio - Laboratorio de Estatistica

**Aluno:** Arthur Morais de Carvalho Rocha - **Matricula:** 72650520
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

**Por que escolhi:** o tema jogos é do meu interesse e o dataset atende todos os
requisitos (mais de 1.000 registros, mais de 4 variaveis numericas e mais de 2 categoricas).

**Valores faltando:** 271 jogos estao sem o ano de lancamento e 58 estao sem a publicadora.
Decidimos **nao apagar essas linhas do dataset inteiro**: a gente tira os valores vazios
apenas da variavel que esta sendo analisada no momento (funcao `pegar_valores` no `app.py`).
Assim nao perdemos dados de vendas por causa de um ano faltando.

**Uma inconsistencia que encontramos:** a ultima linha do arquivo tem `Rank = 16600`,
mas o dataset tem 16.598 registros - ou seja, existem posicoes de ranking que nao
aparecem no arquivo. Isso nao afeta nenhuma analise nossa, porque em nenhum momento
usamos a coluna `Rank` (ela e so o numero da posicao, nao uma medida), mas achamos
importante registrar que conferimos.

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

## 5. Explicacao de cada modulo

### Modulo 0 - Dados Reais
Apresenta o dataset: quantidade de registros, dicionario explicando o que e cada
coluna e o levantamento de valores faltando (271 jogos sem ano, 58 sem
publicadora).

![Modulo 0](imagens/modulo0-dados.png)

### Modulo 1 - Nucleo estatistico
Nao e uma tela: e o arquivo `minhastats.py`, com todas as formulas implementadas
do zero, e o `test_minhastats.py`, com os 38 testes que validam cada funcao
contra o NumPy/SciPy. Detalhado nas secoes 3 e 4.

Os nomes dos testes mostram a cobertura: as duas versoes da variancia (amostral e
populacional), a correlacao comparada tanto com o NumPy quanto com o SciPy, e os
testes rodados sobre o dataset real.

![Testes passando](imagens/testes-pytest.png)

### Modulo 2 - Estatistica Descritiva Interativa

**Medidas e interpretacao** (variavel `Global_Sales`). A interpretacao detecta
sozinha a assimetria a direita, comparando media (0,5374) com mediana (0,1700), e
classifica a dispersao como alta pelo coeficiente de variacao de 289,34%. A
caixinha no topo alterna entre variancia amostral e populacional.

![Modulo 2 - medidas](imagens/modulo2-medidas.png)

**Tabela de frequencias em classes** (variavel `Year`, 10 classes). A frequencia
acumulada fecha em 16.327, o total de jogos com ano preenchido.

![Modulo 2 - frequencias](imagens/modulo2-frequencias.png)

**Graficos para variavel numerica** (`Year`): histograma e boxplot. O boxplot
mostra a mediana em torno de 2007, a caixa entre 2003 e 2010, e marca os jogos
dos anos 80 como outliers.

![Modulo 2 - graficos](imagens/modulo2-graficos.png)

**Graficos para variavel categorica** (`Genre`): barras e pizza, com a
interpretacao identificando a moda (Action, 3.316 jogos, 19,98%).

![Modulo 2 - categorica](imagens/modulo2-categorica.png)

**Deteccao de outliers** pela regra do IQR, aplicada a `Global_Sales`. A tela
mostra a conta inteira: IQR = 0,4700 - 0,0600 = 0,4100, limites em -0,5550 e
1,0850, resultando em 1.893 outliers (11,4% dos dados).

Note que o limite inferior deu negativo. Como nenhum jogo vende um numero
negativo de copias, **nao existe nenhum outlier inferior** - todos os 1.893 estao
na ponta de cima, consequencia direta da assimetria a direita.

![Modulo 2 - outliers](imagens/modulo2-outliers.png)

### Modulo 3 - Probabilidade e Simulacao
Dois experimentos de Monte Carlo, ambos com parametros controlaveis pelo usuario
(numero de repeticoes, tamanho da amostra e semente aleatoria).

**(a) Lei dos Grandes Numeros** - simula lancamentos de moeda ou dado e acompanha
a frequencia relativa acumulada. Na imagem, com 2.000 lancamentos, a frequencia
observada foi 0,4835 contra 0,5000 de probabilidade teorica - diferenca de
0,0165. O formato da curva e o ponto principal: ela oscila muito no comeco (vai a
1,0 e depois a 0,66 nos primeiros lancamentos) e vai se estabilizando conforme o
numero cresce. E exatamente o que a Lei dos Grandes Numeros descreve.

![Modulo 3 - Lei dos Grandes Numeros](imagens/modulo3-lgn.png)

**(b) Teorema Central do Limite** - sorteia amostras repetidas de uma variavel do
dataset e calcula a media de cada uma usando o `minhastats.media`.

Na imagem, 200 amostras de tamanho 30 da variavel `Global_Sales`. O efeito e
visivel: a variavel original vai de 0,01 a 82,74 milhoes, e as medias amostrais ja
se concentram quase todas entre 0,2 e 1,0 - uma reducao drastica na dispersao.

Ainda resta uma assimetria a direita perceptivel, e isso tem explicacao: a
populacao e **extremamente** assimetrica (media 0,5374 contra mediana 0,1700),
entao com n = 30 o Teorema ja atuou bastante, mas nao o suficiente para produzir
uma Normal perfeita. Aumentando o tamanho da amostra na aplicacao, a distribuicao
das medias fica progressivamente mais simetrica - que e justamente o que o
Teorema preve.

![Modulo 3 - TCL](imagens/modulo3-tcl.png)

### Modulo 4 - Distribuicoes Teoricas
Sobrepoe ao histograma a curva **Normal** (sempre) mais uma segunda distribuicao
escolhida pelo usuario (**Exponencial** ou **Uniforme**), com os parametros
estimados dos proprios dados pelo metodo dos momentos.

Na imagem, a variavel `Year` comparada com a Normal (media 2006,41 e desvio 5,83)
e com a Uniforme (que vale 1/40 = 0,025, porque sao 40 anos de intervalo).

**Discussao do ajuste:** a Normal acompanha o formato geral - sobe, tem pico por
volta de 2006 e desce -, mas erra em pontos especificos: subestima o pico de
2008-2009 (a barra chega a 0,13 e a curva so a 0,068) e superestima a faixa de
1990 a 1998. Ja a Uniforme, sendo uma reta horizontal, claramente nao descreve
dados com um pico tao marcado. Conclusao: a Normal e um ajuste razoavel mas
imperfeito, e a Uniforme nao serve.

A Binomial e a Poisson tambem estao implementadas e testadas no `minhastats.py`,
mas nao aparecem nesta tela porque sao distribuicoes discretas (para contagens), e
as variaveis deste dataset sao continuas.

![Modulo 4](imagens/modulo4-distribuicoes.png)

### Modulo 5 - Correlacao e Regressao Linear
O usuario escolhe duas variaveis numericas e recebe a correlacao de Pearson, o
R², a covariancia, o diagrama de dispersao com a reta de minimos quadrados, a
equacao da reta e um campo de predicao interativa - alem da interpretacao dos
coeficientes e de um alerta fixo de que correlacao nao implica causalidade.

Na imagem, `EU_Sales` (X) contra `NA_Sales` (Y), sobre os 16.598 pares:
correlacao de 0,7677, R² de 0,5894 e covariancia de 0,3168.

![Modulo 5 - medidas](imagens/modulo5-medidas.png)

O diagrama de dispersao com a reta ajustada. A nuvem de pontos concentrada perto
da origem e mais uma vista da assimetria discutida na Descoberta 1.

![Modulo 5 - regressao](imagens/modulo5-regressao.png)

### Modulo 6 - Relatorio de Descobertas
Mostra as 3 descobertas da secao 6 com os numeros **recalculados ao vivo** pela
biblioteca a cada carregamento da pagina - nao sao valores digitados no texto,
sao o resultado das funcoes rodando sobre o dataset.

![Modulo 6](imagens/modulo6-descobertas.png)

---

## 6. As 3 descobertas

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
| Reta de regressao | y previsto = 1,2407 * x + 0,0827 |
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

## 7. Conclusao

O trabalho mostrou na pratica que as formulas da estatistica nao sao caixas-pretas: quando
a gente escreve a variancia linha por linha e o resultado bate com o NumPy na decima casa
decimal, fica claro o que cada pedaco da formula faz. E aplicar isso em dados reais mostrou
coisas que nao dava para ver so olhando a tabela - como o fato da media de vendas ser
enganosa, ou a curva Normal aparecer do nada quando trabalhamos com medias.

Tres licoes que ficaram deste laboratorio:

1. **Uma medida sozinha mente.** A media de vendas (0,537 milhao) parece dizer alguma
   coisa, mas comparada com a mediana (0,170) revela que a distribuicao e torta.
   So olhando media, mediana, quartis e coeficiente de variacao **juntos** e que da
   para entender o formato dos dados.

2. **Correlacao alta nao explica o porque.** O r de 0,768 entre vendas na Europa e nos
   EUA e forte, mas a explicacao ("o jogo ser famoso") e uma terceira variavel que nao
   esta em nenhuma coluna do dataset. O numero mostra O QUE acontece, nao POR QUE.

3. **Testar mudou nossa confianca no codigo.** Foram os testes que nos obrigaram a
   entender detalhes que a gente teria passado batido - por exemplo, que existe mais de
   um metodo para calcular percentil, e que precisavamos escolher o mesmo do NumPy para
   os resultados baterem.

**Conferencia final:** todos os numeros da secao 6 foram recalculados pela propria
aplicacao (Modulo 6, que refaz as contas ao vivo) e conferem com o que esta escrito aqui.
