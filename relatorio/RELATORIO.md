# Relatório — Laboratório de Estatística
### Dataset: Video Game Sales

## 1. Dataset

- **Fonte:** [Video Game Sales (Kaggle, gregorut)](https://www.kaggle.com/datasets/gregorut/videogamesales), resultado de uma raspagem do site VGChartz.
- **Tema:** jogos.
- **Registros:** 16.598 jogos.
- **Variáveis numéricas (6):** `Year`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales` (vendas em milhões de cópias).
- **Variáveis categóricas (3 relevantes):** `Platform`, `Genre`, `Publisher`.
- **Valores ausentes:** `Year` (271), `Publisher` (58) — removidos linha a linha apenas na análise da(s) variável(is) envolvida(s), nunca do dataset inteiro.

## 2. Arquitetura e decisões de projeto

```
minhastats/        núcleo estatístico próprio (Módulo 1), sem dependências de
                    estatística pronta — só math e a biblioteca padrão.
    descritiva.py   média, mediana, moda, amplitude, variância, desvio padrão,
                    percentil/quartis, coeficiente de variação.
    correlacao.py   covariância, correlação de Pearson, regressão linear (OLS).
    distribuicoes.py Normal, Binomial, Poisson, Uniforme, Exponencial (pdf/pmf).

app/                interface Streamlit (Módulos 0, 2–6). Usa pandas/numpy
                    apenas para carregar/filtrar dados e gerar números
                    aleatórios nas simulações — nunca para calcular a medida
                    exibida ao usuário, que vem sempre de `minhastats`.

tests/              pytest comparando cada função de `minhastats` com
                    NumPy/SciPy (tolerância 1e-9, ou 1e-6 em datasets maiores
                    por acúmulo de ponto flutuante).
```

- **Por que Streamlit?** interatividade simples (widgets ↔ estado) com pouco código, recomendado pelo próprio enunciado.
- **Percentil/quartis:** implementados por interpolação linear entre os dois pontos mais próximos (mesmo método usado por padrão em `numpy.percentile`), o que permite comparação direta e exata nos testes automatizados.
- **Variância/desvio/CV:** todas as funções aceitam `tipo="amostral"` (÷ n−1) ou `"populacional"` (÷ n); a interface deixa o usuário escolher.
- **Regressão linear:** mínimos quadrados ordinários derivados manualmente a partir das somas de desvios (b = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²; a = ȳ − b·x̄), sem `numpy.polyfit`/`scipy.stats.linregress`/`sklearn`.
- **Outliers:** regra do IQR (Q1 − 1,5·IQR e Q3 + 1,5·IQR), usando os quartis da própria biblioteca.
- **Distribuições teóricas:** parâmetros estimados pelo método dos momentos a partir de média/variância calculadas pela biblioteca (ex.: Exponencial usa λ = 1/x̄; Poisson usa λ = x̄).

## 3. Fórmulas implementadas

| Medida | Fórmula |
|---|---|
| Média | x̄ = (Σxᵢ) / n |
| Variância (amostral) | s² = Σ(xᵢ − x̄)² / (n − 1) |
| Variância (populacional) | σ² = Σ(xᵢ − x̄)² / n |
| Desvio padrão | s = √s² |
| Coeficiente de variação | CV = (s / \|x̄\|) × 100% |
| Percentil p | interpolação linear na posição (p/100)(n−1) da série ordenada |
| Covariância | cov(x,y) = Σ(xᵢ−x̄)(yᵢ−ȳ) / (n−1) |
| Correlação de Pearson | r = cov(x,y) / (sₓ·s_y) |
| Regressão linear | ŷ = a + b·x, com b = Sxy/Sxx e a = ȳ − b·x̄ |
| R² | 1 − SSres/SStot |
| Normal (pdf) | f(x) = 1/(σ√(2π)) · e^(−(x−μ)²/2σ²) |
| Binomial (pmf) | P(X=k) = C(n,k) pᵏ (1−p)ⁿ⁻ᵏ |
| Poisson (pmf) | P(X=k) = λᵏe^−λ / k! |
| Uniforme (pdf) | f(x) = 1/(b−a), a ≤ x ≤ b |
| Exponencial (pdf) | f(x) = λe^(−λx), x ≥ 0 |

## 4. As três descobertas

### Descoberta 1 — O mercado é dominado por poucos "blockbusters"
A média de vendas globais (**≈0,537 milhões de cópias**) é mais de **3× maior** que a mediana (**0,17 milhões**), com coeficiente de variação de **≈289%**. A regra do IQR classifica **1.893 jogos (≈11,4%)** como outliers de venda (acima de ≈1,085 milhão). Isto é: a maioria dos jogos vende pouco, e um pequeno grupo de sucessos (o maior é *Wii Sports*, com 82,74 milhões de cópias) puxa a média para cima — uma distribuição fortemente assimétrica à direita, mais próxima de uma cauda exponencial do que de uma Normal simétrica (Módulo 4).

### Descoberta 2 — Vendas em regiões diferentes crescem juntas, mas isso não é causalidade
Existe correlação forte e positiva entre vendas na Europa e na América do Norte (**r ≈ 0,768**; R² ≈ 0,589), ou seja, cerca de **58,9%** da variação das vendas norte-americanas de um jogo pode ser "explicada" linearmente pelas vendas europeias (b ≈ 1,24: a cada +1 milhão de cópias na Europa, espera-se +1,24 milhão na América do Norte). Isso **não prova causalidade** — o mais plausível é que ambas as variáveis reflitam uma causa comum, a popularidade/qualidade do próprio jogo (franquias como Mario, GTA ou FIFA vendem bem nos dois mercados pelo mesmo motivo, não porque um mercado "causa" o outro).

### Descoberta 3 — O Teorema Central do Limite funciona mesmo com uma variável tão assimétrica
`Global_Sales` é extremamente assimétrica (Descoberta 1) — está longe de ser Normal. Ainda assim, a simulação de Monte Carlo do Módulo 3 mostra que, ao sortear repetidamente amostras dessa variável (com reposição) e calcular a média de cada amostra, a distribuição dessas **médias amostrais** se aproxima visivelmente de uma curva Normal conforme o tamanho da amostra cresce (já perceptível com n ≈ 30), com desvio padrão convergindo para o erro padrão teórico σ/√n. Na prática: mesmo dados "torts" (como vendas, muito comuns em fenômenos de mercado) permitem inferência baseada em médias amostrais aproximadamente normais, desde que a amostra não seja pequena demais — a essência do TCL.

## 5. Como reproduzir

Ver instruções completas em [`README.md`](../README.md). Resumo:

```bash
pip install -r requirements.txt
pytest -v                     # valida minhastats contra NumPy/SciPy
streamlit run app/main.py     # abre a aplicação interativa
```
