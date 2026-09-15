# 🎮 Laboratório de Estatística — Video Game Sales

Aplicação interativa em Streamlit para explorar estatística descritiva,
probabilidade, distribuições, correlação e regressão linear sobre um
dataset real de vendas de jogos, usando uma **biblioteca estatística
própria** (`minhastats`), implementada do zero e validada por testes
automatizados contra NumPy/SciPy.

## Sumário

- [Dataset](#dataset)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como rodar do zero](#como-rodar-do-zero)
- [Rodando os testes](#rodando-os-testes)
- [Módulos da aplicação](#módulos-da-aplicação)
- [Regra de ouro](#regra-de-ouro)

## Dataset

**[Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales)**
— 16.598 jogos com vendas por região, plataforma, gênero e publicadora
(originado de uma raspagem do site VGChartz). O arquivo já está incluído em
[`data/vgsales.csv`](data/vgsales.csv), então não é necessário baixar nada
separadamente.

- Variáveis numéricas: `Year`, `NA_Sales`, `EU_Sales`, `JP_Sales`,
  `Other_Sales`, `Global_Sales`.
- Variáveis categóricas: `Platform`, `Genre`, `Publisher`.

## Estrutura do repositório

```
projeto-matematica/
├── data/
│   └── vgsales.csv            # dataset real (Módulo 0)
├── minhastats/                # Módulo 1: núcleo estatístico próprio
│   ├── descritiva.py          #   média, mediana, moda, variância, quartis...
│   ├── correlacao.py          #   covariância, Pearson, regressão linear
│   └── distribuicoes.py       #   Normal, Binomial, Poisson, Uniforme, Exponencial
├── app/                       # aplicação Streamlit (Módulos 0, 2–6)
│   ├── main.py                #   ponto de entrada / navegação
│   ├── utils.py                #   carregamento de dados, tabelas de frequência, outliers
│   ├── modulo0_dados.py
│   ├── modulo2_descritiva.py
│   ├── modulo3_simulacao.py
│   ├── modulo4_distribuicoes.py
│   ├── modulo5_regressao.py
│   └── modulo6_relatorio.py
├── tests/
│   └── test_minhastats.py     # pytest comparando minhastats com NumPy/SciPy
├── relatorio/
│   └── RELATORIO.md           # fórmulas, decisões de projeto e as 3 descobertas
├── requirements.txt
├── pytest.ini
└── README.md
```

## Como rodar do zero

Pré-requisito: **Python 3.10+** instalado.

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd projeto-matematica

# 2. Crie e ative um ambiente virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Rode a aplicação
streamlit run app/main.py
```

A aplicação abre em `http://localhost:8501`. Use o menu lateral para navegar
entre os módulos.

## Rodando os testes

```bash
pytest -v
```

Isso executa `tests/test_minhastats.py`, que compara cada função de
`minhastats` (média, mediana, moda, variância, desvio padrão, percentis,
covariância, correlação de Pearson, regressão linear e as distribuições
teóricas) com o resultado de referência do **NumPy**/**SciPy**, incluindo
casos com o dataset real do laboratório.

## Módulos da aplicação

| Módulo | Conteúdo |
|---|---|
| 0 — Dados Reais | Apresentação do dataset, dicionário de variáveis, valores ausentes |
| 1 — Núcleo Estatístico | Biblioteca `minhastats/` + testes em `tests/` (não é uma página, é a base de tudo) |
| 2 — Estatística Descritiva | Tabela de frequências, medidas, histograma/boxplot/barras/pizza, outliers (IQR), interpretação automática |
| 3 — Probabilidade e Simulação | Monte Carlo: Lei dos Grandes Números (moeda/dado) e Teorema Central do Limite (médias amostrais do dataset) |
| 4 — Distribuições Teóricas | Sobreposição de Normal + Binomial/Poisson/Uniforme/Exponencial ao histograma, parâmetros estimados dos dados |
| 5 — Correlação e Regressão | Dispersão, correlação de Pearson, reta de mínimos quadrados, R², predição interativa |
| 6 — Relatório de Descobertas | As 3 descobertas estatísticas, recalculadas em tempo real (texto completo em `relatorio/RELATORIO.md`) |

## Regra de ouro

Pandas e NumPy são usados **apenas** para carregar/filtrar o CSV e gerar
números aleatórios nas simulações de Monte Carlo. **Toda medida estatística
exibida ao usuário** (média, mediana, moda, variância, desvio padrão,
quartis, coeficiente de variação, covariância, correlação, coeficientes de
regressão, densidades de distribuições) vem das funções implementadas em
`minhastats/`, validadas em `tests/test_minhastats.py`.
