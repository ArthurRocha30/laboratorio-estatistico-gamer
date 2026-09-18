"""
app.py
======

Aplicacao do nosso Laboratorio de Estatistica (Modulos 0, 2, 3, 4, 5 e 6).

Regra de ouro do trabalho: o pandas so é usado para LER e FILTRAR o arquivo
CSV. Toda medida estatistica que aparece na tela é calculada pelas funcoes
que nós escrevemos em minhastats.py.

Para rodar:  streamlit run app.py
"""

import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import minhastats as ms

# ----------------------------------------------------------
# CONFIGURACAO DA PAGINA
# ----------------------------------------------------------
st.set_page_config(page_title="Laboratorio de Estatistica", page_icon="🎮", layout="wide")

# caminho do CSV. E um caminho relativo, entao o streamlit precisa ser
# aberto de dentro da pasta do projeto (é o que o README manda fazer).
CAMINHO_DO_CSV = "data/vgsales.csv"

COLUNAS_NUMERICAS = ["Year", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
COLUNAS_CATEGORICAS = ["Platform", "Genre", "Publisher"]


def carregar_dados():
    return pd.read_csv(CAMINHO_DO_CSV)


def pegar_valores(tabela, coluna):
    """Pega uma coluna do CSV e devolve uma lista simples, sem valores vazios."""
    return tabela[coluna].dropna().tolist()


# ----------------------------------------------------------
# FUNCOES AUXILIARES (tabelas de frequencia e outliers)
# ----------------------------------------------------------

def tabela_de_frequencias_por_classes(valores, numero_de_classes):
    """
    Monta a tabela de frequencias de uma variavel numerica, separando
    os valores em classes (intervalos) de mesmo tamanho.

    tamanho de cada classe = amplitude / numero de classes
    """
    menor = min(valores)
    maior = max(valores)
    tamanho_da_classe = ms.amplitude(valores) / numero_de_classes

    # conta quantos valores caem em cada classe
    contagens = [0] * numero_de_classes
    for valor in valores:
        posicao = int((valor - menor) / tamanho_da_classe)
        if posicao >= numero_de_classes:   # o maior valor de todos cai na ultima classe
            posicao = numero_de_classes - 1
        contagens[posicao] = contagens[posicao] + 1

    # monta as linhas da tabela
    linhas = []
    acumulada = 0
    for i in range(numero_de_classes):
        inicio = menor + i * tamanho_da_classe
        fim = inicio + tamanho_da_classe
        acumulada = acumulada + contagens[i]
        linhas.append({
            "Classe": f"{inicio:.2f} a {fim:.2f}",
            "Frequencia": contagens[i],
            "Frequencia relativa (%)": round(contagens[i] / len(valores) * 100, 2),
            "Frequencia acumulada": acumulada,
        })

    return pd.DataFrame(linhas)


def tabela_de_frequencias_categorica(valores):
    """Conta quantas vezes cada categoria aparece (ex.: quantos jogos de Acao)."""
    # 1) conta quantas vezes cada categoria aparece
    contagem = {}
    for valor in valores:
        if valor in contagem:
            contagem[valor] = contagem[valor] + 1
        else:
            contagem[valor] = 1

    # 2) monta uma lista de pares (frequencia, categoria) para poder ordenar.
    #    o sorted() com reverse=True ordena pelo primeiro item do par, ou seja,
    #    da maior frequencia para a menor.
    pares = []
    for categoria in contagem:
        pares.append((contagem[categoria], categoria))
    pares = sorted(pares, reverse=True)

    # 3) monta as linhas da tabela ja na ordem certa
    linhas = []
    for par in pares:
        frequencia = par[0]
        categoria = par[1]
        linhas.append({
            "Categoria": categoria,
            "Frequencia": frequencia,
            "Frequencia relativa (%)": round(frequencia / len(valores) * 100, 2),
        })

    return pd.DataFrame(linhas)


def encontrar_outliers(valores):
    """
    Procura outliers pela REGRA DO IQR.

    IQR = Q3 - Q1  (intervalo entre o primeiro e o terceiro quartil)
    limite de baixo = Q1 - 1.5 * IQR
    limite de cima  = Q3 + 1.5 * IQR

    Tudo que ficar fora desses limites é considerado outlier.
    """
    q1, q2, q3 = ms.quartis(valores)
    iqr = q3 - q1
    limite_de_baixo = q1 - 1.5 * iqr
    limite_de_cima = q3 + 1.5 * iqr

    outliers = []
    for valor in valores:
        if valor < limite_de_baixo or valor > limite_de_cima:
            outliers.append(valor)

    return outliers, limite_de_baixo, limite_de_cima


# ==========================================================
# MODULO 0 - DADOS REAIS
# ==========================================================

def modulo_0_dados():
    st.header("Modulo 0 - Dados Reais")

    tabela = carregar_dados()

    st.markdown(
        """
        **Dataset:** [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) (Kaggle)

        Sao vendas de jogos de video game no mundo todo, coletadas do site VGChartz.
        Escolhemos esse dataset porque o tema (jogos) é do interesse do grupo e ele
        tem bastante variavel numerica e categorica para analisar.
        """
    )

    coluna1, coluna2, coluna3 = st.columns(3)
    coluna1.metric("Registros (jogos)", len(tabela))
    coluna2.metric("Variaveis numericas", len(COLUNAS_NUMERICAS))
    coluna3.metric("Variaveis categoricas", len(COLUNAS_CATEGORICAS))

    st.subheader("Primeiras linhas do dataset")
    st.dataframe(tabela.head(10))

    st.subheader("O que significa cada coluna")
    st.table(pd.DataFrame([
        {"Coluna": "Name", "Tipo": "texto", "Significado": "Nome do jogo"},
        {"Coluna": "Platform", "Tipo": "categorica", "Significado": "Console/plataforma (Wii, PS2, X360...)"},
        {"Coluna": "Year", "Tipo": "numerica", "Significado": "Ano de lancamento"},
        {"Coluna": "Genre", "Tipo": "categorica", "Significado": "Genero do jogo (Acao, Esporte, RPG...)"},
        {"Coluna": "Publisher", "Tipo": "categorica", "Significado": "Empresa que publicou o jogo"},
        {"Coluna": "NA_Sales", "Tipo": "numerica", "Significado": "Vendas na America do Norte (milhoes de copias)"},
        {"Coluna": "EU_Sales", "Tipo": "numerica", "Significado": "Vendas na Europa (milhoes de copias)"},
        {"Coluna": "JP_Sales", "Tipo": "numerica", "Significado": "Vendas no Japao (milhoes de copias)"},
        {"Coluna": "Other_Sales", "Tipo": "numerica", "Significado": "Vendas no resto do mundo (milhoes de copias)"},
        {"Coluna": "Global_Sales", "Tipo": "numerica", "Significado": "Vendas totais no mundo (milhoes de copias)"},
    ]))

    st.subheader("Valores faltando")
    faltando = tabela.isna().sum()
    st.write(f"- **Year:** {faltando['Year']} jogos sem o ano de lancamento")
    st.write(f"- **Publisher:** {faltando['Publisher']} jogos sem a empresa publicadora")
    st.info(
        "Nas outras paginas nós sempre tiramos os valores vazios da variavel escolhida "
        "antes de calcular qualquer coisa (é o que a funcao `pegar_valores` faz)."
    )


# ==========================================================
# MODULO 2 - ESTATISTICA DESCRITIVA INTERATIVA
# ==========================================================

def modulo_2_descritiva():
    st.header("Modulo 2 - Estatistica Descritiva")
    st.caption("Todas as medidas desta pagina sao calculadas pelo nosso minhastats.py")

    tabela = carregar_dados()
    tipo = st.radio("Que tipo de variavel voce quer analisar?", ["Numerica", "Categorica"], horizontal=True)

    # ---------- VARIAVEL NUMERICA ----------
    if tipo == "Numerica":
        coluna = st.selectbox("Escolha a variavel:", COLUNAS_NUMERICAS, index=5)
        valores = pegar_valores(tabela, coluna)
        st.caption(f"{len(valores)} valores (tirando os vazios)")

        usar_populacional = st.checkbox(
            "Calcular variancia/desvio/CV como POPULACAO (dividir por n em vez de n-1)"
        )

        # --- as medidas, todas vindas do nosso arquivo minhastats.py ---
        m = ms.media(valores)
        med = ms.mediana(valores)
        mo = ms.moda(valores)
        amp = ms.amplitude(valores)
        var = ms.variancia(valores, usar_populacional)
        dp = ms.desvio_padrao(valores, usar_populacional)
        cv = ms.coeficiente_variacao(valores, usar_populacional)
        q1, q2, q3 = ms.quartis(valores)

        st.subheader("Medidas de tendencia central")
        c1, c2, c3 = st.columns(3)
        c1.metric("Media", f"{m:.4f}")
        c2.metric("Mediana", f"{med:.4f}")
        if len(mo) == 0:
            c3.metric("Moda", "nao tem")
        else:
            c3.metric("Moda", f"{mo[0]:g}")

        st.subheader("Medidas de dispersao")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Amplitude", f"{amp:.4f}")
        c2.metric("Variancia", f"{var:.4f}")
        c3.metric("Desvio padrao", f"{dp:.4f}")
        c4.metric("Coef. de variacao", f"{cv:.2f}%")

        st.subheader("Quartis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Q1 (25%)", f"{q1:.4f}")
        c2.metric("Q2 (50%)", f"{q2:.4f}")
        c3.metric("Q3 (75%)", f"{q3:.4f}")

        # --- interpretacao automatica ---
        st.subheader("Interpretacao automatica")

        if abs(m - med) < 0.05 * dp:
            st.info("A media e a mediana estao bem proximas: a distribuicao é mais ou menos SIMETRICA.")
        elif m > med:
            st.info(
                "A media é MAIOR que a mediana: a distribuicao tem ASSIMETRIA A DIREITA. "
                "Isso acontece quando existem alguns valores muito altos que puxam a media para cima."
            )
        else:
            st.info(
                "A media é MENOR que a mediana: a distribuicao tem ASSIMETRIA A ESQUERDA. "
                "Isso acontece quando existem alguns valores muito baixos que puxam a media para baixo."
            )

        if cv < 15:
            st.info(f"O coeficiente de variacao é {cv:.2f}% (menor que 15%): dispersao BAIXA, dados parecidos entre si.")
        elif cv < 30:
            st.info(f"O coeficiente de variacao é {cv:.2f}% (entre 15% e 30%): dispersao MEDIA.")
        else:
            st.info(f"O coeficiente de variacao é {cv:.2f}% (maior que 30%): dispersao ALTA, dados muito diferentes entre si.")

        # --- tabela de frequencias ---
        st.subheader("Tabela de frequencias (por classes)")
        numero_de_classes = st.slider("Quantas classes?", 5, 20, 10)
        st.dataframe(tabela_de_frequencias_por_classes(valores, numero_de_classes))

        # --- graficos ---
        st.subheader("Graficos")
        grafico1, grafico2 = st.columns(2)

        with grafico1:
            figura, eixo = plt.subplots()
            eixo.hist(valores, bins=numero_de_classes, color="steelblue", edgecolor="white")
            eixo.set_title(f"Histograma - {coluna}")
            eixo.set_xlabel(coluna)
            eixo.set_ylabel("Frequencia")
            st.pyplot(figura)

        with grafico2:
            figura, eixo = plt.subplots()
            eixo.boxplot(valores)
            eixo.set_title(f"Boxplot - {coluna}")
            eixo.set_ylabel(coluna)
            st.pyplot(figura)

        # --- outliers ---
        st.subheader("Outliers (regra do IQR)")
        outliers, limite_de_baixo, limite_de_cima = encontrar_outliers(valores)
        st.write(f"IQR = Q3 - Q1 = {q3:.4f} - {q1:.4f} = **{q3 - q1:.4f}**")
        st.write(f"Limite de baixo = Q1 - 1.5 x IQR = **{limite_de_baixo:.4f}**")
        st.write(f"Limite de cima = Q3 + 1.5 x IQR = **{limite_de_cima:.4f}**")

        if len(outliers) == 0:
            st.success("Nenhum outlier encontrado.")
        else:
            porcentagem = len(outliers) / len(valores) * 100
            st.warning(f"Foram encontrados **{len(outliers)} outliers** ({porcentagem:.1f}% dos dados).")
            st.write(f"Maior outlier: **{max(outliers):.2f}**")

    # ---------- VARIAVEL CATEGORICA ----------
    else:
        coluna = st.selectbox("Escolha a variavel:", COLUNAS_CATEGORICAS)
        valores = pegar_valores(tabela, coluna)
        st.caption(f"{len(valores)} valores (tirando os vazios)")

        frequencias = tabela_de_frequencias_categorica(valores)

        st.subheader("Tabela de frequencias")
        st.dataframe(frequencias)

        quantas_mostrar = st.slider("Quantas categorias mostrar nos graficos?", 3, 15, 8)
        top = frequencias.head(quantas_mostrar)

        st.subheader("Graficos")
        grafico1, grafico2 = st.columns(2)

        with grafico1:
            figura, eixo = plt.subplots()
            eixo.bar(top["Categoria"], top["Frequencia"], color="steelblue")
            eixo.set_title(f"Grafico de barras - {coluna}")
            eixo.set_ylabel("Quantidade de jogos")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(figura)

        with grafico2:
            figura, eixo = plt.subplots()
            eixo.pie(top["Frequencia"], labels=top["Categoria"], autopct="%1.1f%%")
            eixo.set_title(f"Grafico de pizza - {coluna} (top {quantas_mostrar})")
            st.pyplot(figura)

        st.subheader("Interpretacao automatica")
        st.info(
            f"A categoria que mais aparece (a MODA) é **{frequencias['Categoria'][0]}**, "
            f"com {frequencias['Frequencia'][0]} jogos "
            f"({frequencias['Frequencia relativa (%)'][0]}% do total)."
        )


# ==========================================================
# MODULO 3 - PROBABILIDADE E SIMULACAO (MONTE CARLO)
# ==========================================================

def modulo_3_simulacao():
    st.header("Modulo 3 - Probabilidade e Simulacao")

    experimento = st.radio(
        "Escolha o experimento:",
        ["(a) Lei dos Grandes Numeros", "(b) Teorema Central do Limite"],
    )

    # ---------- LEI DOS GRANDES NUMEROS ----------
    if experimento == "(a) Lei dos Grandes Numeros":
        st.subheader("(a) Lei dos Grandes Numeros")
        st.markdown(
            """
            A Lei dos Grandes Numeros diz que, quanto mais vezes repetimos um
            experimento, mais a **frequencia relativa** (o que aconteceu de verdade)
            se aproxima da **probabilidade teorica** (o que deveria acontecer).

            Aqui nós simulamos lancamentos de moeda ou de dado e vamos acompanhando
            essa frequencia lancamento a lancamento.
            """
        )

        tipo = st.selectbox("O que vamos lancar?", ["Moeda (contar caras)", "Dado (contar o numero 6)"])
        quantidade = st.slider("Quantos lancamentos?", 100, 20000, 2000, step=100)
        semente = st.number_input("Semente aleatoria (mesma semente = mesmo resultado)", value=42, step=1)

        random.seed(int(semente))

        if tipo == "Moeda (contar caras)":
            probabilidade_teorica = 1 / 2
            nome_do_evento = "cara"
        else:
            probabilidade_teorica = 1 / 6
            nome_do_evento = "numero 6"

        # simulacao: a cada lancamento guardamos a frequencia relativa ate ali
        sucessos = 0
        frequencias_relativas = []
        for lancamento in range(1, quantidade + 1):
            if tipo == "Moeda (contar caras)":
                resultado = random.randint(1, 2)     # 1 = cara, 2 = coroa
                deu_certo = (resultado == 1)
            else:
                resultado = random.randint(1, 6)     # dado de 6 lados
                deu_certo = (resultado == 6)

            if deu_certo:
                sucessos = sucessos + 1

            frequencias_relativas.append(sucessos / lancamento)

        figura, eixo = plt.subplots(figsize=(8, 4))
        eixo.plot(frequencias_relativas, color="steelblue", label="frequencia relativa")
        eixo.axhline(probabilidade_teorica, color="red", linestyle="--", label="probabilidade teorica")
        eixo.set_title(f"Frequencia relativa de sair {nome_do_evento}")
        eixo.set_xlabel("Numero de lancamentos")
        eixo.set_ylabel("Frequencia relativa")
        eixo.legend()
        st.pyplot(figura)

        frequencia_final = frequencias_relativas[-1]
        diferenca = abs(frequencia_final - probabilidade_teorica)

        c1, c2, c3 = st.columns(3)
        c1.metric("Frequencia obtida", f"{frequencia_final:.4f}")
        c2.metric("Probabilidade teorica", f"{probabilidade_teorica:.4f}")
        c3.metric("Diferenca", f"{diferenca:.4f}")

        st.info(
            "Repare no grafico: no comeco a linha azul fica pulando bastante, mas conforme "
            "o numero de lancamentos aumenta ela vai 'grudando' na linha vermelha. "
            "Experimente aumentar o numero de lancamentos e veja a diferenca diminuir."
        )

    # ---------- TEOREMA CENTRAL DO LIMITE ----------
    else:
        st.subheader("(b) Teorema Central do Limite")
        st.markdown(
            """
            O Teorema Central do Limite diz que, se sortearmos varias amostras de uma
            variavel e calcularmos a **media de cada amostra**, essas medias formam uma
            distribuicao parecida com a **curva Normal** - mesmo que a variavel original
            nao tenha nada de Normal.

            Aqui sorteamos amostras de uma variavel do nosso dataset de jogos.
            """
        )

        tabela = carregar_dados()
        coluna = st.selectbox("Qual variavel vamos sortear?", COLUNAS_NUMERICAS, index=5)
        populacao = pegar_valores(tabela, coluna)

        c1, c2 = st.columns(2)
        numero_de_amostras = c1.slider("Quantas amostras sortear?", 100, 3000, 1000, step=100)
        tamanho_da_amostra = c2.slider("Tamanho de cada amostra (n)", 2, 200, 30)
        semente = st.number_input("Semente aleatoria", value=7, step=1)

        random.seed(int(semente))

        # sorteia varias amostras e calcula a media de cada uma
        medias_das_amostras = []
        for i in range(numero_de_amostras):
            amostra = random.choices(populacao, k=tamanho_da_amostra)  # sorteio com reposicao
            medias_das_amostras.append(ms.media(amostra))

        # valores teoricos previstos pelo TCL
        media_da_populacao = ms.media(populacao)
        desvio_da_populacao = ms.desvio_padrao(populacao, populacional=True)
        erro_padrao = desvio_da_populacao / (tamanho_da_amostra ** 0.5)   # desvio previsto das medias

        # grafico: histograma das medias + curva Normal teorica por cima
        figura, eixo = plt.subplots(figsize=(8, 4))
        eixo.hist(medias_das_amostras, bins=30, density=True, color="steelblue",
                  edgecolor="white", label="medias das amostras")

        menor = min(medias_das_amostras)
        maior = max(medias_das_amostras)
        pontos_x = []
        pontos_y = []
        for i in range(200):
            x = menor + (maior - menor) * i / 199
            pontos_x.append(x)
            pontos_y.append(ms.densidade_normal(x, media_da_populacao, erro_padrao))
        eixo.plot(pontos_x, pontos_y, color="red", label="curva Normal teorica")

        eixo.set_title(f"Medias de {numero_de_amostras} amostras de tamanho {tamanho_da_amostra}")
        eixo.set_xlabel(f"Media amostral de {coluna}")
        eixo.set_ylabel("Densidade")
        eixo.legend()
        st.pyplot(figura)

        c1, c2 = st.columns(2)
        c1.metric("Media da populacao", f"{media_da_populacao:.4f}")
        c1.metric("Media das medias amostrais", f"{ms.media(medias_das_amostras):.4f}")
        c2.metric("Erro padrao teorico (desvio / raiz de n)", f"{erro_padrao:.4f}")
        c2.metric("Desvio das medias (simulado)", f"{ms.desvio_padrao(medias_das_amostras):.4f}")

        st.info(
            "Compare os dois valores de cada lado: a media das medias fica quase igual a "
            "media da populacao, e o desvio das medias fica quase igual ao erro padrao teorico. "
            "Agora aumente o tamanho da amostra (n) e veja o histograma ficar cada vez mais "
            "parecido com a curva vermelha - isso é o Teorema Central do Limite funcionando, "
            "mesmo a variavel original sendo bem torta."
        )


# ==========================================================
# MODULO 4 - DISTRIBUICOES TEORICAS
# ==========================================================

def modulo_4_distribuicoes():
    st.header("Modulo 4 - Distribuicoes Teoricas")
    st.markdown(
        "Aqui colocamos a curva de uma distribuicao teorica por cima do histograma "
        "dos dados reais, para ver se ela 'encaixa'. Os parametros das curvas sao "
        "estimados a partir dos proprios dados."
    )

    tabela = carregar_dados()
    coluna = st.selectbox("Escolha a variavel:", COLUNAS_NUMERICAS, index=5)
    valores = pegar_valores(tabela, coluna)

    segunda_curva = st.selectbox(
        "Alem da Normal, qual outra distribuicao voce quer comparar?",
        ["Exponencial", "Uniforme"],
    )
    numero_de_classes = st.slider("Quantas classes no histograma?", 10, 60, 30)

    # parametros estimados a partir dos dados (usando nossas funcoes)
    m = ms.media(valores)
    dp = ms.desvio_padrao(valores, populacional=True)
    menor = min(valores)
    maior = max(valores)

    st.write(f"Parametros estimados dos dados: **media = {m:.4f}** e **desvio padrao = {dp:.4f}**")

    # monta os pontos das curvas
    pontos_x = []
    for i in range(300):
        pontos_x.append(menor + (maior - menor) * i / 299)

    curva_normal = []
    for x in pontos_x:
        curva_normal.append(ms.densidade_normal(x, m, dp))

    curva_segunda = []
    if segunda_curva == "Exponencial":
        taxa = 1 / m          # na Exponencial, a media vale 1/taxa -> taxa = 1/media
        for x in pontos_x:
            curva_segunda.append(ms.densidade_exponencial(x, taxa))
        nome_da_segunda = f"Exponencial (taxa = {taxa:.4f})"
    else:
        for x in pontos_x:
            curva_segunda.append(ms.densidade_uniforme(x, menor, maior))
        nome_da_segunda = f"Uniforme (de {menor:.2f} a {maior:.2f})"

    figura, eixo = plt.subplots(figsize=(9, 5))
    eixo.hist(valores, bins=numero_de_classes, density=True, color="steelblue",
              edgecolor="white", label="dados reais")
    eixo.plot(pontos_x, curva_normal, color="red", label=f"Normal (media={m:.2f}, dp={dp:.2f})")
    eixo.plot(pontos_x, curva_segunda, color="green", label=nome_da_segunda)
    eixo.set_title(f"Distribuicoes teoricas sobre o histograma - {coluna}")
    eixo.set_xlabel(coluna)
    eixo.set_ylabel("Densidade")
    eixo.legend()
    st.pyplot(figura)

    # --- discussao do ajuste ---
    st.subheader("A curva encaixou bem?")
    med = ms.mediana(valores)

    if abs(m - med) < 0.05 * dp:
        st.info(
            "Como a media e a mediana sao parecidas, os dados sao quase simetricos e a "
            "curva NORMAL (vermelha) acompanha bem o formato do histograma."
        )
    else:
        st.info(
            f"A media ({m:.4f}) é bem diferente da mediana ({med:.4f}), ou seja, os dados sao "
            "assimetricos. Por isso a curva NORMAL (vermelha) nao encaixa bem: ela é simetrica "
            "e acaba prevendo valores negativos, que nao existem aqui. Ja a EXPONENCIAL (verde) "
            "acompanha melhor, porque ela foi feita justamente para dados que se concentram "
            "perto do zero e tem poucos valores muito grandes."
        )

    st.caption(
        "Observacao: a Binomial e a Poisson tambem estao implementadas e testadas no nosso "
        "minhastats.py, mas nao aparecem aqui porque elas sao distribuicoes DISCRETAS "
        "(servem para contagens, tipo 'quantas caras em 10 jogadas'), e as variaveis deste "
        "dataset sao continuas."
    )


# ==========================================================
# MODULO 5 - CORRELACAO E REGRESSAO LINEAR
# ==========================================================

def modulo_5_regressao():
    st.header("Modulo 5 - Correlacao e Regressao Linear")

    tabela = carregar_dados()

    c1, c2 = st.columns(2)
    coluna_x = c1.selectbox("Variavel X (a que explica):", COLUNAS_NUMERICAS, index=2)
    coluna_y = c2.selectbox("Variavel Y (a que queremos prever):", COLUNAS_NUMERICAS, index=1)

    if coluna_x == coluna_y:
        st.warning("Escolha duas variaveis diferentes.")
        return

    # tira as linhas em que falta X ou Y
    duas_colunas = tabela[[coluna_x, coluna_y]].dropna()
    x = duas_colunas[coluna_x].tolist()
    y = duas_colunas[coluna_y].tolist()
    st.caption(f"{len(x)} pares de valores")

    # calculos feitos pelas nossas funcoes
    r = ms.correlacao(x, y)
    a, b = ms.regressao_linear(x, y)
    r2 = ms.r_quadrado(x, y)

    c1, c2, c3 = st.columns(3)
    c1.metric("Correlacao (r)", f"{r:.4f}")
    c2.metric("R²", f"{r2:.4f}")
    c3.metric("Covariancia", f"{ms.covariancia(x, y):.4f}")

    # grafico de dispersao com a reta por cima
    st.subheader("Diagrama de dispersao com a reta de regressao")
    reta_x = [min(x), max(x)]
    reta_y = [ms.prever(a, b, reta_x[0]), ms.prever(a, b, reta_x[1])]

    figura, eixo = plt.subplots(figsize=(8, 5))
    eixo.scatter(x, y, s=8, alpha=0.3, color="steelblue", label="jogos")
    eixo.plot(reta_x, reta_y, color="red", linewidth=2, label="reta de minimos quadrados")
    eixo.set_xlabel(coluna_x)
    eixo.set_ylabel(coluna_y)
    eixo.set_title(f"{coluna_y} em funcao de {coluna_x}")
    eixo.legend()
    st.pyplot(figura)

    st.subheader("Equacao da reta")
    if a >= 0:
        st.write(f"### y previsto = {b:.4f} * x + {a:.4f}")
    else:
        st.write(f"### y previsto = {b:.4f} * x - {abs(a):.4f}")

    # --- predicao interativa ---
    st.subheader("Faca uma previsao")
    valor_digitado = st.number_input(f"Digite um valor de {coluna_x}:", value=1.0, step=0.1)
    previsao = ms.prever(a, b, valor_digitado)
    st.success(f"Com {coluna_x} = {valor_digitado:g}, a reta preve {coluna_y} = **{previsao:.4f}**")

    # --- interpretacao ---
    st.subheader("O que esses numeros querem dizer")

    if abs(r) > 0.7:
        forca = "FORTE"
    elif abs(r) > 0.3:
        forca = "MODERADA"
    else:
        forca = "FRACA"

    if r > 0:
        sentido = "positiva (quando uma sobe, a outra tende a subir)"
    else:
        sentido = "negativa (quando uma sobe, a outra tende a descer)"

    st.markdown(
        f"""
        - **Correlacao r = {r:.4f}**: é uma correlacao {forca} e {sentido}.
        - **Inclinacao b = {b:.4f}**: cada vez que {coluna_x} aumenta 1 unidade,
          espera-se que {coluna_y} varie {b:.4f} unidades.
        - **Intercepto a = {a:.4f}**: é o valor previsto de {coluna_y} quando {coluna_x} = 0.
        - **R² = {r2:.4f}**: a reta explica cerca de {r2 * 100:.1f}% da variacao de {coluna_y}.
          Os outros {100 - r2 * 100:.1f}% dependem de coisas que nao estao no modelo.
        """
    )

    st.warning(
        "**ATENCAO: correlacao NAO é causalidade!** Mesmo com um r alto, nao podemos dizer "
        "que uma variavel CAUSA a outra. No nosso caso, um jogo vender bem na Europa nao FAZ "
        "ele vender bem na America do Norte - as duas coisas acontecem juntas porque existe "
        "uma terceira causa por tras: o jogo ser bom/famoso (tipo Mario, GTA, FIFA)."
    )


# ==========================================================
# MODULO 6 - RELATORIO DE DESCOBERTAS
# ==========================================================

def modulo_6_relatorio():
    st.header("Modulo 6 - Nossas 3 Descobertas")

    tabela = carregar_dados()
    vendas = pegar_valores(tabela, "Global_Sales")

    # numeros da descoberta 1
    m = ms.media(vendas)
    med = ms.mediana(vendas)
    cv = ms.coeficiente_variacao(vendas)
    outliers, limite_de_baixo, limite_de_cima = encontrar_outliers(vendas)

    # numeros da descoberta 2
    duas_colunas = tabela[["EU_Sales", "NA_Sales"]].dropna()
    x = duas_colunas["EU_Sales"].tolist()
    y = duas_colunas["NA_Sales"].tolist()
    r = ms.correlacao(x, y)
    r2 = ms.r_quadrado(x, y)

    # numeros da descoberta 3
    generos = tabela_de_frequencias_categorica(pegar_valores(tabela, "Genre"))

    st.subheader("Descoberta 1: pouquissimos jogos vendem muito")
    st.markdown(
        f"""
        A media de vendas globais é **{m:.3f} milhoes de copias**, mas a mediana é so
        **{med:.3f} milhoes** - a media é **{m / med:.1f} vezes maior** que a mediana!
        O coeficiente de variacao gigante (**{cv:.0f}%**) confirma isso.

        Pela regra do IQR, **{len(outliers)} jogos ({len(outliers) / len(vendas) * 100:.1f}%)**
        sao outliers de venda (venderam mais que {limite_de_cima:.3f} milhoes). O campeao vendeu
        **{max(vendas):.2f} milhoes** de copias.

        **Conclusao:** a maioria esmagadora dos jogos vende pouco e um pequeno grupo de
        "arrasa-quarteirao" vende muito, puxando a media para cima. Por isso a media
        sozinha engana - a mediana descreve muito melhor o jogo "tipico".
        """
    )

    st.subheader("Descoberta 2: as vendas das regioes andam juntas (mas isso nao é causa)")
    st.markdown(
        f"""
        A correlacao entre as vendas na Europa e na America do Norte é **r = {r:.3f}**,
        uma correlacao forte e positiva, com **R² = {r2:.3f}** (a reta explica
        {r2 * 100:.1f}% da variacao).

        **Conclusao:** jogos que vendem bem na Europa quase sempre vendem bem nos EUA.
        Mas isso **nao** significa que uma venda causa a outra: as duas sao consequencia
        da mesma coisa - o jogo ser famoso e bem avaliado no mundo todo.
        """
    )

    st.subheader("Descoberta 3: o Teorema Central do Limite salva a analise")
    st.markdown(
        f"""
        O genero com mais jogos lancados é **{generos['Categoria'][0]}**
        ({generos['Frequencia'][0]} jogos, {generos['Frequencia relativa (%)'][0]}% do total).

        E apesar das vendas serem uma variavel super torta (Descoberta 1), quando rodamos a
        simulacao do Modulo 3 sorteando amostras e calculando as medias, essas medias formam
        uma curva parecida com a Normal ja a partir de amostras de tamanho 30.

        **Conclusao:** mesmo com dados nada normais, o Teorema Central do Limite permite
        trabalhar com as medias amostrais como se fossem normais - foi a descoberta que mais
        nos surpreendeu, porque mostra por que a curva Normal aparece em tanto lugar.
        """
    )


# ==========================================================
# MENU LATERAL (escolhe qual modulo mostrar)
# ==========================================================

st.sidebar.title("🎮 Laboratorio de Estatistica")
st.sidebar.write("Dataset: Video Game Sales")

pagina = st.sidebar.radio(
    "Escolha o modulo:",
    [
        "Modulo 0 - Dados Reais",
        "Modulo 2 - Estatistica Descritiva",
        "Modulo 3 - Probabilidade e Simulacao",
        "Modulo 4 - Distribuicoes Teoricas",
        "Modulo 5 - Correlacao e Regressao",
        "Modulo 6 - Descobertas",
    ],
)

st.sidebar.info(
    "O **Modulo 1** (nossas funcoes de estatistica) nao é uma pagina: "
    "ele é o arquivo `minhastats.py`, que faz todas as contas mostradas aqui. "
    "Os testes dele estao em `test_minhastats.py`."
)

if pagina == "Modulo 0 - Dados Reais":
    modulo_0_dados()
elif pagina == "Modulo 2 - Estatistica Descritiva":
    modulo_2_descritiva()
elif pagina == "Modulo 3 - Probabilidade e Simulacao":
    modulo_3_simulacao()
elif pagina == "Modulo 4 - Distribuicoes Teoricas":
    modulo_4_distribuicoes()
elif pagina == "Modulo 5 - Correlacao e Regressao":
    modulo_5_regressao()
else:
    modulo_6_relatorio()
