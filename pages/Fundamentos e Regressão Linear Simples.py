import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 

st.markdown('## Fundamentos e Regressão Linear Simples')
st.subheader('Conceitos e Matemática Básica')
with st.expander('Conceitos Fundamentais'):
    st.markdown('''
**Variável independente (X):** aquela que explica ou influencia outra variável.
Exemplo: investimento em marketing.

**Variável dependente (Y):** é o resultado ou resposta que se deseja prever.
Exemplo: valor das vendas obtidas.

Na *Regressão Linear*, busca uma relação do tipo:
                
''')
with st.expander('Matemática da Regressão Linear'):
    st.latex(r"Y_i = a + bX_i + \varepsilon_i")
    st.write('''
    **Onde:**  
    - $Y_i$ = variável dependente (resposta observada) [ Causa : Entrada] 
    - $X_i$ = variável independente (preditor)  [ Efeito : Saída]
    - $a$ = intercepto (valor de $Y$ quando $X = 0$)  
    - $b$ = coeficiente angular (taxa de variação de $Y$ por unidade de $X$)  
    - εi= erro aleatório (diferença entre o valor observado e o previsto)
    ''')
with st.expander('Exemplo da vida real'):
    informacoes_vida = {
        'Conceito':['Marketing e Vendas','Logística','Gestão de Estoque'],
        'Variável Independente (X)':['Investimento em anúncios de TV','Distância percorrida (km)','Demanda mensal'],
        'Variável Depende (Y)':['Vendas em reais','Custo do transporte (R$)','Nível de estoque'],
        'Objetivo':['Estimar quanto as vendas aumentam com mais investimento','Estimar o custo total de transporte por distância','Prever o estoque necessário conforme a demanda']
    }
    df_informacoes_vida = pd.DataFrame(informacoes_vida)
    st.write(df_informacoes_vida)

    st.subheader('Exemplos Visual Simples')
    st.write('Imagine uma empresa que coleta os seguintes dados:')
    dados_exemplo_visual = {
        'Investimento em Tv(R$ mil)':[10,20,30,40,50],
        'Vendas(R$ mil)': [30,45,60,80,95]
    }
    df_exemplo_visual = pd.DataFrame(dados_exemplo_visual)
    st.write(df_exemplo_visual)
    st.markdown('''
    Aqui:
                
        X = Investimento em TV
        Y = Vendas 
    ''')
    x = df_exemplo_visual['Investimento em Tv(R$ mil)']
    y = df_exemplo_visual['Vendas(R$ mil)']
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue')
    ax.set_title("Investimento em TV (X) vs Vendas (Y)")
    ax.set_xlabel("Investimento em TV (mil R$)")
    ax.set_ylabel("Vendas (mil R$)")
    ax.grid(True)
    st.pyplot(fig)

with st.expander("📉 Regressão vs. Correlação"):
    st.markdown("""
    **Correlação** mede o grau de associação entre duas variáveis (forte, fraca, positiva ou negativa).  
    👉 Exemplo: quando o investimento em marketing aumenta, as vendas também tendem a aumentar.
    
    **Regressão** vai além — ela **quantifica essa relação** e permite **prever valores**.  
    👉 Exemplo: prever quanto uma venda aumentará ao investir +R$1.000 em publicidade.
    """)

    dados_corr = {
        "Conceito": ["Correlação", "Regressão"],
        "O que faz": [
            "Mede o grau de associação entre duas variáveis.",
            "Quantifica a relação e permite fazer previsões."
        ],
        "Resultado": [
            "Valor entre -1 e 1 indicando força e direção da relação.",
            "Equação matemática que estima o valor de Y a partir de X."
        ],
        "Exemplo": [
            "Aumento de investimento → aumento nas vendas (associação).",
            "A cada R$1.000 investidos, vendas aumentam R$200 (previsão)."
        ]
    }

    df_corr = pd.DataFrame(dados_corr)
    st.dataframe(df_corr, use_container_width=True)

with st.expander("⚖️ Erro, Viés e Variância"):
    st.markdown("""
    Esses três conceitos são **essenciais** para entender o comportamento e a generalização do modelo:
    
    - **Erro (Error):** diferença entre o valor real e o valor previsto pelo modelo.  
    - **Viés (Bias):** indica o quanto o modelo é simplista (subajuste / *underfitting*).  
    - **Variância (Variance):** mostra o quanto o modelo é sensível às variações dos dados (sobreajuste / *overfitting*).  
    
    O objetivo é **encontrar um equilíbrio** entre **viés** e **variância** — conhecido como *trade-off*.
    """)

    dados_erro = {
        "Conceito": ["Erro (Error)", "Viés (Bias)", "Variância (Variance)"],
        "Descrição": [
            "Diferença entre o valor real e o valor previsto.",
            "O quanto o modelo é simplista demais (subajuste).",
            "Sensibilidade do modelo às flutuações dos dados (sobreajuste)."
        ],
        "Exemplo": [
            "O modelo previu 100 e o valor real era 120 → erro = 20.",
            "O modelo linear tenta prever dados não lineares.",
            "O modelo memoriza o conjunto de treino e erra no teste."
        ]
    }

    df_erro = pd.DataFrame(dados_erro)
    st.dataframe(df_erro, use_container_width=True)

with st.expander('Projeto Simples'):
    dados_projetos= {
        'x':[3,2,-1,4],
        'y':[7,5,-1,9]
    }
    df_projeto = pd.DataFrame(dados_projetos)
    st.markdown('Base Original')
    st.dataframe(df_projeto)

    st.markdown('Adicionando x², y² e xy')
    df_projeto_novo = df_projeto
    
    df_projeto_novo['x²'] = df_projeto['x']**2
    df_projeto_novo['y²'] = df_projeto['y']**2
    df_projeto_novo['xy'] = df_projeto['x']*df_projeto['y']

    st.dataframe(df_projeto_novo)

    st.markdown('Calculando o Coeficiente ')
    st.latex(r"""r_{xy} = \frac{\underbrace{n \sum xy - (\sum x)(\sum y)}_{\text{numerador: covariância multiplicada por n}}}{\underbrace{\sqrt{\left(n \sum x^2 - (\sum x)^2\right) \left(n \sum y^2 - (\sum y)^2\right)}}_{\text{denominador: produto das variâncias multiplicadas por n}}}""")

    n = len(df_projeto_novo)

    somatorio_x  = df_projeto_novo['x'].sum()
    somatorio_y  = df_projeto_novo['y'].sum()
    somatorio_xX = df_projeto_novo['x²'].sum()
    somatorio_yY = df_projeto_novo['y²'].sum()
    somatorio_xy = df_projeto_novo['xy'].sum()

    col1, col2  = st.columns(2)

    with col1:
        st.markdown('**1.Calculando o coeficiente Angular**')
        st.latex(r'''b = \frac{n \sum xy -  (\sum x )(\sum y )}{n \sum x^2 - (\sum x)^2}''')
        b =  ((n * somatorio_xy) - (somatorio_x*somatorio_y)) / ((n*somatorio_xX) - (somatorio_x**2))    
        st.write(f'Resultado : {b}')
        st.write('---------------------------')
        
        
    with col2:
        st.markdown('**2.Calculando o Intercepto**')
        st.latex(r'''a = \frac{\sum y -  b \sum x}{n}''')
        a =  (somatorio_y - (b*somatorio_x))/n
        st.write(f'Resultado : {a}')
        st.write('---------------------------')

    col3, col4 = st.columns(2)

    with col3:
            st.markdown('**3.Calculando o erro aleatório**')
            st.latex(r'''Y = a + b_{x}​''')
            df_projeto_novo['Y_pred'] = a + b * df_projeto_novo['x']
            st.dataframe( df_projeto_novo['Y_pred'] )
        
    with col4:
            st.markdown('**4.Calculando o Epsilon**')
            st.latex(r'''ε_{i} = Y_{i} - Yî​''')
            df_projeto_novo['epsilon'] = df_projeto_novo['y'] - df_projeto_novo['Y_pred']
            st.dataframe(df_projeto_novo['epsilon'])

    st.markdown(' Calculando os erros')
    mae = abs(df_projeto_novo['epsilon']).mean()
    mse = (df_projeto_novo['epsilon'] ** 2).mean()
    rmse = mse ** 0.5
    st.write(f'MAE:  {mae}')
    st.write(f'MSE:  {mse}')
    st.write(f'RMSE: {rmse}')
with st.expander("⚙️ Métricas de Erro"):
    st.markdown("""
    As **métricas de erro** são usadas para avaliar o desempenho de um modelo de regressão.  
    Elas medem a diferença entre o valor **real (y)** e o valor **previsto (ŷ)**.""")
    st.markdown(''' - **MSE (Mean Squared Error)** Penaliza   mais fortemente **grandes erros**. Quanto menor, melhor''')
    st.latex(r"MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2")
    st.markdown('' '- **RMSE (Root Mean Squared Error)** Interpretação no mesmo **nível dos dados originais** (mesma unidade de medida)''')
    st.latex(r"RMSE = \sqrt{MSE}")
    st.markdown('''- **MAE (Mean Absolute Error)**  o **erro médio absoluto** sem penalizar tanto os grandes desvios.''')
    st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y_i}|")


with st.expander ('Resumo'):
    dados = {
    "Conceito": ["Variável Independente", "Variável Dependente", "Relação"],
    "Símbolo": ["X", "Y", "( y = a + bx )"],
    "Papel": ["Causa, entrada, preditora", "Efeito, saída, resposta", "Reta de regressão"],
    "Exemplo": [
        "Investimento, horas de máquina, temperatura",
        "Vendas, produção, custo, taxa de defeitos",
        "Interpreta o impacto de X sobre Y"]
}
    df_variaveis = pd.DataFrame(dados)

    # Exibindo o DataFrame
    st.write(df_variaveis)