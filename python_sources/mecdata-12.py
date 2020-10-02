####################################################################################################################################################################################
# CHNAMAZ V 1.0 BETA
# ANTONY APARECIDO PAVARIN
# GABRIEL KENZO KAKIMOTO
# PEDRO HENRIQUE GAZARINI
# PAULO YOSHIO KUGA
# FACULDADE DE ENGENHARIA MECANICA DA UNIVERSIDADE ESTADUAL DE CAMPINAS
# MARÇO DE 2020


#A maioria de nós nunca tinha codado pesado. A grande questão aqui é o fato de que esse script está incompleto, no sentido em que não conseguimos expressar algo alem disso.
#Aprendemos bastante em muito pouco tempo, então esperamos poder compartilhar esse conhecimento com vocês.
#Acredito que a construção do código aparente aos olhos de alguns, marretadas incessantes nos dados e na memória.
#No entanto, vale lembrar, que de formação, nossa faculdade não é de TI. O que nós fizemos, talvez inconscientemente, foi uma usinagem dos dados. :D
#Por efeitos pedagógicos internos e a fim de entendimento posterior, este código tem comentários explicando as funções gerais.
#Gostaríamos de ter computadores mais novos para termos o rodado de forma 100%, inclusive utilizando nos do CVRegressor, para achar a melhor regulagem.
#Esperamos que vocês aproveitem algo. 

#O lider.

#COMENTARIOS EM MAIUSCULO FORAM FEITOS DURANTE A COMPETICAO.
#Os comentarios em minusculo foram feitos apos a competição.
#Os prints servem como debug e para ajudar a indicar com mais facilidade onde está o erro.
####################################################################################################################################################################################

#*******************************************************IMPORTACAO*******************************************************
import pandas as pd     #IMPORTA O PANDAS
#import numpy as np      #IMPORTA O NUMPY
from scipy import stats
import xgboost as xgb
import os
from sklearn.ensemble import RandomForestClassifier
from numba import jit #import numba
print("BIBLIOTECAS CARREGADAS")
####################################################################################################################################################################################
#*******************************************************INSERÇÃO DO CSV NA MEMORIA*******************************************************
dados = pd.read_csv("database_fires.csv") #IMPORTACAO
teste = pd.read_csv("respostas.csv")
print("DADOS CARREGADOS")
dados['mes'] = dados.data.str[4:5].astype(int) #CORTA A STRING DO MES
#dados['ano'] = dados.data.str[7:8]
teste['mes'] = teste.data.str[4:5].astype(int)
#teste['ano'] = teste.data.str[7:8]
####################################################################################################################################################################################
#*******************************************************DEF DE FUNCOES*******************************************************

#****************************************************FUNCAO OPERACAO************************************************
def operacao(coluna, tipo, matriz, lista):
    for index in range(len(coluna)):
        for i in range(len(tipo)):
            if coluna[index] == tipo[i]:
                matriz.append(lista[i])
            else:
                continue

#Ela serve para organizar os dados nas listas. 
####################################################################################################################################################################################
#************************************************************MACHINE LEARNING PARA PREENCHIMENTO DE NULOS*****************************************
#@jit(nopython=True)
#A funcao acabou não sendo usada. Pretendemos melhorar isso caso formos para a final.
def escavadeira(matriz, coluna, ref):
    preenchido = []
    avaliar = []
    reespecifico = []
    linhas = []
    limpa(dados[ref])
    for index in range(len(matriz)):
        if pd.isnull(matriz.iloc[index][coluna]):
            avaliar.append(matriz.iloc[index][ref])
            linhas.append(index)
        else:
            preenchido.append(matriz.iloc[index][coluna])
            reespecifico.append(matriz.iloc[index][ref])
    print(reespecifico)
    print(preenchido)

    Xn_train = reespecifico
    Yn_train = preenchido
    modeln = xgb.XGBClassifier(n_estimators=100)
    modeln.fit(Xn_train,Yn_train)
    Yn_predict = modeln.predict(avaliar)
    print(type(Yn_predict))

####################################################################################################################################################################################
#LIMPA UMA MATRIZ DE VALORES VAZIOS
def limpa(matriz):
    matriz.update(matriz.fillna(matriz.median(0)))
    #Aumentou bastante a precisao.

limpa(dados)
limpa(teste)

print("OUTLIERS REMOVIDOS")
print("DADOS LIMPOS")
####################################################################################################################################################################################

####################################################################################################################################################################################
#DEFINE A PROBABILIDADE
#MATRIZ DE ESTADOS
aux = ['precipitacao','insolacao','evaporacao_piche','temp_max','umidade_rel_med', 'probest', 'probmes']
#Essa matriz antes era declarada apenas proximo a regressão.
uf = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RO", "RS", "RR", "SC", "SE", "SP", "TO"]
#Essa funcao antes era apenas de estado. Depois a generalizei. Mas sobrou a lista de UF's.
def conversao(tipo, coluna):
    comando =  dados[coluna]==tipo
    aux = dados[comando]
    inc = sum(aux.fires)
    total = len(aux.index)
    if total == 0:
        prob = 0
    else:
        prob = inc/total
    return prob
####################################################################################################################################################################################
#DELETA COLUNAS QUE NAO CONTRIBUEM PARA O MODELO
#Isso foi feito porque estavamos usando o sklearn. O sklearn não aceita strings nem nulos.


def dropzone(matriz):
    matriz.set_index('id', inplace=True) #colocando id como indice
    matriz.drop('estacao', inplace=True, axis=1)
    matriz.drop('temp_min', inplace=True, axis=1)
    #matriz.drop('vel_vento_med', inplace=True, axis=1)
    matriz.drop('altitude', inplace=True, axis=1)
    #matriz.drop('temp_max', inplace=True, axis=1)
    matriz.drop('data', inplace=True, axis=1)
    matriz.drop('estado', inplace=True, axis=1)
####################################################################################################################################################################################
#*******************************************************LISTAGEM DE STRINGS, OPERACOES DE INSERCAO*******************************************************
vazio = []
vazio1 = []
cao = []
cao1 = []
pesos0 = []
pesos1 = []
pesos2 = []
pesos3 = []
col_estado = dados['estado'].tolist()
col_estado1 = teste['estado'].tolist()
col_estacao = dados['estacao'].tolist()
col_estacao1 = teste['estacao'].tolist()
col_data = dados['mes'].to_list()
col_data1 = teste['mes'].to_list()

#Isso é confuso de explicar e não houve tempo para explicar na hora.


meses = list(set(col_data)) #Gambiarras
estacoes = list(set(col_estacao))
#anos = list(set(col_ano))
print("LISTAS CRIADAS")
list_prob = [conversao(provincia, 'estado') for provincia in uf]
list_prob1 = [conversao(sensor, 'estacao') for sensor in estacoes]
list_prob2 = [conversao(month, 'mes') for month in meses]
#list_prob3 = [conveano(year) for year in anos]
print("LISTAS ASSOCIADAS") #cada print era um debug. nosso codigo demorava uns 17 min pra rodar a regressao por não termos GPU capacitada. houve uma longa historia pra tentar arrumar uma gpu compativel.
operacao(col_estacao, estacoes, cao, list_prob1)
print("OPERACAO CONCLUIDA")
operacao(col_estacao1, estacoes, cao1, list_prob1)
print("OPERACAO CONCLUIDA")
operacao(col_estado, uf, vazio, list_prob)
print("OPERACAO CONCLUIDA")
operacao(col_estado1, uf, vazio1, list_prob)
print("OPERACAO CONCLUIDA")
operacao(col_data, meses, pesos0, list_prob2)
print("OPERACAO CONCLUIDA")
operacao(col_data1, meses, pesos1, list_prob2)
print("OPERACAO CONCLUIDA")

dados['probest'] = vazio
teste['probest'] = vazio1
#dados['probcao'] = cao
#teste['probcao'] = cao1
dados['probmes'] = pesos0
teste['probmes'] = pesos1

####################################################################################################################################################################################
#*******************************************************DELEÇÃO DE COLUNAS*******************************************************
dropzone(dados)
dropzone(teste)
print("COLUNAS ELIMINADAS")
#dados = dados[(np.abs(stats.zscore(dados)) < 3).all(axis=1)]
#outliers foram cortados no fim a efeito de teste. há uma parte no código que foi alterada (não me lembro qual) que conflitava com os outliers

#escavadeira(dados, 'precipitacao', 'umidade_rel_med')

####################################################################################################################################################################################
#*******************************************************EXPORTACAO*******************************************************
dados.to_csv(r'operacional.csv', index=True)
teste.to_csv(r'testador.csv', index=True)
#era pra tentar pular a parte e ganhar tempo na regressao

####################################################################################################################################################################################
#*******************************************************COMEÇO DO MACHINE LEARNING EM SI*******************************************************

#DEFINICAO DE TERMOS USAVEIS
aux = ['precipitacao','insolacao','evaporacao_piche','temp_max','umidade_rel_med', 'probest', 'probmes'] #,temporada]

#VARIAVEIS DE TREINO
x_train = dados[aux]
y_train = dados.fires

#MODELO
modelo = xgb.XGBClassifier(max_depth=5, objective="binary:logistic", min_child_weight=6, n_estimators=10000, learning_rate=0.001) #esta foi o ultimo envio. não foi bem sucedido. rodei apenas na CPU na ultima meia hora da competição. precisão 2% abaixo do maximo.
modelo.fit(x_train, y_train)
print("TREINADO")
#EXECUÇÃO DO TESTE
modelagem = modelo.predict(teste[aux])

####################################################################################################################################################################################
#*******************************************************EXPORTAÇÃO DE DADOS*******************************************************
#DEFINICAO DO DATASET DE EXPORTAÇÃO
#pada = pd.DataFrame(modelagem)
#pada.to_csv(r'logistico.csv', index=True)
dataframe = pd.DataFrame(modelagem, columns = ['fires'])
dataframe.set_index(teste.index, inplace=True)
#EXPORTAÇÃO DO DATASET PARA CSV
dataframe.to_csv(r'envio1.csv', index = True)
print("EXPORTADO")


#FIM DO PROGRAMA. FORTRAN FEELINGS.