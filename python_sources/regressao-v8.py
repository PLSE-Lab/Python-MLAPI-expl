# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:08:22 2019

@author: alexy_000
"""

#Importação dos pacotes
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.mixture import GaussianMixture
from xgboost import XGBRegressor
import time

import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
import seaborn as sns
import imblearn
import itertools

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


#Numero do arquivo de envio
num_submission = 16

#caminho dos dataset
caminho_bases = "C:\\Users\\alexy_000\\Desktop\\python\\dsa\\competicao_02\\bases\\"

#nome dos arquivos de treino e teste
arquivo_treino = "dataset_treino.csv"
arquivo_teste = "dataset_teste.csv"

#lista de valores vazios a serem preenchidos na leitura dos data sets
na_values = ["Not Available", "", 0]

#definindo a variavel alvo a ser prevista
variavel_target = "ENERGY STAR Score"

#importação do data frame de treino
df_treino = pd.read_csv(caminho_bases + arquivo_treino,na_values = na_values)
df_treino.head()

#importação do data frame de teste
df_teste = pd.read_csv(caminho_bases + arquivo_teste,na_values = na_values)

#lista de colunas que não serão utilizadas para o modelo, muitos valores missing ou informações pouco relevantes
colunas_drop = ['Order', 'Property Id', 'Property Name', 'Parent Property Id', 'Parent Property Name', 'NYC Borough, Block and Lot (BBL) self-reported', 'NYC Building Identification Number (BIN)', 'Address 1 (self-reported)', 'Address 2', 'Street Number', 'Street Name', 'Borough', 'Primary Property Type - Self Selected', 'List of All Property Use Types at Property', 'Fuel Oil #1 Use (kBtu)', 'Fuel Oil #2 Use (kBtu)', 'Fuel Oil #4 Use (kBtu)', 'Fuel Oil #5 & 6 Use (kBtu)', 'Diesel #2 Use (kBtu)', 'District Steam Use (kBtu)', 'Water Use (All Water Sources) (kgal)', 'Water Intensity (All Water Sources) (gal/ft²)', 'Release Date', 'Water Required?', 'DOF Benchmarking Submission Status'
]

#drop das colunas listadas
df_treino = df_treino.drop(columns = colunas_drop)
df_treino.head()


#criação de um dicionario para agrupar alguns tipos de propriedades
dic_agrup_construcao = {'Bank Branch':'OFFICE', 'Courthouse':'OFFICE', 'Wholesale Club/Supercenter':'SUPERMARKET', 'Parking':'WAREHOUSE', 'Refrigerated Warehouse':'REFRIGERATED WAREHOUSE', 'Worship Facility':'SCHOOL', 'Supermarket/Grocery Store':'SUPERMARKET', 'Financial Office':'OFFICE', 'Hospital (General Medical & Surgical)':'HOSPITAL', 'Medical Office':'OFFICE', 'Distribution Center':'DISTRIBUTION CENTER', 'Retail Store':'STORE', 'Senior Care Community':'RESIDENCE', 'Residence Hall/Dormitory':'RESIDENCE', 'K-12 School':'SCHOOL', 'Non-Refrigerated Warehouse':'WAREHOUSE', 'Hotel':'HOTEL', 'Office':'OFFICE', 'Multifamily Housing':'HOUSE'
}

dic_agrup_construcao_2 = {'Adult Education':'OTHER', 'Museum':'OTHER', 'Enclosed Mall':'OTHER', 'Personal Services (Health/':'OTHER', 'Performing Arts':'OTHER', 'Senior Care Community':'RESIDENCE', 'Other - Utility':'OTHER', 'Laboratory':'OFFICE', 'Other - Public Services':'OFFICE', 'Automobile Dealership':'WAREHOUSE', 'Self-Storage Facility':'WAREHOUSE', 'Bar/Nightclub':'RESTAURANT', 'Other - Mall':'STORE', 'Residence Hall/Dormitory':'RESIDENCE', 'Distribution Center':'DISTRIBUTION CENTER', 'Social/Meeting Hall':'RESTAURANT', 'Food Sales':'STORE', 'Refrigerated Warehouse':'REFRIGERATED WAREHOUSE', 'Other - Education':'SCHOOL', 'Hotel':'HOTEL', 'Worship Facility':'SCHOOL', 'Food Service':'RESTAURANT', 'Swimming Pool':'SCHOOL', 'Other - Recreation':'SCHOOL', 'College/University':'SCHOOL', 'Other - Restaurant/Bar':'RESTAURANT', 'Pre-school/Daycare':'SCHOOL', 'Fitness Center/Health Club':'OFFICE', 'Non-Refrigerated Warehouse':'WAREHOUSE', 'K-12 School':'SCHOOL', 'Data Center':'OFFICE', 'Convenience Store without':'STORE', 'Outpatient Rehabilitation/Physical Therapy':'OFFICE', 'Fast Food Restaurant':'RESTAURANT', 'Other - Entertainment/Public Assembly':'OTHER', 'Other - Services':'OTHER', 'Supermarket/Grocery Store':'SUPERMARKET', 'Multifamily Housing':'HOUSE', 'Bank Branch':'OFFICE', 'Urgent Care/Clinic/Other Outpatient':'OFFICE', 'Financial Office':'OFFICE', 'Restaurant':'RESTAURANT', 'Medical Office':'OFFICE', 'Office':'OFFICE', 'Other':'OTHER', 'Retail Store':'STORE'
}

#função para tratar os campos do data, terá que aplicar no dataset de treino também
#por isso foi construida a função
def tratar_campos(df_in):
#a variavel da função e´o proprio dataset

    #copia o dataset na memoria    
    df = df_in.copy()
    
    #preenche os campos vazios com 0    
    df.loc[:,"BBL - 10 digits"]=df.loc[:,"BBL - 10 digits"].fillna("00")
    #remove caracteres especiais    
    df.loc[:,"BBL - 10 digits"] =df.loc[:,"BBL - 10 digits"].map(lambda x: x.replace("\u200b", "").replace(" ","").replace(";","").replace(",",""))

    # quebra o codigo BBL em 3 outros codigos, conforme no manual       
    df["borough_code"] = df.loc[:,"BBL - 10 digits"].map(lambda x: str.zfill( str(x[0]),2)).fillna("-")   
    df["tax_block"] = df.loc[:,"BBL - 10 digits"].map(lambda x: x[1:6])
    df["tax_lot"] = df.loc[:,"BBL - 10 digits"].map(lambda x: x[6:10].strip(";"))
    
    #cria uma nova variavel ajustada para a area de construção    
    df["area_construcao_1"] = df["Largest Property Use Type - Gross Floor Area (ft²)"].fillna(0)
    df["area_construcao_2"] = df["2nd Largest Property Use - Gross Floor Area (ft²)"].fillna(0)
    df["area_construcao_3"] = df["3rd Largest Property Use Type - Gross Floor Area (ft²)"].fillna(0)
    
#    calcula a area total
    df["total_gfa"] = df["Largest Property Use Type - Gross Floor Area (ft²)"].fillna(0) +\
    df["2nd Largest Property Use - Gross Floor Area (ft²)"].fillna(0) + \
    df["3rd Largest Property Use Type - Gross Floor Area (ft²)"].fillna(0)
    
#    cria uma metrica de consumo de gas natural pelo tamanho da area da construção
    df["natural_gas_use_per_ft"] = df["Natural Gas Use (kBtu)"]/df["total_gfa"]
    df["electric_use_purchase_ft"] = df["Electricity Use - Grid Purchase (kBtu)"]/df["total_gfa"]
    
#    agrupa os tipos de construção usando os dicionarios definidos anteriormente
    df["tp_construcao"] = df["Largest Property Use Type"].map(dic_agrup_construcao).fillna("OTHER")
    df["tp_construcao_2"] = df["2nd Largest Property Use Type"].map(dic_agrup_construcao_2).fillna(df["tp_construcao"])
    df["tp_construcao_3"] = df["3rd Largest Property Use Type"].map(dic_agrup_construcao_2).fillna(df["tp_construcao"])
    

    return df
    del df
    gc.collect()

#aplica a função e cria um novo dataframe
df = tratar_campos(df_treino)

df.columns

#definindo as variaveis categoricas que serão utilizadas  no modelo
variaveis_categoricas = ["tp_construcao", "tp_construcao_2", "borough_code"]
#variaveis_categoricas = ["tp_construcao"]

#definindo as variaveis numericas para o modelo
variaveis_numericas = ["Site EUI (kBtu/ft²)",
"Weather Normalized Site EUI (kBtu/ft²)",
"Weather Normalized Site Electricity Intensity (kWh/ft²)",
#"Weather Normalized Site Natural Gas Intensity (therms/ft²)",
"Weather Normalized Source EUI (kBtu/ft²)",
#"Natural Gas Use (kBtu)",
"natural_gas_use_per_ft",
#"Total GHG Emissions (Metric Tons CO2e)",
#"Occupancy",
"total_gfa",
"area_construcao_1",
"area_construcao_2",
#"electric_use_purchase_ft",
#"area_construcao_3",
#"Direct GHG Emissions (Metric Tons CO2e)",
#"Indirect GHG Emissions (Metric Tons CO2e)",
"Source EUI (kBtu/ft²)"]


#função para preencher valores missing
preenche_0 = Imputer(missing_values = np.nan,strategy = "mean", axis=0)

#tabela de frequencia dos tipos de construcao
df.loc[:,"tp_construcao"].value_counts()
df.loc[:,"tp_construcao_2"].value_counts()
df.loc[:,"tp_construcao_3"].value_counts()


#função para detectar os outliers
def marca_outliers(df, atributos_finais):
    df_temp = df.copy()   
    df_temp["flag_outlier"] = 0
    df_temp["var_outlier"] = ""
    contador = 1
    
    for atr in atributos_finais:
        first_quartile = df_temp.loc[:,atr].quantile(0.25)
        third_quartile = df_temp.loc[:,atr].quantile(0.75)
        IQR = (third_quartile - first_quartile)
        lower_bound_iqr = first_quartile - 1.5*IQR
        upper_bound_iqr = third_quartile + 1.5*IQR
        
        var_temp = df_temp.loc[:,atr].apply(lambda x: 1 if (x < lower_bound_iqr or x> upper_bound_iqr ) else 0)
            
        df_temp["flag_outlier"]  = df_temp["flag_outlier"] + var_temp
        filtro_outlier = (var_temp == 1)
        
        df_temp.loc[filtro_outlier,"var_outlier"] = df_temp.loc[filtro_outlier,"var_outlier"] + ", " + atr
        df_temp.loc[:,"var_outlier"] = df_temp.loc[:,"var_outlier"].fillna("")
    
    return df_temp["flag_outlier"].values, df_temp["var_outlier"].values



#gera os graficos boxplot para cada variavel categoricas e as variaveis numericas
for var_categ in variaveis_categoricas:
    for var in variaveis_numericas:
        plt.title("%s x %s"%(var_categ, var))
        sns.boxplot(x = var_categ, y=var, data = df)
        plt.show()
        
#variaveis que serão sujeitas aos testes de outliers
var_temp = list(filter(lambda x: x not in ["total_gfa","area_construcao_1","area_construcao_2","area_construcao_3",],variaveis_numericas))

#gera um novo data set
df_2 = df.copy()

#varre o dataset com base no tipo de construção e preenche os valores missing
for borough in df_2.loc[:,"tp_construcao"].drop_duplicates().values:
    filtro = (df_2.loc[:,"tp_construcao"] == borough)
    df_2.loc[filtro, variaveis_numericas] = preenche_0.fit_transform(df_2.loc[filtro, variaveis_numericas])
    plt.title("historico de score em %s"%(borough))
    sns.distplot(df_2.loc[filtro, variavel_target])
    df_2.loc[filtro,"flag_outlier"], df_2.loc[filtro,"var_outlier"]= marca_outliers(df_2.loc[filtro,:], var_temp )   
    
    plt.show()
    for var in variaveis_numericas:
        plt.title("historico de %s em %s"%(var, borough))
        sns.distplot(df_2.loc[filtro, var])
        plt.show()


df_2.loc[:,"flag_outlier"].value_counts()        
df_2.loc[:,"var_outlier"].value_counts()        
agrup = df_2.groupby(["flag_outlier","var_outlier"]).size().reset_index()

corr = df_2.corr()

for var in variaveis_numericas:
    plt.title(var)
    sns.distplot(df_2[var])  
    plt.show()

#cria um encoder para fazer um label para as variaveis categoricas
encoder = LabelEncoder()

for label in variaveis_categoricas:
    df_2.loc[:, label] = df_2.loc[:, label].fillna("00")
    
    df_2.loc[:, label] = encoder.fit_transform(df_2.loc[:, label])


#copia o data set 
df_treino_2 = df_2


#listando os modelos e os parametros para fazer  grid search
modelos = []
valores_grid = dict()
modelos.append(("LR", LinearRegression(), valores_grid))
   
valores_grid = dict()
modelos.append(("LASSO", Lasso(), valores_grid))

valores_grid = dict()

modelos.append(("ElasticNet", ElasticNet(), valores_grid))


valores_grid = dict(n_neighbors = np.arange(1,21))
modelos.append(("KNN", KNeighborsRegressor(), valores_grid))

valores_grid = dict(max_depth = np.arange(5,15))
modelos.append(("DT", DecisionTreeRegressor(), valores_grid))


#c_values = np.linspace(0.1, 2, num = 10)
#kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
#valores_grid = dict(C = c_values, kernel = kernel_values)
#modelos.append(('SVM', SVR(),valores_grid))

#help(AdaBoostClassifier())
valores_grid = dict(n_estimators = np.arange(50,500,50))
modelos.append(("AD", AdaBoostRegressor(), valores_grid))

help(RandomForestRegressor())
valores_grid = dict(n_estimators = np.arange(100,500,100), max_depth = np.arange(5,15,5))
modelos.append(("GBM", GradientBoostingRegressor(),valores_grid))
modelos.append(("RF", RandomForestRegressor(),valores_grid))
modelos.append(("EXT", ExtraTreesRegressor(),valores_grid))

#valores_grid = dict()
modelos.append(("XGB", XGBRegressor(),valores_grid))


#objeto k fold para fazer a cross validation do modelo
kfold = model_selection.KFold(n_splits= 10, random_state = 7)
#listsa para guardar os resultados dos modelos
scores = []

#separa em variaveis X e Y
y_treino = df_treino_2.loc[:,variavel_target].values
x_treino = df_treino_2.loc[:,variaveis_categoricas+variaveis_numericas].values

#lista de scalers para testar as normalizações
lista_scalers = [StandardScaler(), MinMaxScaler()]
tempo_inicio = time.time()

#gera-se os modelos
for nome, modelo, valores_grid in modelos:
    for scaler in lista_scalers:
        print("Estimando modelo: %s" %(nome))
        escala = scaler.fit(x_treino)
        rescaled_x = escala.transform(x_treino)
        inicio = time.time()
        grid = model_selection.GridSearchCV(estimator = modelo, param_grid = valores_grid, scoring = "neg_mean_absolute_error", return_train_score = True, cv = kfold)
        grid.fit(rescaled_x, y_treino)
        fim = time.time()
        
        print("Tempo Execucao = %f minutos" %(float( (fim - inicio )/60)))
        df_temp = pd.DataFrame(grid.cv_results_)
        df_temp["modelo"] = nome
        df_temp["scaler"] = scaler      
        scores.append(df_temp)

tempo_fim = time.time()
#empilha todos os resultados
df_scores = pd.concat(scores)

#realiza um box plot
sns.boxplot(x = "modelo", y = "mean_test_score", data = df_scores.loc[df_scores["modelo"]!="SVM",:])

#agrupa a base por modelo buscando o melhor resultado
df_scores_agg = df_scores.groupby(["modelo"])["mean_test_score"].agg(max).reset_index()
df_scores_agg.sort_values(by = "mean_test_score", ascending = False, inplace = True)
df_scores_agg

#cria um dicionario
df_scores_agg_dict = {modelo:valor for modelo, valor in df_scores_agg.values}
df_scores_agg_dict

#filtra na base de resultados a melhor parametrização para cada modelo
df_scores["best_result"] = df_scores["modelo"].map(df_scores_agg_dict)
df_best = df_scores.loc[df_scores["mean_test_score"]== df_scores["best_result"],:]
df_best.sort_values(by = "mean_test_score",inplace = True)



#copia o dataset de teste
df_teste2 = df_teste.copy()
df_teste2  = tratar_campos(df_teste)

#aplica os tratamentos do dataset de treino no de teste
df_teste2["borough_code"].value_counts()
de_para_zip_borough = df_teste2.loc[df_teste2["borough_code"] != "00",["Postal Code", "borough_code"]].drop_duplicates()
dic_de_para_zip = {zip_code:borough_code for zip_code, borough_code in de_para_zip_borough.values}

df_teste2.loc[df_teste2["borough_code"] == "00", "borough_code"] = df_teste2.loc[df_teste2["borough_code"] == "00", "Postal Code"].map(dic_de_para_zip)
df_teste2["borough_code"].value_counts()



#gera um novo dataset de teste
df_2 = df.copy()

for borough in df_2.loc[:,"tp_construcao"].drop_duplicates().values:
    filtro = (df_2.loc[:,"tp_construcao"] == borough)
    input_values = preenche_0.fit(df_2.loc[filtro, variaveis_numericas].values)
    df_2.loc[filtro, variaveis_numericas] = preenche_0.transform(df_2.loc[filtro, variaveis_numericas])
    filtro_teste = (df_teste2.loc[:,"tp_construcao"] == borough)
    df_teste2.loc[filtro_teste, variaveis_numericas] = preenche_0.transform(df_teste2.loc[filtro_teste, variaveis_numericas].values)
    plt.title("historico de score em %s"%(borough))
    sns.distplot(df_2.loc[filtro, variavel_target])
    df_2.loc[filtro,"flag_outlier"], df_2.loc[filtro,"var_outlier"]= marca_outliers(df_2.loc[filtro,:], var_temp )   
    plt.show()
    for var in variaveis_numericas:
        plt.title("historico de %s em %s"%(var, borough))
        sns.distplot(df_2.loc[filtro, var])
        plt.show()
       

encoder = LabelEncoder()

for label in variaveis_categoricas:
    print(label)
    df_2.loc[:, label] = df_2.loc[:, label].fillna("00")
    encoder.fit(df_2.loc[:, label])
    df_teste2.loc[:, label] = df_teste2.loc[:, label].fillna("00")
    df_2.loc[:, label] = encoder.fit_transform(df_2.loc[:, label])
    df_teste2.loc[:, label] = encoder.fit_transform(df_teste2.loc[:, label])

df_teste2["borough_code"]
df_2["borough_code"].drop_duplicates()

#testando os melhores modelos
modelos_finais = []
modelos_finais.append(("RF",RandomForestRegressor(n_estimators = 250, max_depth = 10), StandardScaler()))
modelos_finais.append(("GBM",GradientBoostingRegressor(n_estimators = 100, max_depth = 5), StandardScaler()))
modelos_finais.append(("XGB",XGBRegressor(n_estimators = 100, max_depth = 5), StandardScaler()))
modelos_finais.append(("XGB_Max",XGBRegressor(n_estimators = 100, max_depth = 5), MinMaxScaler()))

#modelos_finais.append(("XGB_200_5",XGBRegressor(n_estimators = 200, max_depth = 5), StandardScaler()))
#modelos_finais.append(("XGB_100_10",XGBRegressor(n_estimators = 100, max_depth = 10), StandardScaler()))
#modelos_finais.append(("XGB_200_20",XGBRegressor(n_estimators = 200, max_depth = 20), StandardScaler()))

#modelo = XGBRegressor(n_estimators = 100, max_depth = 5)
result_final = []
for label,modelo, scaler in modelos_finais:
    print("Estimando %s"%(label))
    x = df_2.loc[:, variaveis_numericas + variaveis_categoricas].values
    y = df_2.loc[:, variavel_target].values
    
    x_train, x_test, y_train, y_test =  model_selection.train_test_split(x,y, test_size= 0.3, random_state = 7)
    
#    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    rescaled_x_train = scaler.transform(x_train)
    
    
    modelo.fit(rescaled_x_train, y_train)
    rfe = RFE(modelo)
    rfe.fit(rescaled_x_train, y_train)
    ranking = rfe.ranking_
    colunas = df_2.loc[:, variaveis_numericas + variaveis_categoricas].columns.values
    colunas  = colunas.T
    ranking =ranking.T
    
    dicionario = {"ranking":ranking, "variaveis":colunas}   
    df_features = pd.DataFrame(dicionario)
    df_features["modelo"] = label
    previsao_teste = modelo.predict(scaler.transform(x_test))
    
    erro_medio_abs = mean_absolute_error(previsao_teste, y_test)
    erro_medio_abs
    
    print("Modelo: %s Erro medio abs = %f"%(label, erro_medio_abs))
    df_features["erro_medio_abs"] = erro_medio_abs
    result_final.append(df_features)

df_result_final = pd.concat(result_final)

#modelo = GradientBoostingRegressor(n_estimators = 100, max_depth = 5)


#o melhor modelo foi o XGB
modelo = XGBRegressor(n_estimators = 100, max_depth = 5)
scaler = StandardScaler()


#a previsao sera estimada aplicando o modelo em 1000 dataset de treino e a previsão final
#será uma média desses 1000 resultados
#será feito isso para convergir os valor para o valor mais provável (Monte Carlo)

n_range = 1000
resultados = []

for prev in range(1,n_range+1):
    print("gerando solucao numero %i"%(prev))
    inicio = time.time()
    #separa a base em treino e teste    
    x_train, x_test, y_train, y_test =  model_selection.train_test_split(x,y, test_size= 0.3, random_state = prev)
    #aplica a escala    
    scaler = scaler.fit(x_train)
    rescaled_x_train = scaler.transform(x_train)
    #estima o modelo       
    modelo.fit(rescaled_x_train, y_train)
    
    #pega as variaveis x e y do dataset de teste que será enviado    
    x_final = df_teste2.loc[:, variaveis_numericas + variaveis_categoricas].values
    #aplica a mesma escala do data set de treino    
    x_final  = scaler.transform(x_final)
    
    #realiza previsao    
    previsao = modelo.predict(x_final)
    
    #copia o dataset de teste    
    df_out = df_teste2.copy()
    
    #atribui a previsao no dataset    
    df_out["score"] = previsao
    df_out["num_previsao"] = prev
    
#    append na lista de resultados
    resultados.append(df_out)
    del df_out
    fim = time.time()
    delta_t = fim-inicio
    delta_t = delta_t/60
    print("tempo execucao %i/%i = %f minutos"%(prev,n_range,delta_t))


#empilha todos os resultados
df_out = pd.concat(resultados)
#tira a media e desvio padrão de todos os resultados
df_out = df_out.groupby("Property Id")["score"].aggregate({"score":"mean", "std":"std"}).reset_index()
df_out.head()

#ajuste nas previsoes para ficar entre 0 e 100 e numeros inteiros
df_out["score"] = df_out["score"].map(lambda x: np.round(max(min(x,100),0),0))
df_out["score"] = df_out["score"].map(int)
df_out["score"].plot(kind = "hist")

#gera CSV final de envio
df_out.loc[:,["Property Id", "score"]].to_csv(caminho_bases + "arquivo_envio_"+str(num_submission)+".csv", index = False)




