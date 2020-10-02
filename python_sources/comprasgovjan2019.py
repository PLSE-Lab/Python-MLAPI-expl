# Importa os dados de compras de Janeiro de 2019
import pandas as pd
compras = pd.read_csv('../input/201901_Compras.csv', sep=';', encoding='latin1')

# Visualização Inicial
compras.info()  # Temos 3.291 linhas e 19 colunas
compras['word_count'] = compras['Objeto'].apply(lambda x: len(str(x).split(" ")))
compras[['Objeto','word_count']].head()
compras.word_count.describe() # Temos em media 22 palavras em cada objeto, o objeto vai de 2 a 83 palavras

# Define stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('portuguese')

# Pre-processamento do texto. Queremos só a coluna 'Objeto'. Faz a limpeza.
import re
texto = compras['Objeto'].astype(str)
texto = texto.str.lower() # transforma para letra minuscula
texto = [re.sub("(\\d|\\W)+", " ", e) for e in texto] # remove caracteres especiais, pontuação e digitos

# Nuvem de Palavras
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(texto))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)

# Cria TF. Cria o vocabulário removendo stopwords e ignorando palavras que aparecem em 85% dos documentos
# Retorna o Term-document matriz: cada coluna é uma palavra e cada linha um documento e os valores a contagem de palavras
from sklearn.feature_extraction.text import CountVectorizer
vocab = CountVectorizer(max_df=0.85,stop_words=stopwords) # É possível limitar o maximo de palavras com o max_features=1000
#list(vocab.vocabulary_.keys())[:10] # Mostra 10 palavras do vocabulário
vetor=vocab.fit_transform(texto) # Cria a Term-document matriz com 3291 linhas e 7299 colunas

# Cria o IDF 
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf = tfidf_transformer.fit(vetor)

feature_names=vocab.get_feature_names()
doc=texto[0] # Seleciona um documento
tf_idf_vector=tfidf_transformer.transform(vocab.transform([doc]))


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results

# Ordena os itens da matriz 
sorted_items=sort_coo(tf_idf_vector.tocoo())
 
# Mostra apenas os top 10 
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
 
# Imprime os resultados
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords: print(k,keywords[k])

# Se testarmos com outro documento com sentença aleatoria
doc2 = "o sapo não tem chulé mas tem caneta"
vector2 = tfidf_transformer.transform(vocab.transform([doc2]))
sorted_items2=sort_coo(vector2.tocoo())
keywords2=extract_topn_from_vector(feature_names,sorted_items2,10)

# Imprime os resultados
print("\n=====Doc=====")
print(doc2)
print("\n===Keywords===")
for k in keywords2: print(k,keywords2[k])
 
