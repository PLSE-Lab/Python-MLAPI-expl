# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


#Zeigt Korrelationen zwischen allen Paaren von Variablen als Heatmap Plot an
def show_correlations_heatmap(filename):

    dataframe = pd.read_csv(filename,sep=';',encoding = "ISO-8859-1")
    
    
    #kategorische daten zu numerischen umwandeln für Korrelationen
    dataframe = dataframe.replace({'Zielvariable':{'ja':1, 'nein':0}})
    dataframe = dataframe.replace({'Geschlecht':{'w':1, 'm':0}})
    #Versuch, Kategorien ordinal nach Gehalt zu sortieren
    dataframe = dataframe.replace({'Art der Anstellung':{'Arbeitslos':0,
                                                         'Student':1,
                                                         'Hausfrau':2,
                                                         'Arbeiter':3,
                                                         'Verwaltung':4,
                                                         'Unbekannt':5,
                                                         'Dienstleistung':6,
                                                         'SelbstÃ\x83Â¤ndig':7,
                                                         'Technischer Beruf':8,
                                                         'Rentner':9,
                                                         'Management':10,
                                                         'GrÃ\x83Â¼nder':11}})
    
    dataframe = dataframe.replace({'Kontaktart':{'Festnetz':1, 'Handy':2, 'Unbekannt':0}})
    dataframe = dataframe.replace({'Kredit':{'ja':1, 'nein':0}})
    
    dataframe = dataframe.replace({'Ergebnis letzte Kampagne':{'kein Erfolg':0,
                                                               'Sonstiges':1,
                                                               'Unbekannt':2,
                                                               'Erfolg':3,}})
    
    print(dataframe['Art der Anstellung'].unique())
    
    Selected_features = ['Dauer','Zielvariable','Alter','Geschlecht','Art der Anstellung','Familienstand','Ausfall Kredit',
            'Kontostand','Haus','Kredit','Kontaktart','Anzahl der Ansprachen','Anzahl Kontakte letzte Kampagne',
            'Ergebnis letzte Kampagne']
    
    X = dataframe[Selected_features]
    
    
    plt.subplots(figsize=(10, 6))
    #mit annotation und rot,gelb, grün
    ax = sns.heatmap(X.corr(),annot=True, cmap="RdYlGn")
    
    plt.show()

show_correlations_heatmap('../input/TrainData.csv')



