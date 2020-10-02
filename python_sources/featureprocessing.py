# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime  


""" löscht irrelevante Spalten von columnnames aus der csv-Tabelle 'filename' und
    speichert die entstandene csv-Tabelle unter outfile ab."""

def delete_column(dataframe,columnnames):
    
    #CSV Daten lesen
    df = dataframe
    
    print ("Anzahl der Zeilen = {}\n".format(df.shape[0]))
    
    #erste paar Zeilen des dataframe.
    print('{}\n'.format(df.head()))
    #Anzeige fehlender Werte
    print('Anzahl der leerstehenden Felder je Spalte: \n{}'.format(df.isnull().sum()))
    
    #lösche Spalte aus dataframe.
    df.drop(columnnames ,axis=1, inplace = True)
    
    return df


""" wandle Tag und Monat des Anrufs in 'Tage zwischen erstem Anruf und diesem Anruf' um."""
def convert_day_month_to_days(filename,outfile):
    dataframe = pd.read_csv(filename,sep=';',encoding = "ISO-8859-1")
    dataframe = delete_column(dataframe,['Anruf-ID','Tage seit letzter Kampagne'])
    
    #füge Tage und Monate in Listen ein
    #speichere beide Spalten in separate Listen ohne Überschriften
    days = []
    months = []
    
    days = dataframe['Tag'].tolist()
    months = dataframe['Monat'].tolist()
    
    """
    for i in range(0,10):
        print(months[i])
    """
    
    #lösche beide Spalten aus Hauptliste
    dataframe.drop(['Tag','Monat'] ,axis=1, inplace = True)
    
    #erzeuge Datum-Liste aus Tag- und Monatsangaben
    #Annahme: alle Tage und Monate sind vom selben Jahr.
    #Annahme: es handelt sich um kein Schaltjahr, da kein 29.2. in Tabelle vorkam.
    monthSwitcher = { 
        'jan': 1, 
        'feb': 2, 
        'mar': 3,
        'apr': 4, 
        'may': 5, 
        'jun': 6, 
        'jul': 7, 
        'aug': 8, 
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }
    
    dates = []
    for i in range(0,len(days)):
        date = datetime.date(year = 2018, day = days[i] ,month = monthSwitcher[months[i]])
        dates.append(date)
    
    #Kopie
    dummydates = dates[:]   
    #suche kleinstes Datum heraus
    earliestDate = sorted(dummydates)[0]
    
    #berechne Anzahl von Tagen seit 'earliestDate' aus den Daten
    daysSinceEarliest = []
    for i in range(0,len(dates)):
        daySinceEarliest = (dates[i] - earliestDate).days
        daysSinceEarliest.append(daySinceEarliest)
    
    
    #Länge der neuen Liste
    print('Länge der neuen Liste "Tage zwischen Beginn und Anruf" = {}\n',format(len(daysSinceEarliest)))
    
    #addiere neue variable zu Hauptliste hinzu mit Korrekter Überschrift und speichere Datei ab
    
    #muss erst in ein Series umgewandelt werden, bevor es zum dataframe hinzugefügt werden kann.
    s = pd.Series(daysSinceEarliest)
    dataframe['Tage zwischen Beginn und Anruf'] = s
    dataframe.to_csv(outfile, sep=';', index = False) 
    
    
convert_day_month_to_days('../input/TrainData.csv','Traindata.csv')
convert_day_month_to_days('../input/TestData.csv','TestData.csv')