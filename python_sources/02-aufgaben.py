#!/usr/bin/env python
# coding: utf-8

# # Aufgaben
# In diesem Notebook werdet Ihr die Grundlagen der Daten-Exploration ueben.
# Dazu werden wir vor Allem auf die Bibliotheken numpy und pandas setzen.
# 
# Bereitet alle Aufgaben bis zur naechsten Worksession vor.
# Schreibt euch alle Fragen auf, damit wir diese in der Session beantworten koennen.
# Es ist nicht schlimm, wenn nicht alle Loesungen auf Anhieb funktionieren.
# Am Ende des Notebooks gibt es ein Skript mit dem Ihr eure Loesungen zusammentragen und auf der Kaggle Seite pruefen koennt.

# ## Vorbereitung
# Importieren wir als erstes unsere Bibliotheken und den Trainingsdatensatz.
# Der Datensatz beinhaltet die Verkaufszahlen von Avocados der Sorte Hass aus den Jahren 2015 bis 2018. 
# 
# Diese Daten koennen aus verschiedenen Gruenden interessant sein. Beispielsweise fuer Anleger, die auf Lebensmittelpreise spekulieren (Kein Fan). Aber auch fuer Restaurants oder Supermarktketten, die daran die Beliebtheit von Avocados ablesen und ihr Sortiment auf schwankende Preise anpassen koennen.

# In[ ]:


import pandas as pd # Datensets
import numpy as np # Data Manipulation


# In[ ]:


# Read in all data
data = pd.read_csv("../input/avocado.csv")


# ## Beispielsdatensatz
# 
# Um euch ein besseres Bild zu vermitteln, wie Funktionen angewandet werden, werde ich Beispiele anhand eines kleinen Datensatzes vor jeder Aufgabe geben.
# Dieser Datensatz hat keine wirkliche Bedeutung und dient nur als Beispiel.

# In[ ]:


# Create Sample Dataset

testcolumns = ["Date", "Price", "Name", "Volume", "42"]

testdata = [
    ["01-01-2019", 100, "Redhat", 20, 42],
    ["02-01-2019", 200, "Pop", 10, 42],
    ["03-01-2019", 300, "Mint", 5, 42],
    ["04-01-2019", 400, "Arch", 2.5, 42],
    ["05-01-2019", 500, "Suse", 1.25, ]
]

testdata = pd.DataFrame(columns = testcolumns, data = testdata)


# --------------------------

# ## Aufgabe 1: Schema betrachten
# 
# Als erstes wollen wir Wissen, womit wir es zu tun haben.
# Bearbeiten wir hier einen riesigen Datensatz oder nur einen kleinen Auszug?
# 
# Je nach Groesse muss man anders mit Daten umgehen.
# 
# Untersuche die Datensaetze auf ihr Schema.
# Wie viele Spalten und Zeilen haben die Daten?
# Notiere deine Ergebnisse in einer Markdown Zeile.
# 
# Trage die Anzahl an Zeilen in Ergebnis 1 am Ende des Notebooks ein.
# Trage die Anzahl der Spalten als Ergebnis 2 ein.
# 
# Tipp:

# Verwenden die pandas .shape funktion https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shape.html

# In[ ]:


# ToDo: Read out schema of the Dataset


# ## Aufgabe 2: Daten betrachten
# Bevor man gross Daten analysiert, macht es Sinn sich ersteinmal die Daten mit blossem Auge und unveraendert anzuschauen. Wenn man versteht, was in den Daten steht, laesst sich viel einfacher damit arbeiten. Schaut euch die Daten genauer an. Was steht in den Spalten drin? 
# 
# Lese zuerst die Namen aller Spalten aus. 
# 
# Lass dir dann die ersten zehn Eintraege im Datensatz anzeigen.
# 
# ### Trage den Namen der letzten Spalte (als String) als Ergebnis drei am Ende des Notebooks ein.
# 
# ### Trage die Summe von AveragePrice der ersten zehn Eintraege als Ergebnis 4 ein. (gerundet auf zwei Nachkommstellen)
# 
# ### Trage das Datum des hoechsten Preises (als String) als Ergebnis 5 ein.
# 
# Tipp:

# Verwende die pandas Attribute/Funktionen
# ```
# .columns # shows list columns
# 
# .head(<Count>) # Shows first <Count> rows in Dataset
# 
# ```
# .colums gibt eine Liste aus. Mithilfe von eckigen Klammern kann man das gewuenschte Element waehlen.

# In[ ]:


# This is how the testdata looks:
print(testdata.columns.values)
testdata.head()


# Jetzt seid Ihr an der Reihe!

# In[ ]:


# ToDo: print column names of data

# ToDo: print last column name

# ToDo: print first 10 rows of data


# ---------------------------------

# ## Aufgabe 3: Datensatz bereinigen

# Eine der Hauptaufgaben bei der Datenaufbereitung (engl. "Data Wrangling") ist das Bereinigen der Daten. Datensaetze aus der Wirtschaft sind nur in den seltensten Faellen sauber und schoen aufbereitet wie dieser hier. Oftmals fehlen signifikante Mengen der Daten, die Spaltennamen sind kryptisch und ein Grossteil der Daten ist redundant oder einfach falsch. Das ist aktuell vermutlich die groesste Herausforderung im Bereich von Data Analytics, Data Science und Machine Learning. 
# Wenn wir uns nicht auf die Daten verlassen koennen, koennen wir auf deren Basis auch keine Entscheidungen treffen.
# 
# Mit der Funktion 
# ```
# datensatz.isnull().sum() 
# ```
# 
# koennen wir herausfinden, wie viele Daten in der Tabelle fehlen.
# Diese sollten wir moeglichst entfernen, damit die Analyseergebnisse spaeter nicht verfaelscht werden.
# Ausserdem haben wir noch eine Spalte mit dem Namen "Unnamed: 0". Diese ist wohl ein ueberbleibsel aus einem alten Speichervorgang. Die steht nur im Weg und liefert uns keine Informationen.
# 
# Loesche zuerst die unnoetige Zeile aus dem Datensatz.
# Schaue dir dann die Anzahl fehlender Daten an.
# Loesche anschliessend alle Zeilen mit fehlenden Daten.
# 
# ### Geben Sie die neue Anzahl an Zeilen als Ergebnis 6 an.
# 
# Tipp:

# Mit .drop() koennt ihr Zeilen oder Spalten loeschen.
# mit axis bestimmt ihr, ob ihr Zeilen oder Spalten loescht.
# Um eine Spalte mit dem Namen "Spalte" zu loeschen, wuerde das wie folgt aussehen:
# ```
# datensatz.drop(["Spaltenname"], axis = 1)
# ```
# 
# Um leere Spalten zu loeschen koennt Ihr .dropna (steht fuer drop not available) verwenden.
# mit dem attribute "how" bestimmt ihr was geloescht werden soll.
# ```
# datensatz.dropna(how = "any") # Loescht alle Zeilen die fehlende Daten enthalten
# datensatz.dropna(how="all") # Loescht Zeilen in denen ALLE Spalten leer sind.
# ```
# 
# Mehr Infos zum loeschen von Zeilen: https://www.kaggle.com/aliendev/example-of-pandas-dropna

# In[ ]:


# Example
testdata = testdata.drop(["Name"], axis = 1)
print("missing data\n", testdata.isnull().sum())
testdata = testdata.dropna(how = "any")
testdata.head()


# In[ ]:


# ToDo: delete "Unnamed: 0"

# ToDo: Read out missing data

# ToDo: delete rows with missing Data


# ---------------------------------

# ## Aufgabe 4: Daten transformieren

# Um spaeter ordentlich mit den Daten arbeiten zu koennen, muessen Sie erst aufbereitet werden.
# Ein Datum, das haeufig Schwierigkeiten bereitet ist das kalendarische Datum. Das wird je nach Region anders geschrieben und als Text gespeichert.
# Gluecklicherweise bringt Pandas ein Tool mit, dass Datumsangaben problemlos aufbereitet und interpretiert, solange das Format nicht absolut kaputt ist.
# Dafuer gibt es die pandas Datetime Funktionen. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html 
# 
# Hole dir mit 
# ```
# "data.Date.describe()" 
# ```
# eine uebersicht ueber die aktuellen Datumsangaben
# 
# Wandel die Datumsangaben in der Spalte "Date" in echte Datumsangaben um verwende dafuer pd.to_Datetime(~Daten).
# 
# ```
# datensatz["Datumspalte"] = pd.to_Datetime(datensatz["Datumspalte"])
# ```
# 
# Schaue dir danach nochmal die uebersicht mit "data.Date.describe()" an.
# Ihr koennt jetzt viel mehr ueber die Daten ablesen, da der Dataframe und die Datumsangaben als solche Interpretieren kann.
# 
# Gib die komplette Angabe von "last" in der uebersicht als String in Ergebnis 7 an.
# 

# In[ ]:


print("before transformation\n", testdata["Date"].head())
testdata["Date"] = pd.to_datetime(testdata["Date"])
print("after transformation\n", testdata["Date"].head())


# In[ ]:


# Overview of Date Data


# In[ ]:


# Convert Date to Datetime


# In[ ]:


# New Overview of Data


# ---------------------------------

# ## Aufgabe 5: Features extrahieren

# Eine fortgeschrittene Technik der Datenmodelierung ist Feature Extraction. Das umfasst alle Aktivitaeten, die neue Daten aus der Verarbeitung, Kombination oder Anreicherung von Daten in Datensaetzen beinhalten. Feature Extraction bindet das Wissen des Analysten in die Daten mit ein und generiert dadurch reichere Datensaetze. Weil die Informationen oftmals bereits vorhanden sind, spricht man hier von der Extraktion.
# 
# In unserem Avocado Datensatz koennen wir noch einige interessante Informationen herausholen.
# Zum einen den Gesamtumsatz am angegebenen Tag und den inflationsbereinigten Preis. 
# 
# Gluecklicherweise wissen wir, dass die Inflation seit 2015 im Durchschnitt 2.015% betrug. Der Einfachhalt halber rechnen wir alle Durchschnittspreise auf das Jahr 2018 hoch.
# 
# Trage den bereinigten Preis am 29.11.2015 in Albany als Ergebnis 8 ein (drei Nachkommastellen).
# 
# Tipp:

# Mit:
# 
# ```
# datensatz["neuer Spaltenname"] = xyz
# ```
# 
# Koennen wir neue Spalten hinzufuegen.
# 
# Eine neue Spalte kann auch die Kombination anderer Saplten sein:
# ```
# datensatz["summe"] = datensatz["A"] + datensatz["B"] # Addiert Spalten A und B zur Spalte Summe
# datensatz["halbiert"] = datensatz["A"] / 2 # Halbiert alle Daten in Spalte A und speichert sie in einer neuen Spalte "halbiert"
# ```
# 
# 
# 
# 

# In[ ]:


# example
testdata["meaningless"] = testdata["Price"] + testdata["Volume"]
testdata["just dividing by 5"] = testdata["Volume"] / 5
testdata.head()


# In[ ]:


# ToDo: Add "Total Sales" to Dataset(Price * Volume)

# ToDo: Add "real Price" to Dataset (Price corrected by inflation)


# ----------------------------

# # Ergebnisse ueberpruefen
# Trage hier deine Ergebnisse ein.
# Das kleine Script unten spuckt eine CSV Datei aus, die Ihr als "prediction" hochladen koennt.

# In[ ]:


# 42 zu ersetzen:
a1 = [1, 42] #Ergebnis 1
a2 = [2, 42] #Ergebnis 2
a3 = [3, "42"] #Ergebnis 3
a4 = [4, 42] #Ergebnis ...
a5 = [5, "42"]
a6 = [6, 42]
a7 = [7, "42"]
a8 = [8, 42]
antworten = [a1,a2,a3,a4,a5,a6, a7, a8]
meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Category"])
meine_antworten.to_csv("meine_loesung_Aufgaben1.csv", index = False)

