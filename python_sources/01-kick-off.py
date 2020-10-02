#!/usr/bin/env python
# coding: utf-8

# # Kick-off
# ----------------------------------------

# Mit den Grundlagen aus den Learnpython Kursen habt Ihr jetzt alle Grundlagen, die noetig sind um ein paar Datenanalysen in Python durchzufuehren.
# Dafuer werdet ihr diese Python Notebooks auf Kaggle verwenden.
# 
# Fuer jede Worksession und die Arbeit dazwischen gibt es ein eigenes Notebook, in denen Ihr eure Aufgaben finden werdet.
# Am Ende des Kurses werdet Ihr eure Abgabe auch in einem Notebook erstellen und abgeben.
# Fuer jedes Team muss nur ein Notebook abegegben werden.

# # Ich will nicht Coden, sondern Managen!
# ----------------------------------------------
# Kann ich verstehen, aber Niemand wird als Leader geboren. Wer die richtigen Entscheidungen trifft, bekommt auch die Verantwortung uebertragen. Und dafuer muss man zuerst verstehen, was man ueberhaupt tut. Deswegen lernen wir hier ein weiteres Werkzeug kennen, das uns den Weg an die Spitze ebnet!
# 
# Darum wird hier auch nicht eure Leistung im Programmieren beurteilt, sondern nur ob Ihr euch zu helfen wisst die richtigen Antworten in den Daten zu finden. In eurer endgueltigen Abgabe bitte ich nur darum, dass der Code sich ausfuehren laesst. Sonst kann ich eure Schritte nicht nachvollziehen ;)

#  

# # Jupyter Notebooks
# -----------------------------------------
# Jupyter Notebooks sind ein Tool, dass vor allem von Data Scientists und Data Analysts verwendet wird. 
# Sie bieten eine Programmierumgebung, vor Allem fuer python und R, die auf Visualisierung und Prototyping spezialisiert ist.
# Sie heissen Notebooks, weil Sie, wie ein Notizbuch in z.B. Mathe, eine Mischung aus Rechnungen/Funktionen und Notizen beinhalten.
# 
# Jupyter Notebooks bestehen grundaetzlich aus zwei Elementen:
# 
# ### Kernel --> Interpretiert den Inhalt (fuer uns nicht relevant)
# 
# ### Zellen --> Inhalt des Notebooks

# ## Zellen
# Zellen stellen den Inhalt unsere Notebooks dar. 
# Sie koennen in zwei Formen verwendet werden:
# 
# 1. Code
# 
# und
# 
# 2. Markdown
# 
# 
# Das kann ueber den Dropdown oben in der Titelleiste angepasst werden.
# In Code Zellen wird Python Code geschrieben.
# Beim Aktivieren der Zelle wird dieser Code ausgefuehrt.

# In[ ]:





# **Fett**

#  

# In[1]:


# This is a code cell


# --------------------------
# 
# Das hier ist eine Markdown Zelle
# 
# --------------------------

# ## ueberschriften und Listen
# 
# In Markdown Zellen kommt Text. Beim Ausfuehren wird der Text formatiert.
# Diese Beschreibung ist ebenfalls in einer Markdown Zelle.
# Bei Markdown kann man mithilfe von ein paar Tricks formatieren:
# 
# (# Heading)
# # Heading
# 
# 
# (## SubHeading)
# ## SubHeading
# 
# (### SubSubHeading)
# ### SubSubHeading
# 
# 
# (- List)
# - List
# 
# (1. Enumeration) 
# 1. Enumeration
# 
# 
# -------------------------------------------------------------------------------------------------------

# 

# ## Zellen Logik
# 
# Die Logik in Notebooks ist, dass einzelne Aufgaben in eigenen Zellen abgehandelt werden.
# Zellen koennen nacheinander in einer beliebigen Reihenfolge ausgefuehrt werden.
# Dadurch lassen sich kleine modulare Bausteine aus Code und Text bauen, die sich immer wieder verwenden lassen.
# 
# Man probiert im Code etwas aus und beschreibt den Inhalt mit Notizen.

#  

# # Best Practices fuer Arbeit in Notebooks
# ------------------------------------
# Notebooks sind eines der beliebtesten Tools fuer Data Science, da sie rapid prototyping ermoeglichen und durch Visualisierung das Verstaendnis der Daten unterstuetzen. Jedes Notebook besteht aus Zellen, die entweder Code oder Markdown-text beinhalten koennen.
# Jedes Notebook kann beliebig viele Zellen enthalten.

# ## Zellen ausfuehren
# Um den Code in einer Zelle auszufuehren koennt Ihr auf den blauen Pfeil auf der linken Seite der Zelle klicken oder einfach mit shift + enter.
# 
# Das Ergebnis eures codes wird unterhalb der Zelle ausgegeben.
# Hat eure Zelle keine Ausgabe, wird auch nichts ausgegeben.

# In[2]:


# Execute this cell with shift + enter
print("Es hat funktioniert")


# In[8]:


# Kommentar
print("Hallo")
print(5)


#  

# ## Zellen aufteilen
# Beim erstellen von Zellen ist es hilfreich den Code sinnvoll aufzuteilen.
# Dabei solltet Ihr auch sinnvoll kommentieren und Beschreibungstexte einfuegen.
# 
# Beispielsweise sollten alle Importe in einer Zelle ganz am Anfang des Notebooks stehen
# 
# 

# In[3]:


# All Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from mpl_toolkits.mplot3d import Axes3D # 3D visualization
import os # file system access


# ## Kommentieren
# 
# In der naechsten Zelle definieren wir eine einfache Funktion zur Verwendung weiter unten im Notebook.
# Diese ist mit Kommentaren beschriftet. Kommentieren ist manchmal etwas laestig, aber absolut unabdingbar.
# Kommentarbloecke werden mit drei Apostrophen gekennzeichnet.
# ```
# '''
# Hier kommen Kommentare rein
# Diese sind mehrzeilig
# '''
# ```
# 
# Mit "#" koennen Kommentare in einer Zeile definiert werden.

# In[9]:


def print_some_things(things, num):
    '''
    ---------------------------------------------
    This function prints things a number of times
    
    things --> array/list of objects to print

    num --> integer: amount of times to print
    
    ---------------------------------------------
    '''
    
    # Loop over num
    for i in range(num):
        # print the content of things to console
        print(i, "\t",things)


# In der naechsten Zelle koennen wir die Funktion verwenden:

# In[10]:


things = [1, 2, 3, 42]
print_some_things(things, 42)


# ---------------------------------

# # Warum benutzen wir Notebooks?
# 
# Notebooks haben einen entscheidenden Vorteil.
# Sie bieten "rich output". Also Output der viele Informationen enthaelt und in der Regel visuell dargestellt ist.
# Fuer Analysts und Data Scientist ist das besonders wichtig, da wir Muster vor allem visuell erkennen koennen.
# 
# Folgend einige Beispiele, wie Notebooks Daten und Informationen gut darstellen koennen:

# ## DataFrames
# Pandas DataFrames bieten die Moeglichkeit effizient und schnell mit Daten zu arbeiten.
# Sie sind perfekt in die Notebooks integriert und eines der wichtigsten Werkzeuge eines Analysts.

# In[11]:


import pandas as pd

#define Data and structure
columns = ["OEM", "Modell", "Price", "TFlops"]

data = [
    ["Apple", "Macbook Pro", 2200, 1],
    ["Apple", "Macbook", 1400, 0.8],
    ["Dell", "XPS 15", 1900, 1.1],
    ["Lenovo", "Yoga", 1100, 0.6],
    ["HP", "Elitebook", 1600, 0.9],
    ["HP", "X360", 1900, 1.1],
    ["Asus", "Zephyrus", 2900, 1.5],
    ["Microsoft", "Surface Pro", 1800, 0.9]
]

# build dataframe
dataframe = pd.DataFrame(columns = columns, data = data)

# display dataframe
dataframe.head(10)


# In[12]:


# only show OEMs
dataframe.OEM


# In[13]:


# count how often every OEM is included
dataframe.OEM.value_counts()


# ## Grafiken und Plots
# Mit Matplotlib lassen sich tolle Grafiken erstellen, die euch helfen ein gutes Notebook zu bauen.
# Auf der [API-Seite](https://matplotlib.org/gallery/index.html) findet ihr tolle Beispiele fuer Visualisierungen inklusive Code.
# Ihr duerft natuerlich auch andere Visualisierungspakete wie seaborn verwenden.
# Hier ein Beispiel:

# In[14]:


import matplotlib.pyplot as plt # data visualization
# define data to plot
x = dataframe.Price
y = dataframe.TFlops

# create canvas
fig, ax = plt.subplots()

# plot data
ax.scatter(x, y)

# plot figure to console
plt.show()


# So schnell kann man sich Daten darstellen lassen.
# 

# In[15]:


# Random Values for Plot
x = np.random.uniform(1,0,200)
y = np.random.uniform(1,0,200)
z = np.random.uniform(1,0,200)

x2 = np.random.uniform(1.5,0,200)
y2 = np.random.uniform(1.5,0,200)
z2 = x2 + y2

# create canvas
fig, ax = plt.subplots()

# plot data
ax.scatter(x, y)

# set some visualizations
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_title('random plot')

# plot figure to console
plt.show()


# In[16]:


# create 3D canvas
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot data
ax.scatter(x, y, z, c="blue")
ax.scatter(x2, y2, z2, c = "red")

# name axis
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("y", fontsize=15)
ax.set_zlabel("z", fontsize=15)
ax.set_title('also random')

# plot figure to console
plt.show()


# ## Praesentieren in Notebooks
# Zu Praesentationszwecken koennen Zellen versteckt werden.
# So lassen sich Notebooks auch zu Praesentationszwecken verwenden.

# # Beispiel: Ergebnisse CRM Analyse
# Die Analyse der Kundendaten hat ergeben, dass langjaehrige Kunden immer seltener unsere Produkte weiterempfehlen.
# Am Graphen ist eindeutig ersichtlich, dass Kunden mit unseren Produkten unzufrieden sind und diese Unzufriedenheit immer weiter steigt.
# Das deckt sich ebenfalls mit den Kundenbefragungen aus dem Marketing.

# In[17]:


import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

fig = sns.jointplot(x,y, kind="hex", color="#4CB391")
fig.set_axis_labels('Customer Retention time', 'Willingness to rcommend', fontsize=16)
fig


# Aufgrund dessen empfehlen wir folgende Schritte:
# 1. Neues Entwicklungsteam fuer komplettes redesign einstellen
# 2. Agile Entwicklung einfuehren
# 3. Lead-User finden und einbinden

# # Abgabe von Ergebnissen:
# Ihr koennt zwischen den Worksession eure Abgaben ueberpruefen.
# Dafuer wird am Ende der Notebooks ein Template fuer euch bereitstehen.
# Dort koennt Ihr dann eure Ergebnisse eintragen.
# 
# Beispiel:

# In[18]:


# 42 zu ersetzen:
a1 = [1, 42]
a2 = [2, 42]
a3 = [3, 42]
a4 = [4, 42]
a5 = [5, 42]
a6 = [6, 42]
antworten = [a1,a2,a3,a4,a5,a6]
meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Category"])
meine_antworten.to_csv("meine_loesung.csv", index = False)


# In[ ]:




