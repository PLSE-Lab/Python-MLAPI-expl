import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
import datetime
import matplotlib.dates as mdates
# download data
ship_data = pd.read_csv('../input/CLIWOC15.csv')
# look at the slaves info in Cargo field
memo_data = ship_data[pd.notnull(ship_data.Cargo)][['CargoMemo','LifeOnBoardMemo','OtherRem']].drop_duplicates()
indexes = []
# find records that have slaves words in them 
slave_words = {'slave',  'slaves', 'slaaf', 'slaven', 'meisjesslaaf', 'manslaaf', 'manslaven', 
               'slavenjong','jongensslaaf', 'meidslaaf', 'servant',
               'slavenmeid', 'vrouwslaaf', 'vrouwslaven', 'slavenhandel', 'slaaf',
               'esclavo', 'esclavos', 'esclave', 'esclaves'}
#gs = goslate.Goslate()
for index, row in memo_data.iterrows():
    try: 
        sentence = row.CargoMemo #+ " " + row.LifeOnBoardMemo + " " + row.OtherRem
        tokens = nltk.word_tokenize(sentence)
        tokens = [w.lower() for w in tokens]
        for token in tokens:
            if token in slave_words:
                indexes.append(index)
                break
    except: pass
slave_data = ship_data.iloc[indexes].drop_duplicates()
ship_names = slave_data.ShipName.unique()
# extracting number of died slaves from records
died = {name : [] for name in ship_names}
die_words = {'died', 'passed', 'deceased','overleden'}
for index, row in slave_data.iterrows():
    try: 
        sentence = row.CargoMemo 
        ship = row.ShipName
        sentence_english = sentence #gs.translate(row.CargoMemo, 'en')
        # In Dutch logs, if a slave dies, the record has substring = "no. 'number of deaths'"
        if ("no." in sentence_english) or ("No." in sentence_english) or ("Nos." in sentence_english):
            numbers = re.findall('no..\d+-\d+|no..\d+|No..\d+|No..\d+-\d+|Nos..\d+-\d+', sentence_english)
            numbers = [re.findall('\d+', number) for number in numbers]
            tokens_english = nltk.word_tokenize(sentence_english)
            tokens_english = [w.lower() for w in tokens_english]
            for token in tokens_english:
                if token in die_words:
                    died[ship].append((index, numbers))
                    break
    except: pass
# print the numer of records about slaves for each ship
print (slave_data.ShipName.value_counts())
# make some data transformations
slave_data = {name : [] for name in ship_names}
for name in ship_names:
    died_slaves = died[name]
    for death in died_slaves:
        index = death[0]
        number_of_deaths = death[1][0]
        slave_data[name].append((name, ship_data.VoyageIni.iloc[index], \
                        datetime.date(ship_data.Year[index], ship_data.Month[index], ship_data.Day[index]), \
                        int(number_of_deaths[-1]), ship_data.CargoMemo[index]))
# print out info for, example, the first ship 
name = ship_names[0]
ship_slaves = pd.DataFrame.from_records(slave_data[name], columns=["name", "voyage_id", "date", "number_of_slaves_died", "memo"]).drop_duplicates()
ship_slaves = ship_slaves.sort(['date'])
# print info about slaves on the ship 
print(ship_slaves)
# make a plot of slaves' deaths
voyages = ship_slaves.voyage_id.unique()
for voyage in voyages:
    voyage_slaves =  ship_slaves[ship_slaves.voyage_id == voyage]
    if voyage_slaves.shape[0] != 0:
        fig = plt.figure(figsize=(20, 5))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.plot(voyage_slaves.date, voyage_slaves.number_of_slaves_died)
        plt.plot(voyage_slaves.date, voyage_slaves.number_of_slaves_died, 'o')
        plt.title("ShipName = " + name)
        plt.xlabel("Date")
        plt.ylabel("Number of slaves' deaths")
        plt.gcf().autofmt_xdate()
        plt.savefig("pic.png")