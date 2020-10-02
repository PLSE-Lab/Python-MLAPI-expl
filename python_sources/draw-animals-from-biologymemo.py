import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap 
import nltk

#import goslate

ship_data = pd.read_csv('../input/CLIWOC15.csv')
biology_data = ship_data[pd.notnull(ship_data.BiologyMemo)][['Nationality','BiologyMemo']].drop_duplicates()
#gs = goslate.Goslate()
#for animal in english_animals:
#    animals[gs.translate(animal, 'nl')] = animals[animal]
#    animals[gs.translate(animal, 'fr')] = animals[animal]
#    animals[gs.translate(animal, 'es')] = animals[animal]
# as there is no goslate module available in kaggle scripts, just insert the names of animals
animals = {'turtle': 'o', u'ping\xfcinos': 'p', 'shark': 'sh', u'requins': 'sh', u'canards': 'o', 'serpent': 'o', u'vogelstand': 'o', u'vogel': 'o', u'dauphins': 'd', 'dolphin': 'd', 'gulls': 'o', 'seals': 'o', u'canard': 'o', u'zeeschildpad': 'o', u'haai': 'sh', u'Grampa': 'd', u'delfines': 'd', 'seal': 'o', u'pingu\xefn': 'p', u'mouette': 'o', u'Sellos': 'o', 'birds': 'o', 'duck': 'o', 'sharks': 'sh', u'baleines': 'w', u'Abuelo': 'd', u'tibur\xf3n': 'sh', u'meeuw': 'o', u'gaviotas': 'o', u'Orcas': 'd', 'whales': 'w', u'ballenas': 'w', u'manchot': 'p', 'gull': 'o', u'pingouins': 'p', u'tortue': 'o', u'eend': 'o', u'haaien': 'sh', u'walvis': 'w', u'oiseaux': 'o', 'Grampuse': 'd', 'penguin': 'p', u'dauphin': 'd', 'bird': 'o', u'p\xe1jaros': 'o', u'afdichtingen': 'o', u'requin': 'sh', u'eenden': 'o', u'delf\xedn': 'd', u'tortugas': 'o', u'pingu\xefns': 'p', 'bonetta': 'o', u'zegel': 'o', u'Tortuga': 'o', u'joint': 'o', u'dolfijnen': 'd', 'albotross': 'o', u'p\xe1jaro': 'o', u'baleine': 'w', u'gaviota': 'o', u'ping\xfcino': 'p', 'whale': 'w', 'ducks': 'o', u'ballena': 'w', u'pato': 'o', 'dolphins': 'd', u'schildpadden': 'o', u'tortues': 'o', 'turtles': 'o', u'oiseau': 'o', 'Grampuses': 'd', u'meeuwen': 'o', u'serpiente': 'o', u'walvissen': 'w', u'Bonetta': 'o', u'mouettes': 'o', 'penguins': 'p', u'phoques': 'o', u'patos': 'o', u'foca': 'o', u'tiburones': 'sh', u'dolfijn': 'd'}
indexes_of_animals = []
code_of_animals = []
for index, row in biology_data.iterrows():
    try: 
        tokens = nltk.word_tokenize(row.BiologyMemo)
        taged_tokens = nltk.pos_tag(tokens)
        for tag_token in taged_tokens:
            if tag_token[1] == 'NNS' or tag_token[1] == 'NN':
                if tag_token[0] in animals:#animals[tag_token[0]]:
                    indexes_of_animals.append(index)
                    code_of_animals.append(animals[tag_token[0]])
    except: pass
lon = ship_data.Lon3[indexes_of_animals]
lat = ship_data.Lat3[indexes_of_animals]
coord = np.column_stack((list(lon),list(lat)))
code_of_animals = np.array(code_of_animals)
mask = ~np.isnan(coord).any(axis=1)
code_of_animals = code_of_animals[mask]
coord = coord[mask]

m = Basemap(projection='robin',lon_0=180,resolution='c', llcrnrlon=120, urcrnrlon=-30)
m.drawcoastlines()
m.drawcountries()
m.drawmeridians(np.arange(0,360,30))
m.drawparallels(np.arange(-90,90,30))
m.fillcontinents(color='grey')
#draw path on the background
#draw dolphins
x,y=m(coord[code_of_animals == 'd', 0], coord[code_of_animals == 'd', 1])
m.plot(x, y, '.', color='blue', label='dolphins')
#draw whales
x,y=m(coord[code_of_animals == 'w', 0], coord[code_of_animals == 'w', 1])
m.plot(x, y, '.', color='cyan', label='whales')
#draw sharks
x,y=m(coord[code_of_animals == 'sh', 0], coord[code_of_animals == 'sh', 1])
m.plot(x, y, '.', color='red', label='sharks')
#draw penguins
x,y=m(coord[code_of_animals == 'p', 0], coord[code_of_animals == 'p', 1])
m.plot(x, y, '.', color='magenta', label='penguins')
#draw others
x,y=m(coord[code_of_animals == 'o', 0], coord[code_of_animals == 'o', 1])
m.plot(x, y, '.', color='green', label='others')
plt.legend(bbox_to_anchor=(0.4, 0), loc=2, borderaxespad=0.)
plt.savefig("animals.png")
plt.show()
