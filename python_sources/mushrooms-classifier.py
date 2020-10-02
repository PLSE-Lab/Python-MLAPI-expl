import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import sklearn
import sklearn.preprocessing


data=pd.read_csv('../input/mushrooms.csv',index_col=False)

data['class'] = data['class'].map( {'p': 0, 'e': 1} ).astype(int)
data['cap-shape'] = data['cap-shape'].map( {'b': 0, 'c': 1, 'x': 2, 'f': 3, 'k': 4, 's' : 5 } ).astype(int)
data['cap-surface'] = data['cap-surface'].map( {'f': 0, 'g': 1, 'y': 2, 's' : 3} ).astype(int)
data['cap-color'] = data['cap-color'].map( {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'r': 4, 'p': 5, 'u': 6, 'e': 7, 'w': 8, 'y': 9} ).astype(int)
data['bruises'] = data['bruises'].map( {'f': 0, 't': 1} ).astype(int)
data['odor'] = data['odor'].map( {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8} ).astype(int)
data['gill-attachment'] = data['gill-attachment'].map( {'a': 0, 'd': 1, 'f': 2, 'n': 3} ).astype(int)
data['gill-spacing'] = data['gill-spacing'].map( {'c': 0, 'w': 1, 'd': 2} ).astype(int)
data['gill-size'] = data['gill-size'].map( {'n': 0, 'b': 1} ).astype(int)
data['gill-color'] = data['gill-color'].map( {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'g' : 4, 'r' : 5, 'o' : 6, 'p' : 7, 'u' : 8, 'e' : 9, 'w' : 10, 'y' : 11} ).astype(int)
data['stalk-shape'] = data['stalk-shape'].map( {'t': 0, 'e': 1} ).astype(int)
# missing values in stalk-root 
data['stalk-root'] = data['stalk-root'].map( {'b': 0, 'c': 1, 'u': 2, 'e': 3, 'z': 4, 'r': 5, '?' : -1} ).astype(int)
data['stalk-surface-above-ring'] = data['stalk-surface-above-ring'].map( {'f': 0, 'y': 1, 'k': 2, 's': 3} ).astype(int)
data['stalk-surface-below-ring'] = data['stalk-surface-below-ring'].map( {'f': 0, 'y': 1, 'k': 2, 's': 3} ).astype(int)
data['stalk-color-above-ring'] = data['stalk-color-above-ring'].map( {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o' : 4, 'p' : 5, 'e' : 6, 'w' : 7, 'y' : 8} ).astype(int)
data['stalk-color-below-ring'] = data['stalk-color-below-ring'].map( {'n': 0, 'b': 1, 'c': 2, 'g': 3, 'o' : 4, 'p' : 5, 'e' : 6, 'w' : 7, 'y' : 8} ).astype(int)
data['veil-type'] = data['veil-type'].map( {'p': 0, 'u': 1} ).astype(int)
data['veil-color'] = data['veil-color'].map( {'n': 0, 'o': 1, 'w': 2, 'y': 3} ).astype(int)
data['ring-number'] = data['ring-number'].map( {'n': 0, 'o': 1, 't': 2} ).astype(int)
data['ring-type'] = data['ring-type'].map( {'c': 0, 'e': 1, 'f': 2, 'l': 3, 'n' : 4, 'p' : 5, 's' : 6, 'z' : 7} ).astype(int)
data['spore-print-color'] = data['spore-print-color'].map( {'k': 0, 'n': 1, 'b': 2, 'h': 3, 'r' : 4, 'o' : 5, 'u' : 6, 'w' : 7, 'y' : 8} ).astype(int)
data['population'] = data['population'].map( {'a': 0, 'c': 1, 'n': 2, 's': 3, 'v' : 4, 'y' : 5} ).astype(int)
data['habitat'] = data['habitat'].map( {'g': 0, 'l': 1, 'm': 2, 'p': 3, 'u' : 4, 'w' : 5, 'd' : 6} ).astype(int)

#data= [[np.nan   if x == '?' else  x for x in y] for y in data]



target=data['class'][0:7000]
target2=data['class'][7000:]

train=data[['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color',
'ring-number','ring-type','spore-print-color','population','habitat']][:7000].values

test=data[['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color',
'ring-number','ring-type','spore-print-color','population','habitat']][7000:].values

names=['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color',
'ring-number','ring-type','spore-print-color','population','habitat']


rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)

print("Accuraccy is: " + str(rf.score(test,target2)))


([print(names[i] +': ' + str(rf.feature_importances_[i]) ) for i in range(len(names))])
