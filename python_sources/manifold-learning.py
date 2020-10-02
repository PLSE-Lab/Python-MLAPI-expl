import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

events = pd.read_csv('../input/events.csv')
app_events = pd.read_csv('../input/app_events.csv')
gender_age_train = pd.read_csv('../input/gender_age_train.csv')
#app_labels = pd.read_csv('../input/app_labels.csv', nrows=100)
#phone_brand_device_model = pd.read_csv('../input/phone_brand_device_model.csv', nrows=100)

df = pd.merge(gender_age_train, events, on='device_id')
df = pd.merge(df, app_events, on='event_id')

print(df.head())

#from sklearn.manifold import TSNE
#X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
#tsne = TSNE(n_components=2)
#points = tsne.fit_transform(X)

#import matplotlib.pyplot as plt
#plt.scatter(X[:,0], X[:,1])
#plt.savefig('plot.png')
