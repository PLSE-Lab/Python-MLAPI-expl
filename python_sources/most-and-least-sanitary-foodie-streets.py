import pandas as pd

df = pd.read_csv("../input/Restaurant_Scores_-_LIVES_Standard.csv")

def splitstr(r):
    st = r['business_address']
    return st.split(" ")[-2] if len(st.split(" ")) > 2 else None

df['street'] = df.apply(splitstr, axis='columns')
srs = df.groupby('street')['inspection_score'].median()
srs = srs[df['street'].value_counts() > 25].sort_values(ascending=False)

# Select the top 10 and bottom 10 correctly parsed street names.
poi = pd.concat(
    [srs.iloc[[0, 1, 2, 3, 5, 6, 7, 9, 11, 13]],
     srs.iloc[range(-10, 0, 1)]]
)
poi.index = [n.title() for n in poi.index]

import matplotlib.pyplot as plt
poi.plot.bar(figsize=(10, 5), fontsize=12)
fig = plt.gcf()
fig.subplots_adjust(bottom=0.3)
plt.axvline(x=9.5)
plt.savefig("output.png")