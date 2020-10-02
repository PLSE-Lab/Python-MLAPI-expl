import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

list_ = []
for file_ in ["../input/pums/ss13husa.csv"]:
    df = pd.read_csv(file_)
    list_.append(df)
df = pd.concat(list_)

df[["TYPE"]].plot(kind="bar")
print(df[["TYPE", "ACCESS", "ACR"]].head())
plt.savefig("Hist.png")