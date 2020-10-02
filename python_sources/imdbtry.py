import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

file_path = "../input/" + check_output(["ls", "../input"]).decode("utf8").strip()

df = pd.read_csv(file_path)

for col in df.columns:
    print (col)

gString = ""
for genere in df['genres']:
    gString += "|" + genere
    
# print (df['director_name'])

# for i, dirName in enumerate(df['director_name']):
#     if dirName == "Zack Snyder":
#         print (df[')
    
unique_generes = list(set(gString.strip("|").split("|")))