import numpy as np
import pandas as pd
import csv as csv

#Print you can execute arbitrary python code
#train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

csv_file_object = csv.reader(open('../input/train.csv', 'rU')) 
header = next(csv_file_object) 
data=[] 

for row in csv_file_object:
    data.append(row)
data = np.array(data)

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('../input/train.csv', header=0)
#print(df.head(3))
#print(df.dtypes)
#print(df.info())
#print(df.describe())

#print(df['Age'][0:10])
#print(df.Age[0:10])
#print(df.Age[0:10].mean())

print(df[ ['Age','Sex','Pclass'] ])