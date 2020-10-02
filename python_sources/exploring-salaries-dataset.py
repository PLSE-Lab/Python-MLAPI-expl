
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk as nl



salaries = pd.read_csv("../input/Salaries.csv")
salaries.drop(['Benefits','Notes','Agency', 'Status'],axis=1,inplace=True)

for year in salaries['Year'].unique():
    print("Number of records in " + str(year) + ": " + str(len(salaries[salaries['Year']==year])))

salaries_2014 = salaries[salaries['Year']==2014]
jobnames = salaries_2014['JobTitle']

            
    
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


jobword = ""
for word in jobnames:
     jobword+=word
        
tokens=tokenizer.tokenize(jobword)
print(tokens)


           
                         
                       




