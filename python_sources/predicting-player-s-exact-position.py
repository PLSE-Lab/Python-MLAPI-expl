# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from unidecode import unidecode
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


df = pd.read_csv('../input/CompleteDataset.csv')



def convertToFloat(df_col):
    result = []
    units = {"K":1000,"M":1000000}
    for val in df_col:
            try:
                result.append( float(val) )  #try to comber it to a number
            except ValueError:
                unit=val[-1]                 #get the letter
                val = float( val[:-1] )        #convert all but the letter
                result.append( val * units[unit] )
    
    df_col = [ value for value in result]
    return df_col
    
def fieldPositions(atk,mid,defn,df,i):
    if atk == 0 and mid == 0 and defn == 0:
        return 'GK'
    elif atk>mid and atk>defn:
        return 'Attack'
    elif mid>=atk and mid>= defn:
        return 'Midfield'
    elif defn>mid and defn>atk:
        return 'Defense'
    elif atk == mid and mid > defn:
        for pos in attack:
            if df['Preferred Positions'][i] == pos:
                return 'Attack'
        return 'Midfield'
    elif defn == mid and mid > atk:
        for pos in defense:
            if df['Preferred Positions'][i] == pos:
                return 'Defense'
        return 'Midfield'
    elif atk == defn and atk > mid:
        for pos in attack:
            if df['Preferred Positions'][i] == pos:
                return 'Attack'
        return 'Defense'
    elif atk != 0 and mid != 0 and defn != 0 and atk == mid and atk == defn and mid == defn :
        return 'Midfield'    
    else:
        return 'Unknown'
# Step 1 : Read the dataset
df = pd.read_csv('../input/CompleteDataset.csv')

columns=['ID', 'Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club',
       'Value', 'Wage', 'Height_cm', 'Weight_kg', 'Acceleration', 'Aggression',
       'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve',
       'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving',
       'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes',
       'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing',
       'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions',
       'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed',
       'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys', 'CAM',
       'CB', 'CDM', 'CF', 'CM', 'LAM', 'LB', 'LCB', 'LCM', 'LDM', 'LF', 'LM',
       'LS', 'LW', 'LWB', 'RAM', 'RB', 'RCB', 'RCM',
       'RDM', 'RF', 'RM', 'RS', 'RW', 'RWB', 'ST','Preferred Positions',
       'FieldPositions', 'Position']

# list of attributes which are needed for predictions
attrs=['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure',
   'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy',
   'GK diving', 'GK handling', 'GK kicking', 'GK positioning',
   'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots',
	'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 
   'Sliding tackle', 'Sprint speed',	'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys'] 

# re-arranged dataframe
df=pd.DataFrame(df,columns=columns)
df=df.head(n=100)
print("--------------------Step 1 DONE !!! ---------------------------")

# Step 2 : Convert accented characters to normal form.
for i in list(range(len(df))):
    name = unidecode(str(df['Name'][i]))
    club = unidecode(str(df['Club'][i]))
    df['Name'][i] = name
    df['Club'][i] = club
    
df["Value"] = df["Value"].str.split('€').str.get(1)
df["Value"]=convertToFloat(df["Value"])
df["Wage"] = df["Wage"].str.split('€').str.get(1)
df["Wage"]=convertToFloat(df["Wage"])    
print("--------------------Step 2 DONE !!! ---------------------------")
# Step 3 : Replace missing values

nan_cols = df.isna().sum()      # get columns which contain NaN values along with count
nan_cols = nan_cols.to_dict()   
nans = []                       
for key, value in nan_cols.items():
    if value != 0 :
        nans.append(key)        # Inserting columns those contains NaN

cat=["Name","Nationanlity","Club","FieldPositions", "Continent","Position"]
for var in cat:
    for n in nans:
        if n == var:
            df[var]=df[var].fillna("unknown")
            nans.remove(var)
            break

for i in range(len(nans)):
    df[nans[i]] = df[nans[i]].fillna(0)
  
print("--------------------Step 3 DONE !!! ---------------------------") 
# step 4 : make each observations in each column uniform and convert into appropriate data type
            
for attr in attrs:
    if is_string_dtype(df[attr]):
        for i in range(len(df)):
            df[attr][i] = eval(df[attr][i])
                        
for attr in attrs:
    df[attr] = pd.to_numeric(df[attr])
   
print("--------------------Step 4 DONE !!! ---------------------------")    

# Step 5 : assign fieldPosition
df['Atk']=0
df['Mid']=0
df['Def']=0
attack = [ 'ST', 'CF', 'RF', 'LF', 'RW', 'LW', 'RS', 'LS' ] 
midfield = [ 'CAM', 'RAM', 'LAM', 'CM', 'RCM', 'LCM', 'LM', 'RM', 'CDM', 'RDM', 'LDM' ]
defense = [ 'CB', 'RB', 'LB', 'RCB', 'LCB', 'RWB', 'LWB' ]
    
for i in list(range(0,len(df))):
    atk=0
    mid=0
    defn=0
    temp = []
    for pos in attack:
        temp.append(int(df[pos][i]))    
    atk = int(np.mean(temp))
    
    temp = []
    for pos in midfield:
        temp.append(int(df[pos][i]))     
    mid = int(np.mean(temp))
    
    temp = []
    for pos in defense:
        temp.append(int(df[pos][i]))     
    defn = int(np.mean(temp))
    
    df['Atk'][i]=atk
    df['Mid'][i]=mid
    df['Def'][i]=defn 
    df['FieldPositions'][i] = fieldPositions(atk,mid,defn,df,i)

print("--------------------Step 5 DONE !!! ---------------------------")
# step 6: from preferred position obtain best exact position.
    
df["Preferred Positions"] = df["Preferred Positions"].str.split()
for i in list(range(len(df.head(n=15)))):
    positions_values = []
    if len(df["Preferred Positions"][i])>1:
        for pos in df["Preferred Positions"][i]:
            positions_values.append(df[pos][i])
        df['Position'][i] = df["Preferred Positions"][i][ positions_values.index(max(positions_values))]
    else:
        df['Position'][i] = df["Preferred Positions"][i][0]

print("--------------------Step 6 DONE !!! ---------------------------")
# Any results you write to the current directory are saved as output.

# Predicitng where player is suitable to play, Attack Midfield or Defense
        
columns = ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing',
           'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling',
           'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 
           'Jumping', 'Long passing', 'Long shots',	'Marking', 'Penalties', 'Positioning', 
           'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed',	'Stamina',
           'Standing tackle', 'Strength', 'Vision', 'Volleys','FieldPositions']

df1 = pd.DataFrame(df,columns = columns)
df1_x = df1.ix[:,:34]
df1_y = df1.ix[:,-1]

x_train, x_test, y_train, y_test = train_test_split(df1_x,df1_y,train_size = 0.8, random_state =123 )

model=XGBClassifier(n_estimators=100,nthread=4)
model = model.fit(x_train,y_train)
output = model.predict(x_test)

print("Accuracy: ")
print(accuracy_score(y_test, output)*100)


label=["Attack","Midfield","Defense","GK"]

cm = confusion_matrix(y_test, output,labels=label)
print(cm)

print("--------------------Step 7 DONE !!! ---------------------------")


# Predicting Exact field position

columns = ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing',
           'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling',
           'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 
           'Jumping', 'Long passing', 'Long shots',	'Marking', 'Penalties', 'Positioning', 
           'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed',	'Stamina',
           'Standing tackle', 'Strength', 'Vision', 'Volleys','Position']

df2 = pd.DataFrame(df,columns = columns)
df2_x = df2.ix[:,:34]
df2_y = df2.ix[:,-1]

x_train, x_test, y_train, y_test = train_test_split(df2_x,df2_y,train_size = 0.8, random_state =123 )
xgb_param = model.get_xgb_params()
xgb_param['num_class'] = 14
model=XGBClassifier(n_estimators=125,objective="multi:softmax",nthread=4)
model = model.fit(x_train,y_train)
output = model.predict(x_test)

print("Accuracy: ")
print(accuracy_score(y_test, output)*100)

print("--------------------Step 8 DONE !!! ---------------------------")