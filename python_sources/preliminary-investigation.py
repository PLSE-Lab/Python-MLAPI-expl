import numpy as np
import pandas as pd

#---- Prepare some data types
dtype = {"Age" : np.float64,
         "Sex" : pd.Categorical,
         "Embarked" : pd.Categorical}

#___________________
#---- Function to extract the title of the passenger
def GetTitle(df):
    fullName = df["Name"]
    #---- Assume full name in the form: Bloggs, Mr. Joe Alex
    allTitles = [name for name in fullName.split() if "." in name]
    if len(allTitles) != 1:
        print("An odd name",fullName)
        return fullName
    return allTitles[0]
    
#___________________
#---- Helper function to read the csv file
def ReadCSV(fileName):
    df = pd.read_csv(fileName, dtype=dtype, )
    df["Name"] = df.apply(GetTitle,axis=1)
    df["Name"] = df["Name"].astype('category')
    return df

#___________________
if __name__ == "__main__":

    #---- Read the csv files
    train = ReadCSV("../input/train.csv")
    test = ReadCSV("../input/test.csv")
    
    #---- 
    print("\n\nTop of the training data:")
    print(train.head())

    print("\n\nSummary statistics of training data")
    print(train.describe())
    
    #----
    print("\n\nUnique passenger titles are")
    print(train["Name"].value_counts())
    
    #---- Some ideas:
    #Convert name-->len(name), sex-->factor, embarked->factor
    #Look at patterns in ticket name vs survival
    #Look at patterns in cabin name (eg letter-number)
    
    #Any files you save will be available in the output tab below
    #train.to_csv('copy_of_the_training_data.csv', index=False)