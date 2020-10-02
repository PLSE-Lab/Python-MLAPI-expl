import datetime
import numpy as np
import pandas as pd
import collections

from learntools.core import *




class Step1(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.rename(columns={'PClass': 'Passenger Class'})
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Rename column 'PClass' to 'Passenger Class'
        new_dataframe = dataframe.rename(columns={'PClass': 'Passenger Class'})

        # Select the five firt rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step2(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'})
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Rename columns 'PClass' to 'Passenger Class' and 'Sex' to 'Gender'
        new_dataframe = dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'})

        # Select the five firt rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


class Step3(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
names = ['PClass', 'Sex'] 
new_names = ['Passenger Class', 'Gender']
dictionary = dict(zip(names, new_names))
new_dataframe = dataframe.rename(columns=dictionary)
    """)

    def check(self, df):

        # Load data from a csv file as a data frame 
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")
        
        # Create keys
        names = ['PClass', 'Sex'] 

        # Create values    
        new_names = ['Passenger Class', 'Gender']

        # Rename columns 'PClass' to 'Passenger Class' and 'Sex' to 'Gender'    
        dictionary = dict(zip(names, new_names))


        # Rename values from dictionary
        new_dataframe = dataframe.rename(columns=dictionary)

        # Select the five firt rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


       
class Step4(EqualityCheckProblem):
    _var = 'mean_value'
    _hint = ""
    _solution = CS(
    """
mean_value = dataframe['Age'].mean()
    """)

    def check(self, val):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")
        
        # Show mean in 'Age' column
        mean_value = dataframe['Age'].mean()
        
        assert mean_value == val, ("Se esperaban el valor: {}, "
                "pero se encontró el valor: {}").format(mean_value, df)          



class Step5(EqualityCheckProblem):
    _var = 'count_values'
    _hint = ""
    _solution = CS(
    """
count_values = dataframe.count()
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Show count values to the whole data frame
        count_values = dataframe.count()
        
        assert df.equals(count_values), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(count_values, df)      


 
class Step6(EqualityCheckProblem):
    _var = 'values'
    _hint = ""
    _solution = CS(
    """
values = dataframe['Sex'].unique()
    """)

    def check(self, na):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Find unique values in 'Sex' column
        values = dataframe['Sex'].unique()
        
        assert np.array_equal(na, values), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(values, na)      

       

class Step7(EqualityCheckProblem):
    _var = 'values'
    _hint = ""
    _solution = CS(
    """
values = dataframe['Sex'].value_counts()
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Find unique values with the number of times each value appears in 'Sex' column
        values = dataframe['Sex'].value_counts()
        
        assert df.equals(values), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(values, df)      


class Step8(EqualityCheckProblem):
    _var = 'classes'
    _hint = ""
    _solution = CS(
    """
classes = dataframe['PClass'].value_counts()
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Find number of classes in 'PClass' column
        classes = dataframe['PClass'].value_counts()

        assert df.equals(classes), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(classes, df)      


class Step9(EqualityCheckProblem):
    _var = 'count'
    _hint = ""
    _solution = CS(
    """
count = dataframe['PClass'].nunique()
    """)

    def check(self, val):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Find number of unique values in 'PClass' column
        count = dataframe['PClass'].nunique()

        assert val == count, ("Se esperaba el valor: {}, "
                "pero se encontró el valor: {}").format(count, val)      



class Step10(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe[dataframe['Age'].isnull()]
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Find missing values in 'Age' column
        new_dataframe = dataframe[dataframe['Age'].isnull()]

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step11(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe= dataframe['Sex'].replace('male', np.nan)
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Replace all strings containing 'male' with NaN in 'Sex' column
        dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)
        new_dataframe= dataframe['Sex'].replace('male', np.nan)

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


class Step12(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.drop('Age', axis=1)
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Delete 'Age' column
        new_dataframe = dataframe.drop('Age', axis=1)

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step13(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.drop(['Age', 'Sex'], axis=1)
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Drop 'Age' and 'Sex' columns
        new_dataframe = dataframe.drop(['Age', 'Sex'], axis=1)

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


class Step14(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.drop(dataframe.columns[1], axis=1)
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Drop the second column
        new_dataframe = dataframe.drop(dataframe.columns[1], axis=1)

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


class Step15(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe[dataframe['Sex'] != 'male']
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Delete rows containing 'male' in 'Sex' column 
        new_dataframe = dataframe[dataframe['Sex'] != 'male']

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step16(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine']
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Delete rows containing 'Allison, Miss Helen Loraine' in 'Name' column 
        new_dataframe = dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine']

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


class Step17(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe[dataframe.index != 0]
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Delete first row using its index
        new_dataframe = dataframe[dataframe.index != 0]

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step18(EqualityCheckProblem):
    _var = 'len_new_dataframe'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.drop_duplicates()
len_dataframe = len(dataframe)
len_new_dataframe = len(new_dataframe)
    """)

    def check(self, val):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Drop duplicates, show first two rows of output
        new_dataframe = dataframe.drop_duplicates()

        # Show number of rows
        len_dataframe = len(dataframe)
        len_new_dataframe = len(new_dataframe)

        assert val == len_new_dataframe, ("Se esperaba el valor: {}, "
                "pero se encontró el valor: {}").format(len_new_dataframe, val)      



class Step19(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.drop_duplicates(subset=['Sex'])
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Drop duplicates in 'Sex' column
        new_dataframe = dataframe.drop_duplicates(subset=['Sex'])

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step20(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.drop_duplicates(subset=['Sex'], keep='last')
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Drop duplicates in 'Sex' column
        new_dataframe = dataframe.drop_duplicates(subset=['Sex'], keep='last')

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step21(EqualityCheckProblem):
    _var = 'mean_values'
    _hint = ""
    _solution = CS(
    """
mean_values = dataframe.groupby('Sex').mean()
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Group rows by the values of the column 'Sex' and calculate mean of each group
        mean_values = dataframe.groupby('Sex').mean()
        
        assert df.equals(mean_values), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(mean_values, df)      



class Step22(EqualityCheckProblem):
    _var = 'values_count'
    _hint = ""
    _solution = CS(
    """
values_count = dataframe.groupby('Survived')['Name'].count()
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Group rows into survived or not and count the number of names in each group
        values_count = dataframe.groupby('Survived')['Name'].count()
        
        assert df.equals(values_count), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(values_count, df)      


class Step23(EqualityCheckProblem):
    _var = 'values_mean'
    _hint = ""
    _solution = CS(
    """
values_mean = dataframe.groupby(['Sex','Survived'])['Age'].mean()
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Group rows 'Sex' and'Survived', calculate mean in 'Age' column
        values_mean = dataframe.groupby(['Sex','Survived'])['Age'].mean()
        
        assert df.equals(values_mean), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(values_mean, df)      

class Step24(EqualityCheckProblem):
    _var = 'values'
    _hint = ""
    _solution = CS(
    """
values = dataframe.groupby('Sex').apply(lambda x: x.count())
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Group rows in 'Sex' column and apply count function to groups
        values = dataframe.groupby('Sex').apply(lambda x: x.count())
        
        assert df.equals(values), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(values, df)      




        
qvars = bind_exercises(globals(), [
    Step1,
    Step2,
    Step3,
    Step4,
    Step5,
    Step6,
    Step7,
    Step8,
    Step9,
    Step10,
    Step11,
    Step12,
    Step13,
    Step14,
    Step15,
    Step16,
    Step17,
    Step18,
    Step19,
    Step20,
    Step21,
    Step22,
    Step23,
    Step24
    ],
    tutorial_id=199,
    var_format='step{n}',
    )
__all__ = list(qvars)