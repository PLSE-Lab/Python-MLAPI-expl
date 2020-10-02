import datetime
import numpy as np
import pandas as pd
import collections

from learntools.core import *

class Step1(EqualityCheckProblem):
    _var = 'dataframe'
    _hint = ""
    _solution = CS(
    """
dataframe = pd.DataFrame()    
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]
    """)

    def check(self, df):

        # Create data frame
        dataframe = pd.DataFrame()

        # Add three columns: 'Name' (string), 'Age' (number) and 'Driver' (boolean)
        dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
        dataframe['Age'] = [38, 25]
        dataframe['Driver'] = [True, False]
        
        
        assert df.equals(dataframe), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(dataframe, df)          

        
        
class Step2(EqualityCheckProblem):
    _var = 'new_dataframe'
    _hint = ""
    _solution = CS(
    """
row = pd.Series(['Molly Mooney', 40, True], index=['Name','Age','Driver'])
new_dataframe = dataframe.append(row, ignore_index=True)
    """)

    def check(self, df):

        # Create data frame
        dataframe = pd.DataFrame()

        # Add three columns: 'Name' (string), 'Age' (number) and 'Driver' (boolean)
        dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
        dataframe['Age'] = [38, 25]
        dataframe['Driver'] = [True, False]

        # Create row
        row = pd.Series(['Molly Mooney', 40, True], index=['Name','Age','Driver'])

        # Append row
        new_dataframe = dataframe.append(row, ignore_index=True)        
        
        assert df.equals(new_dataframe), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(new_dataframe, df)          

        
        
       
class Step3(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
dataframe = pd.read_csv("https://tinyurl.com/titanic-csv")
rows = dataframe.head(5)
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Select the first five rows
        rows = dataframe.head(5)     
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)          

       
class Step4(EqualityCheckProblem):
    _var = 'dimensions'
    _hint = ""
    _solution = CS(
    """
dimensions = dataframe.shape
    """)

    def check(self, val):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")
        
        # Show dimensions
        dimensions = dataframe.shape
        
        assert dimensions == val, ("Se esperaban el valor: {}, "
                "pero se encontró el valor: {}").format(dimensions, val)          


       
class Step5(EqualityCheckProblem):
    _var = 'statistics'
    _hint = ""
    _solution = CS(
    """
statistics = dataframe.describe()
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")
        
        # Show statistics
        statistics = dataframe.describe()
        
        assert df.equals(statistics), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(statistics, df)        


class Step6(EqualityCheckProblem):
    _var = 'row'
    _hint = ""
    _solution = CS(
    """
row = dataframe.iloc[0]  
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Select first row
        row = dataframe.iloc[0]  
        
        assert df.equals(row), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(row, df)      



class Step7(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
rows = dataframe.iloc[1:4]
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Select the second, third, and fourth row
        rows = dataframe.iloc[1:4]
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


        

class Step8(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
rows = dataframe.iloc[:4]
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Select all rows up to and including the fourth row
        rows = dataframe.iloc[:4]
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


class Step9(EqualityCheckProblem):
    _var = 'row'
    _hint = ""
    _solution = CS(
    """
dataframe = dataframe.set_index(dataframe['Name'])
row = dataframe.loc['Allen, Miss Elisabeth Walton']
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Set 'Name' columnn as index
        dataframe = dataframe.set_index(dataframe['Name'])

        # Select row where 'Name' value is 'Allen, Miss Elisabeth Walton'
        row = dataframe.loc['Allen, Miss Elisabeth Walton']
        
        assert df.equals(row), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(row, df)      




class Step10(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
rows = dataframe[dataframe['Sex'] == 'female'].head(2)
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

         # Select top two rows where column 'Sex' is 'female'
        rows = dataframe[dataframe['Sex'] == 'female'].head(2)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step11(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
rows = dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Filter rows where the passenger is a 'female' and 'Age' is 65 or older
        rows = dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      




class Step12(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe['Sex'].replace("female", "Woman")
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Replace "female" in 'Sex' column with "Woman"  
        new_dataframe = dataframe['Sex'].replace("female", "Woman")

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


class Step13(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"])
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Replace "female" and "male" with "Woman" and "Man"
        new_dataframe = dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"])

        # Select the five first rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step14(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.replace(1, "One")
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Replace value 1 with "One" in the whole data frame
        new_dataframe = dataframe.replace(1, "One")

        # Select the five firt rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      



class Step15(EqualityCheckProblem):
    _var = 'rows'
    _hint = ""
    _solution = CS(
    """
new_dataframe = dataframe.replace(r"1st", "First", regex=True)
    """)

    def check(self, df):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Replace "1st" with "First" using regular expressions
        new_dataframe = dataframe.replace(r"1st", "First", regex=True)

        # Select the five firt rows
        rows = new_dataframe.head(5)
        
        assert df.equals(rows), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(rows, df)      


class Step16(EqualityCheckProblem):
    _var = 'names_array'
    _hint = ""
    _solution = CS(
    """
i = 0
names = ['', '']
for name in dataframe['Name'][0:2]:
    names[i] = name.upper()
    i = i + 1
    """)

    def check(self, na):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Find first two names uppercased using loops
        i = 0
        names = ['', '']
        for name in dataframe['Name'][0:2]:
            names[i] = name.upper()
            i = i + 1

        names_array = np.array(names)
        
        assert np.array_equal(names_array, na), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(names_array, na)      



class Step17(EqualityCheckProblem):
    _var = 'names_array'
    _hint = ""
    _solution = CS(
    """
names = [name.upper() for name in dataframe['Name'][0:2]]
    """)

    def check(self, na):

        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Find first two names uppercased using use list comprehensions
        names = [name.upper() for name in dataframe['Name'][0:2]]

        names_array = np.array(names)
        
        assert np.array_equal(names_array, na), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(names_array, na)      



class Step18(EqualityCheckProblem):
    _var = 'names_array'
    _hint = ""
    _solution = CS(
    """
def uppercase(x):
    return x.upper()

names = dataframe['Name'].apply(uppercase)[0:2]
    """)

    def check(self, na):
        
        # Create function to uppercase a string
        def uppercase(x):
            return x.upper()
        
        # Load data from a csv file as a data frame
        dataframe = pd.read_csv("/kaggle/input/titanic/titanic.csv")

        # Apply function to first two names
        names = dataframe['Name'].apply(uppercase)[0:2]
        
        names_array = np.array(names)
        
        assert np.array_equal(names_array, na), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(names_array, na)  
            




class Step19(EqualityCheckProblem):
    _var = 'new_dataframe'
    _hint = ""
    _solution = CS(
    """
new_dataframe = pd.concat([dataframe_a, dataframe_b], axis=0)
    """)

    def check(self, df):

        # Create first data frame
        data_a = {'id': ['1', '2', '3'],
                  'first': ['Alex', 'Amy', 'Allen'],
                  'last': ['Anderson', 'Ackerman', 'Ali']}
        dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

        # Create second data frame
        data_b = {'id': ['4', '5', '6'],
                  'first': ['Billy', 'Brian', 'Bran'],
                  'last': ['Bonder', 'Black', 'Balwner']}
        dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

        # Concatenate data frame by rows
        new_dataframe = pd.concat([dataframe_a, dataframe_b], axis=0)
        
        assert df.equals(new_dataframe), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(new_dataframe, df)      


        
class Step20(EqualityCheckProblem):
    _var = 'new_dataframe'
    _hint = ""
    _solution = CS(
    """
new_dataframe = pd.concat([dataframe_a, dataframe_b], axis=1)
    """)

    def check(self, df):

        # Create first data frame
        data_a = {'id': ['1', '2', '3'],
                  'first': ['Alex', 'Amy', 'Allen'],
                  'last': ['Anderson', 'Ackerman', 'Ali']}
        dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

        # Create second data frame
        data_b = {'id': ['4', '5', '6'],
                  'first': ['Billy', 'Brian', 'Bran'],
                  'last': ['Bonder', 'Black', 'Balwner']}
        dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

        # Concatenate data frame by columns
        new_dataframe = pd.concat([dataframe_a, dataframe_b], axis=1)
        
        assert df.equals(new_dataframe), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(new_dataframe, df)      


      
class Step21(EqualityCheckProblem):
    _var = 'new_dataframe'
    _hint = ""
    _solution = CS(
    """
row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])
new_dataframe = dataframe_a.append(row, ignore_index=True)
    """)

    def check(self, df):

        # Create first data frame
        data_a = {'id': ['1', '2', '3'],
                  'first': ['Alex', 'Amy', 'Allen'],
                  'last': ['Anderson', 'Ackerman', 'Ali']}
        dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

        # Create row
        row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])

        # Create data frame appending row to first data frame 
        new_dataframe = dataframe_a.append(row, ignore_index=True)
        
        assert df.equals(new_dataframe), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(new_dataframe, df)      

      
class Step22(EqualityCheckProblem):
    _var = 'new_dataframe'
    _hint = ""
    _solution = CS(
    """
new_dataframe = pd.merge(dataframe_employees, dataframe_sales, on='employee_id')
    """)

    def check(self, df):

        # Create first data frame
        employee_data = {'employee_id': ['1', '2', '3', '4'],
                         'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                         'Tim Horton']}
        dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id', 'name'])

        # Create second data frame
        sales_data = {'employee_id': ['3', '4', '5', '6'],
                      'total_sales': [23456, 2512, 2345, 1455]}
        dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id','total_sales'])

        # Merge data frames
        new_dataframe = pd.merge(dataframe_employees, dataframe_sales, on='employee_id')
        
        assert df.equals(new_dataframe), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(new_dataframe, df)      

      
class Step23(EqualityCheckProblem):
    _var = 'new_dataframe'
    _hint = ""
    _solution = CS(
    """
new_dataframe = pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')
    """)

    def check(self, df):

        # Create first data frame
        employee_data = {'employee_id': ['1', '2', '3', '4'],
                         'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                         'Tim Horton']}
        dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id', 'name'])

        # Create second data frame
        sales_data = {'employee_id': ['3', '4', '5', '6'],
                      'total_sales': [23456, 2512, 2345, 1455]}
        dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id','total_sales'])

        # Merge data frames on 'employee_id' column using 'outer' join type
        new_dataframe = pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')
        
        assert df.equals(new_dataframe), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(new_dataframe, df)      


      
class Step24(EqualityCheckProblem):
    _var = 'new_dataframe'
    _hint = ""
    _solution = CS(
    """
new_dataframe = pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')
    """)

    def check(self, df):

        # Create first data frame
        employee_data = {'employee_id': ['1', '2', '3', '4'],
                         'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                         'Tim Horton']}
        dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id', 'name'])

        # Create second data frame
        sales_data = {'employee_id': ['3', '4', '5', '6'],
                      'total_sales': [23456, 2512, 2345, 1455]}
        dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id','total_sales'])

        # Merge data frames on 'employee_id' column using 'left' join type
        new_dataframe = pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')
        
        assert df.equals(new_dataframe), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(new_dataframe, df)      

      
class Step25(EqualityCheckProblem):
    _var = 'new_dataframe'
    _hint = ""
    _solution = CS(
    """
new_dataframe = pd.merge(
    dataframe_employees,
    dataframe_sales,
    left_on='employee_id',
    right_on='employee_id'
)
    """)

    def check(self, df):

        # Create first data frame
        employee_data = {'employee_id': ['1', '2', '3', '4'],
                         'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                         'Tim Horton']}
        dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id', 'name'])

        # Create second data frame
        sales_data = {'employee_id': ['3', '4', '5', '6'],
                      'total_sales': [23456, 2512, 2345, 1455]}
        dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id','total_sales'])

        # Merge data frames on 'employee_id'
        new_dataframe = pd.merge(
            dataframe_employees,
            dataframe_sales,
            left_on='employee_id',
            right_on='employee_id'
        )
        
        assert df.equals(new_dataframe), ("Se esperaban los elementos: [{}], "
                "pero se encontraron los elementos: [{}]").format(new_dataframe, df)      


        
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
    Step24,
    Step25
    ],
    tutorial_id=199,
    var_format='step{n}',
    )
__all__ = list(qvars)