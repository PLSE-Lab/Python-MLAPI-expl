# This Python 3 environment comes with many helpful analytics libraries installed
# Code Written By Darien Schettler
# June 1, 2018

# This is code I wrote when I was going through literature related to numpy
# It is the second in a series I will be doing covering numpy, pandas, scipy, and matplotlib
# I hope this is helpful to people

# You can run the code as it stands now and it will explain what is happening and the code to do it
# As such you can get an understanding of the commands just from running it without having to scroll through

# It has multiple "please press enter to continue..." input commands to help break up the learning
# This method of stopping doesn't seem to work well with the kernel system on kaggle as such they are commented out...
# Enjoy

import numpy as np
import pandas as pd

print("\nPandas lets us work with, and manipulate, data structures (DataFrames and Series mostly) "
      "even letting us perform statistical and financial analysis...\n\n")

print("A Series is the primary building block in Pandas and it represents a 1D LABELLED Numpy array\n\n")

#input("Press Enter to continue...")

print("\n\nWe will start with creating a series from a list-->"
      "a =[1,3,5,6] --> pd.Series(a) --> index = ['A1','A2','A3','A4']\n")

# Creating a series from a list --- note we could do this directly with pd.Series([1,3,5,6])
a = [1, 3, 5, 6]
a = pd.Series(a)
print("Here is the simple array A")
print(a)

print("\nHere is the elements of the index")
index_a = ['A1', 'A2', 'A3', 'A4']
print(index_a)

print("\nNow we combine them by using a.index = index_a")
a.index = index_a
print(a)

#input("\n\nPress Enter to continue...")

print("\n\nWe will move onto creating a Series from a Numpy array\n")

print("Lets create a numpy array of random integers from 95 to 100")
a = np.random.randn(100) * 5 + 100
print("\n\nNew Numpy Array A")
print(a)

print("\nLets now create something to index our random numbers -->"
      " We will utilize the pd.date_range command to create 100 sequential dates\n"
      "This is done using the command pd.date_range('20180101', periods=100)\n")
date = pd.date_range('20180101', periods=100)
print("\nHere's the array of index elements")
print(date)

print("\n\nNow we combine the two as shown above using pd.Series(a,index=date)")
s = pd.Series(a, index=date)
print(s)

#input("\n\nPress Enter to continue...")

print("\n\nWe will move onto creating a Series from a Dictionary (keys and values)\n")
print("Lets create a dictionary of a = {'A1':5,'A2':3,'A3':6,'A4':2}")
a = a = {'A1': 5, 'A2': 3, 'A3': 6, 'A4': 2}
print("\n\nNew Dictionary A")
print(a)
print("\na.keys() followed by a.values()\n")
print(a.keys())
print(a.values())

print("\n\nNow we simply equate it and create a series using s = pd.Series(a)")
s = pd.Series(a)
print(s)

#input("\n\nPress Enter to continue...")

print("\n\nNow we look at arithmetic for a Series noting it matches the index values\n")
print("We will use two Series, a and b\n"
      "a = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])\n"
      "b = pd.Series([4, 3, 2, 1], index=['d', 'c', 'b', 'a'])")
a = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
b = pd.Series([4, 3, 2, 1], index=['d', 'c', 'b', 'a'])
print("\nSeries A")
print(a)
print("\nSeries B")
print(b)

#input("\n\nPress Enter to continue...")

print("\n\nNow we perform simple addition, subtraction, multiplication, and division noting index calls")

print("\n---Addition (note how the first value of array A interacts with the last value of array B"
      "due to their matching index---")
print(a + b)
#input("\n\nPress Enter to continue...")

print("\n---Subtraction---")
print(a - b)
#input("\n\nPress Enter to continue...")

print("\n---Multiplication---")
print(a * b)
#input("\n\nPress Enter to continue...")

print("\n---Division---")
print(a / b)
#input("\n\nPress Enter to continue...")

print("\n\nWe can now look into some series attributes such as .index, .values, and len()\n")
print("We will use the previous A Array")
print(a)

#input("\n\nPress Enter to continue...")

print("\nShow the indices")
print(a.index)
print("\nShow the values")
print(a.values)
print("\nShow the number of elements")
print(len(a))

#input("\n\nPress Enter to continue...")

print("\n\nWe will now look at viewing and selecting data from a Series\n\n")
print("The command head/tail give the first/last rows specified to the number in the brackets"
      " i.e. the command head(8) will give the first 8 rows of a series")
# Note that you can simply use .tail() or .head() with a df or Series and it will use the default number of rows (5)
print("\nHere is a head of 2 for our previous A array")
print(a.head(2))
print("\nHere is a tail of 3 for our previous A array")
print(a.tail(3))

#input("\n\nPress Enter to continue...")

print("\nTo select data based on the label we use a['b']")
print(a['b'])
#input("\n\nPress Enter to continue...")

print("\nTo select data based on the integer index we use a[1]")
print(a[2])
#input("\n\nPress Enter to continue...")

print("\nTo select multiple pieces of data based on the label we use a[['b', 'd']]")
print(a[['b', 'd']])
#input("\n\nPress Enter to continue...")

print("\nTo select multiple pieces of data based on the integer index we use a[[1,3]]")
print(a[[1, 3]])
#input("\n\nPress Enter to continue...")

print("\nNote that a[1] will only produce the number 2 but a[[1]] will produce the label and the value  a & 1")
print(a[1])
print(a[[1]])
#input("\n\nPress Enter to continue...")

print("\nWe can also 'slice' data using traditional matrix/vector indexing [0:3], [1:], [:2]")
print(a[0:3])
print(a[1:])
print(a[:2])
print("Notice again that when slicing we get the index AND the value even though we only used one bracket\n")

#input("\n\nPress Enter to continue...")

print("{CHALLENGE} Slice out the data from 2018-01-05 to 2018-01-10 given"
      "a Pandas series of 20 random numbers indexed sequentially from 20180101 {CHALLENGE}")
date = pd.date_range('20180101', periods=20)
s = pd.Series(np.random.randn(20), index=date)
print("\nHere is the pd Series")
print(s)
print("\nOne way to do it will be simply to index by number s[4:10]\n")
print(s[4:10])
print("\nWhile another way would be to use the labels s['2018-01-05':'2018-01-10']\n")
print(s['2018-01-05':'2018-01-10'])

#input("\n\nPress Enter to continue...")

print("We now move onto DataFrames which are similar to spreadsheets in excel having"
      "indexed rows and named columns\n")

print("We will create a simple data frame using d = [[1,2],[3,4]] and the command --> "
      "df = pd.DataFrame(d,index=[1,2], columns=['a','b'])\n")

d = [[1, 2], [3, 4]]
df = pd.DataFrame(d, index=[1, 2], columns=['a', 'b'])
print(df)

#input("\n\nPress Enter to continue...")

print("We will create a DF from a NP array using d = np.arange(24).reshape(6,4) and the command --> "
      "df = pd.DataFrame(d,index=np.arange(1,7), columns=list('ABCD'))\n")

d = np.arange(24).reshape(6, 4)
df = pd.DataFrame(d, index=np.arange(1, 7), columns=list('ABCD'))
print(df)
print("\nNote that we can use np.arange and list('BLAH') to shorten our code")

#input("\n\nPress Enter to continue...")

print("We will create a DF from a Dictionary using...\n"
      "d = {'name': ['Ally','Jane','Belinda'],'height':[160,155,163]} "
      "\n... and the command ...\n"
      "df = pd.DataFrame(d,index=['A1', 'A2', 'A3'], columns=list('Name','Height'))\n")

d = {'Name': ['Ally', 'Jane', 'Belinda'], 'Height': [160, 155, 163]}
df = pd.DataFrame(d, index=(['A1', 'A2', 'A3']), columns=(['Name', 'Height']))
print(df)

#input("\n\nPress Enter to continue...")

print("Finally we will learn the most common way which is creating a DF from a Series\n"
      "We will start with creating two random number series with indexed matching dates\n"
      "We will then combine them as if they were a dictionary with an appropriate heading.. in this case the Continent")

date = pd.date_range('20180101', periods=6)
s1 = pd.Series(np.random.randn(6), index=date)
s2 = pd.Series(np.random.randn(6), index=date)
df = pd.DataFrame({'Asia': s1, 'Europe': s2})
print(df)
print("\nIt's pretty much that the Series are columns and we just give them a title"
      " and the order we want and they make a DF")

#input("\n\nPress Enter to continue...")

print("The challenge is to create a DF with name, height, and age as col headers"
      " and A1 A2 A3 as row indexers and the appropriate stats...\n\n")

df = pd.DataFrame({'Name': ['Ally', 'Jane', 'Belinda'], 'Height': [160, 155, 163], 'Age': [40, 35, 42]},
                  index=(['A1', 'A2', 'A3']), columns=(['Name', 'Height', 'Age']))
print(df)
print("\n This was done by essentially writing a dictionary and mapping to a DF, there are other ways of course")

#input("\n\nPress Enter to continue...")

print("Like everything else we can gather information about our DF using commands\n"
      "We will use the commands on our previously created DF (df)\n")
print(df)

#input("\n\nPress Enter to continue...")

print("\n--- The first command is to get the dimensionality of our DF with df.shape ---")
print(df.shape)
#input("\n\nPress Enter to continue...")

print("\n--- The next command is to get the columns of our DF with df.columns ---")
print(df.columns)
#input("\n\nPress Enter to continue...")

print("\n--- The next command is to get the indices of our DF with df.index ---")
print(df.index)
#input("\n\nPress Enter to continue...")

print("\n--- The next command is to get the values of our DF with df.values ---")
print(df.values)
#input("\n\nPress Enter to continue...")

print("\nTo add to our DF we can 'append' a new column on to it using "
      "df['Hair Colour'] = ['Blonde','Brown','Red']")

df['Hair Colour'] = ['Blonde', 'Red', 'Brown']
print(df)

#input("\n\nPress Enter to continue...")
print("\nWe can do something similar by using the command "
      "df.insert(1, 'Handedness',['Left','Right','Right'])")

df.insert(1, 'Handedness', ['Left', 'Right', 'Right'])
print(df)
print("\nNotice that this column will fit itself in alphabetically within the other columns..."
      " where the previous method simply added the column to the end")

#input("\n\nPress Enter to continue...")

print("\nNow we will look at DF arithmetic using two new df as shown below")
a = [[3, 4], [5, 6]]
b = [[6, 5], [4, 3]]
a2 = pd.DataFrame(a, index=[1, 2], columns=['d', 'b'])
b2 = pd.DataFrame(b, index=[3, 2], columns=['c', 'b'])
print("\n--- Here is DataFrame a2 ---")
print(a2)
print("\n--- Here is DataFrame b2 ---")
print(b2)

#input("\n\nPress Enter to continue...")

print("\n\nNow we perform simple addition, subtraction, multiplication, and division")
print("\nYou'll note that only matching indexed and titled values will interact all others will become NA in new DF")
print("\n---Addition---"
      "due to their matching index---")
print(a2 + b2)
#input("\n\nPress Enter to continue...")

print("\n---Subtraction---")
print(a2 - b2)
#input("\n\nPress Enter to continue...")

print("\n---Multiplication---")
print(a2 * b2)
#input("\n\nPress Enter to continue...")

print("\n---Division---")
print(a2 / b2)
#input("\n\nPress Enter to continue...")

#input("\n\nPress Enter to Continue...")

print("\nNow we look at how to view and select data from a DF using a 20x5 matrix of random numbers")
print("\nHere's the A Matrix")
a = pd.DataFrame(np.random.randn(20, 5))
print(a)

print("\nSimilar to a Series, we access the data using head and tail with a number to see swathes (row based) of data")
print("\n---a.head()---")
print(a.head())
print("\n---a.head()---")
print(a.tail())

print("\nLooking at the previous dataframe (df)...\n")
print(df)

print("\n\nWe want to know how to extract data... one way is using column titles\n"
      "i.e. the command df[['Name','Height']] will print out those two columns with indices\n")
print(df[['Name', 'Height']])

print("\n\nAnother way is using a single column title in index form\n"
      "i.e. the command df.Name will print out that column with indices\n")
print(df.Name)

print("\n\nTo select row data we use similar commands except they incorporate the .ix command\n"
      "i.e. df.ix[['A1', 'A2']] or df.ix[[0,1]]\n")
print(df.ix[['A1', 'A2']])
print(df.ix[[0, 1]])

print("\n\nTo select only data from a cell/element/row/col we use the command iloc or loc...\n"
      "iloc --> will give you a specific slice of data using integer indices\n"
      "loc --> will give you a specific slice of data using labelled indices")

print("\ndf.iloc[0]")
print(df.iloc[0])

print("\ndf.loc['A1']")
print(df.loc['A1'])

print("\ndf.loc['A2']['Name'] i.e. df.loc[row][col]")
print(df.loc['A2']['Name'])

print("\ndf.iloc[1][0] i.e. df.iloc[row][col]")
print(df.iloc[1][0])

print("\nWe can also use the df.ix command with multiple brackets as above...")
print("\ndf.ix['A1']['Height']")
print(df.ix['A1']['Height'])

print("\ndf.ix[0]['Height']")
print(df.ix[0]['Height'])

print("\ndf.ix[1,1]")
print(df.ix[1, 1])

print("\ndf.ix[['A1','A2']['Name','Height']]")
print(df.ix[['A1', 'A2'], ['Name', 'Height']])

print("\n\nTo get row slices we use the .ix with a regular matrix range command df.ix[0:2]\n")
print("\ndf.ix[0:2]")
print(df.ix[0:2])

print("\ndf.ix[1:]")
print(df.ix[1:])

print("\ndf.ix[:3]")
print(df.ix[:3])
#input("\n\nPress Enter to Continue...")

print("We will now import a csv file with some data in it. The file I chose is a csv of stock data for TSLA... slimmed")
print("We first define the file name as a string variable\n")
file_name = '../input/TSLA_TICKER.csv'
print(file_name)
print("\nNext we use the pandas command .read_csv and reference the file of choice")
print("\nThis looks like tsla = pd.read_csv(file_name,index_col=?,usecols=?")
print("\nNote that we can specify a column to be our index and which coumns we want to use\n"
      "For our case we will use date as our index and take the open, close, and volume columns\n")
print("Therefore we get the command:\n"
      "tsla = pd.read_csv(file_name, index_col='date', usecols=['close', 'open', 'volume'])")
tsla = pd.read_csv(file_name, index_col='date', usecols=['date', 'close', 'open', 'volume'])
print(tsla.index)
print("\n\nTo see the last 5 days and from 5 days forward from 60 days ago (from my date)"
      " we will use .head(5) and .tail(5)\n")
print(tsla.head(5))
print(tsla.tail(5))

#input("\n\nPress Enter to Continue...")
print("Now we will print just the opens and closes by selecting those from the df\n"
      "This is done using the tsla[['open','close']] command\n")
print(tsla[['open', 'close']])

#input("\n\nPress Enter to Continue...")

print("\n\nNow we will print the mean values for the open and close data for the most"
      "recent 10 days and the most historical 10 days\n"
      "We will do this with [['open', 'close']].head(10).mean() and similarly for tail")
print("\nMean for most recent 10 days")
print(tsla[['open', 'close']].head(10).mean())
print("\nMean for most historical 10 days")
print(tsla[['open', 'close']].tail(10).mean())
print("\n\nNOTE: To open an excel file, everything remains the same except you use the pd.read_excel"
      "command as opposed to the pd.read_csv command\n")

#input("\n\nPress Enter to Continue...")

print("\n\nIf we wish to export the csv data we use the command .to_csv\n"
      "So if we wanted to export the current tsla df as a csv we would use the command -->"
      "tsla.to_csv('tsla_w_fewer_cols.csv')")

tsla.to_csv('tsla_w_fewer_cols.csv')

print("\nNow we will have an additional file in our directory named tsla_w_fewer_cols.csv")

#input("\n\nPress Enter to Continue...")

print("\n\nIt can be observed that we can filter DF and Series in the same way\n"
      "Using the commands tsla[tsla.open > 300] for instance or tsla[tsla>300]\n"
      "Note that the second example will replace values less than 300 with Nan\n"
      "Wheras if you use the first example, it will only display rows meeting the requirement")

print("\n\nHere's the first example of tsla[tsla.open>300]")
print(tsla[tsla.open > 300])

#input("\n\nPress Enter to Continue...")

print("\n\nHere's the second example of tsla[tsla>300]")
print(tsla[tsla > 300])

#input("\n\nPress Enter to Continue...")

print("\n\nSome other useful filtering techniques are the following...\n")
print("Find the day with the maximum closing value using tsla[tsla.close == tsla.close.max()]")
print(tsla[tsla.close == tsla.close.max()])

#input("\n\nPress Enter to Continue...")

print("\n\nFind values within a range by using boolean operator and -->  &\n")
print("Lets grab the days where the volume was between 7 and 8 million trades\n"
      "This would use the code --> tsla[(tsla.volume > 7000000)&(tsla.volume <= 8000000)]")

print(tsla[(tsla.volume > 7000000) & (tsla.volume <= 8000000)])

#input("\n\nPress Enter to Continue...")

print("\n\nWhat if we want closing values greater than opening values in the month of april?")
print("\nWe use --> tsla[(tsla.index > '2018-03-29') & (tsla.index <= '2018-04-30') & (tsla.close > tsla.open)]\n")
print(tsla[(tsla.index > '2018-03-29') & (tsla.index <= '2018-04-30') & (tsla.close > tsla.open)])

#input("\n\nPress Enter to Continue...")

print(
    "\n\nNow we look into handling missing data or nan data... we will first replace some pieces of data with Nans...")
print("\nTo do this we will simply specify a value range for the df and anything within it will be replaced by nan\n"
      "Using the command tsla = tsla[(tsla<300) | (tsla>304)] {note that | is or}\n"
      "In addition we will do the same thing for 304 to 305 but replace it with a blank ''")

tsla = tsla[(tsla < 300) | (tsla > 303)]
tsla[(tsla > 304) & (tsla <= 305)] = ''
print(tsla)

#input("\n\nPress Enter to Continue...")

print("\n\nA missing (nan) value is expressed as np.nan")
print(np.nan)

print("\nTo find the missing values in a df or series we use the .isna() or .isnull() commands")
print("\n\nHere is the tsla.isna() command it will have exactly the same output as tsla.isnull()")
print(tsla.isna())

#input("\n\nPress Enter to Continue...")

print("\nAt this point I like to point out that the blanks are not being picked up as nan as they are a string\n"
      "To fix this we will need to replace the blanks with the traditional np.nan using the following\n"
      "tsla[tsla==''] = np.nan\n")
tsla[tsla == ''] = np.nan
print(tsla)

#input("\n\nPress Enter to Continue...")

print("\n\nIf we now reran the tsla.isna() function it would see those places that were blank as np.nans\n")
print("Often we wish to count the missing data totals to be able to see if a column or feature is worht keeping\n"
      "To do this we use the command .isna().sum() or .isnull.sum()... they will have the same result\n")

print(tsla.isnull().sum())
print("\nNow we know we have %d missing values for close and %d missing values for open"
      % (tsla.isnull().sum()['close'], (tsla.isnull().sum()['open'])))

#input("\n\nPress Enter to Continue...")

print("\nOften we will wish to fill missing data with the 0 value as missing and 0 are often synonymous\n"
      "to do that we would use the command .fillna(0)\n")
print(tsla.fillna(0))
print("\nThis clearly does a bad job filling in OUR particular data though so we will try something else")

#input("\n\nPress Enter to Continue...")

print("\nAnother method is to fill in the nan values using column specific values\n"
      "The command for this is still .fillna but now we add a dictionary reference with value and key\n"
      "i.e. tsla.fillna({'open':0, 'close':1000})")
print(tsla.fillna({'open': 0, 'close': 1000}))

#input("\n\nPress Enter to Continue...")
print("\nAnother method would be to use the values that follow the missing nan values or "
      "the values prior to the missing nan values\n"
      "The commands would then look like...")

print("\ntsla.fillna(method='ffill)")
print(tsla.fillna(method='ffill'))

#input("\n\nPress Enter to Continue...")

print("\ntsla.fillna(method='bfill)")
print(tsla.fillna(method='ffill'))

print("These are probably the best ways to fill our particular data\n"
      "Other methods include using the mean/median/mode to fill missing data, column specific usually")

#input("\n\nPress Enter to Continue...")

print("Another way to handle missing nan elements is to just drop the row or column entirely\n"
      "This would be done with the .dropna(axis=(0 or 1)) command, the 0 axis is rows {default} and the 1 axis is cols")

print("\n\nTo drop rows of our tsla df we use tsla.dropna()\n")
print(tsla.dropna())

#input("\n\nPress Enter to Continue...")

print("\n\nTo drop cols of our tsla df we use tsla.dropna(axis=1)\n")
print(tsla.dropna(axis=1))

print("Finally we have another option which will not do anything for us, but is often a useful tool\n"
      "This is the df.dropna(how='all') command which will only drop rows if ALL the values are missing!")

print("\nOften dropping columns (features) is only a good idea if you are missing a lot of data\n"
      "It is usually better to inspect the data and insert educated guesses or model driven values\n")

#input("\n\nPress Enter to Continue...")

print("\nNext let's determine if we have any duplicates, sometimes duplicates can compromise the "
      "accuracy and/or consistency of our data\nRemoving/checking them can help prevent errors in our stats\n")
print("To find out if we have duplicates in the open column we use tsla.open.duplicated()\n")
print(tsla.open.duplicated())

#input("\n\nPress Enter to Continue...")

print("\n\nTo sum up the number of duplicates we use tsla.open.duplicated().sum()")
print(tsla.open.duplicated().sum())

#input("\n\nPress Enter to Continue...")

print("\nWe can drop rows or columns we use the .drop_duplicates() command\n"
      "i.e. tsla.open.drop_duplicates() or tsla.drop_duplicates('open')\n")

print(tsla.drop_duplicates('open'))

#input("\n\nPress Enter to Continue...")

print("\nSometimes we will need to reindex our df to better represent the data\n"
      "In this case we will replace the counting backwards date index with an index counting date backwards\n"
      "To do this we need the length of the df and then we will generate a range of values and use the command...\n"
      ".reindex(tsla.index[::-1])... [::-1] is a trick to reverse a Series\n")

print(tsla.reindex(tsla.index[::-1]))

#input("\n\nPress Enter to Continue...")

print("\nWe can also fill in nan values that were CREATED when we reindex... note that we didn't create any so "
      "this won't do anything.\n"
      "The way to reindex as we did before, with back fill would be tsla.reindex(tsla.index[::-1], method='bfill')\n")
print(tsla.reindex(tsla.index[::-1], method='bfill'))

#input("\n\nPress Enter to Continue...")

print("\nWe will now focus on joining and transforming data\n"
      "This is useful for joining data from different sources or transforming the data to a format easier to analyze\n"
      "There exists 4 methods in pandas for joining data:\n\n"
      "Method 1: CONCAT\n"
      "Method 2: APPEND\n"
      "Method 3: JOIN\n"
      "Method 4: MERGE\n\n")

#input("\n\nPress Enter to Continue...")

print("\nWe will start with the CONCAT command\n"
      "The concat command is used to join two Series\n"
      "\nWe will make two Series s1 --> pd.Series(['a','b']) and s2 --> pd.Series(['c','d'])\n")
s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['c', 'd'])

print("---s1 Array---")
print(s1)

print("\n---s2 Array---")
print(s2)

#input("\n\nPress Enter to Continue...")

print("\nThe concat command works as an assignment...\n"
      "i.e. s3 = pd.concat([s1,s2])")
s3 = pd.concat([s1, s2])
print("\ns3 = pd.concat([s1,s2])\n")
print(s3)
print("\nNotice how the index repeats itself... this is often undesirable, and as such there is a command to fix it\n"
      "This command is a parameter in the concat tool called ignore_index which is by default set to False\n"
      "We will set it to true and repeat the concatenation we just completed and print it out\n")

#input("\nPress Enter to Continue...")

print("\ns3 = pd.concat([s1,s2], ignore_index=True)\n")
s3 = pd.concat([s1, s2], ignore_index=True)
print(s3)

print("\nAnother important parameter in the concat tool is keys=[] tool which essentially allows a secondary"
      "index to identify what the index belongs too")

#input("\n\nPress Enter to Continue...")

print("\ns3 = pd.concat([s1,s2], keys=['s1','s2'])\n")
s3 = pd.concat([s1, s2], keys=['s1', 's2'])
print(s3)

print("\nThere are two other parameters which are usually only used when concating dataframes\n"
      "These are the join='' and axis='' commands\n")

print("Lets create two df using random numbers...\n")
df1 = pd.DataFrame(np.floor(np.random.randn(4, 3) * 10), index=np.arange(1, 5), columns=list('ABC'))
df2 = pd.DataFrame(np.ceil(np.random.randn(4, 3) * 8), index=np.arange(3, 7), columns=list('BCD'))

print("---df1---")
print(df1)
print("\n---df2---")
print(df2)

#input("\n\nPress Enter to Continue...")

# NOTE: that the sort=False is simply to ignore sorting warning that will appear if not set

print("\nLets try simply using pd.concat([df1,df2]) to see what happens\n")
print(pd.concat([df1, df2], sort=False))

print("\nWe notice the output follows similar rules to those that existed when using concat on Series dtypes\n"
      "As such the ignore index, and keys commands will do the same thing they did before...\n\n"
      "We will look at the join=''... there exist two options, the 'outer' vs. the 'inner' command\n"
      "The first command we try is pd.concat([df1,df2], join='inner')\n"
      "The second command we try is pd.concat([df1,df2], join='outer')")

#input("\n\nPress Enter to Continue...")

print("\n---pd.concat([df1,df2], join='inner')---")
print(pd.concat([df1, df2], join='inner', sort=False))
print("\n---pd.concat([df1,df2], join='outer')---")
print(pd.concat([df1, df2], join='outer', sort=False))

print("\nAs can be seen the inner and outer refers to a seeming ven diagram of the information to be retained\n"
      "Inner only retains data with matching keys, wheras outer retains everything and fills missing with nan")

#input("\n\nPress Enter to Continue...")

print("\nOther types of merges will have other options for this join command,for now we will move onto "
      "the axis='' command, allowing for concatenation along a particular axis\n")

print("By passing axis=1 into the concat we will combine along the x axis with no merging\n")

print(pd.concat([df1, df2], axis=1, sort=False))

print("\nBy passing axis=0 into the concat we will combine along the y axis (indices) with no merging\n")

print(pd.concat([df1, df2], axis=0, sort=False))

#input("\n\nPress Enter to Continue...")

print("\nWe now move on to the second command which is APPEND, a method for combining DFs by appending "
      "the 2nd DFs ROWS to the first DF\nignore_index may be passed as it was for concat with similar results\n")

print("df1.append(df2)")
print(df1.append(df2, sort=False))

print("\n\ndf1.append(df2, ignore_index=True)")
print(df1.append(df2, ignore_index=True, sort=False))

#input("\n\nPress Enter to Continue...")

print("\nWe now move on to the third command which is JOIN, a method for combining DFs by appending "
      "the 2nd DFs COLUMNS to the first DF\n")

# Note we have to put a suffix in as we join won't allow columns with the same title

print("We use the pd.DataFrame.join(df1,df2, rsufifix='_new') command"
      "that is going to be added\n"
      "Also note that it will match the indices, hence the missing data in the 1 index row and 2 index row\n"
      "NOTE: The rsuffix part is because Join will not allow columns with the same title and suffix alters col title\n")
print(pd.DataFrame.join(df1, df2, rsuffix='_new'))

#input("\n\nPress Enter to Continue...")

print("The last technique to combine DFs is the MERGE command which performs a database-style join operation by "
      "columns or indices\nIt has a significant number of options from on, how, left_on, right_on to sort\n")

print("\n-----Keeping in mind the Ven Diagram idea here are the different joining methods-----\n")
print("INNER JOIN --> Only retains information where both DFs indices and columns match\n")
print(df1.merge(df2, on='B', how='inner'))
print("FULL JOIN  --> Retains all information filling holes with np.nan values (also known as OUTER)\n")
print(df1.merge(df2, on='B', how='outer'))
print("LEFT JOIN  --> Retains all 'left' DF information AND information where both DFs indices and columns match\n")
print(df1.merge(df2, on='B', how='left'))
print("RIGHT JOIN --> Retains all 'right' DF information AND information where both DFs indices and columns match")
print(df1.merge(df2, on='B', how='right'))

#input("\n\nPress Enter to Continue...")

print("\nWe can also merge the different dataframes based on their own column values specified by "
      "left_on= and right_on=\n"
      "i.e. df1.merge(df2,left_on='A', right_on='C', how='inner')\n"
      "NOTE: Our inner merges will rarely give us anything with an inner merge due to the random number selection\n"
      "NOTE: If we had more matching data we would retain more of the data\n")
print(df1.merge(df2, left_on='A', right_on='C', how='inner'))

#input("\n\nPress Enter to Continue...")

print("\nLets create a new DF that we can use to explore SORTING within a DF\n")
print("----New DataFrame df using random numbers----\n")

d = np.floor(np.random.rand(24).reshape(12, 2)*5)
df = pd.DataFrame(d, columns=['a', 'b'])
print(df)

#input("\n\nPress Enter to Continue...")

print("\nNow lets sort the DF based on the column 'a' (like you would in Excel)\n")
print("df.sort_values(by=['a'])")
print(df.sort_values(by=['a']))

#input("\n\nPress Enter to Continue...")

print("Note that we can pass multiple arguments into the by variable so if there are duplicates it will move to the "
      "next sorting parameter\nAdditionally you can specifiy ascending or descending for the order\n")
print("df.sort_values(by=['a', 'b'], ascending=[1, 0]) NOTE: the ascending binary works for the respective "
      "sorting value\n")
print(df.sort_values(by=['a', 'b'], ascending=[1, 0]))
print("\nNotice how the a values take priority and are ascending while the b values are secondary and descend")

#input("\n\nPress Enter to Continue...")

print("\nWe will now take a look at aggregating and grouping data\n"
      "Grouping         --> Useful for comparing subsets, deducing differences in subgroups, and general subsetting\n"
      "Aggregating      --> Useful in combining similar data, summarizing data, "
      "and/or reducing query time via data simplification\n")

#input("\nPress Enter to Continue...")

print("\nWe will import a csv containing some car data called mtcars.csv and display the head below to help"
      "with understanding of grouping and aggregating\n")

car_file_name = '../input/mtcars.csv'
car_df = pd.read_csv(car_file_name, usecols=['model', 'mpg', 'cyl', 'hp'])
print(car_df.head(10))

#input("\n\nPress Enter to Continue...")

print("\nNow we will group all cars according to the number of cylinders they possess\n"
      "The command car_df.groupby('cyl') will accomplish this\n"
      "NOTE: GroupBy objects can be difficult to work with, use them to extract data such"
      "as mean, std, min, max, and use the describe() command to access this info")

print(car_df.groupby(['cyl']).describe())

#input("\n\nPress Enter to Continue...")

print("\nNow we look at data pivoting using the df.pivot command\n"
      "The index parameter lets you pick a column to use as the pivot table's index\n"
      "The columns parameter lets you pick a column to draw new columns values from\n"
      "And the values parameter lets you pick the column containing the values you care about\n"
      "i.e. car_df.pivot(index='cyl', columns='model', values='hp')")

print(car_df.pivot(index='cyl', columns='model', values='hp'))

#input("\n\nPress Enter to Continue...")

print("\nFinally we will cover some basic statistical commands\n"
      "df.describe() --> Yields mean, min, max, std, and other useful information\n"
      "df(or s1).pct_chage() --> Yields the percent change from one row to the next\n"
      "df(or s1).cov() --> Compute pairwise covariance of columns, excluding NA/null values\n"
      "df(or s1).corr() --> Compute pairwise correlation of columns, excluding NA/null values\n")

print("The following four outputs are those described above in respective order...\n\n")
#input("Press Enter to See the 4 Outputs Using The TSLA DF...\n\n")
tsla = tsla.fillna(method='ffill')
print("\ntsla.describe()")
print(tsla.describe())
print("\ntsla.pct_change()")
print(tsla.pct_change())
print("\ntsla.cov()")
print(tsla.cov())
print("\ntsla.corr()")
print(tsla.corr())

