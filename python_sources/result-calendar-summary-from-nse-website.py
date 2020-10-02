'''
Find out dates of forthcoming corporate events for a list of scripts. 
Does not check for internet connection, overwrites CSV file and writes a
summary to text file named boardMeetDates.txt.

Code uses Selenium, Pandas and hence a browser window will appear in the
backgroud. This can be circumvented using QtPy where a browser window is
created within Python and does not rely on web-drivers like Mozilla.
'''
import requests

# Download the CSV file and save it in same folder as this Python code
url = 'https://www.nseindia.com/corporates/datafiles/BM_All_Forthcoming.csv'
bmDates = requests.get(url, allow_redirects = True)
open('BM_All_Forthcoming.csv', 'wb').write(bmDates.content)

# Import the CSV file and seach for BM dates for given list of scrips
# Reading CSV file in Pandas
import pandas as pnd
bmDate_CSV = pnd.read_csv(BM_All_Forthcoming.csv, index_col=None, encoding='utf-8')
# r is special character for carriage return - prefix for string literals

#Delete duplicate row entries for same scrips
dFA = pnd.DataFrame(bmDate_CSV)
dFB = dFA.drop_duplicates(subset=None, keep='first', inplace=False)

#Assign a column with unique values as the index (X-Axis) of the dataframe 
#dFC = dFB.set_index("Symbol", drop = True)

dFB = pnd.DataFrame(dFB.sort_values(by = ['Symbol']),
                    columns = ['Symbol', 'BoardMeetingDate'])

#print(bmDate_CSV.head())  # Print top 5 rows of the data - for debugging only
#print(bmDate_CSV)         # Print entire content of the CSV file

#print(dFA[dFA.Symbol == "HDFCBANK"])  #Print row containing 'HDFCBANK'

# Input data files on Kaggle are available in the "../input/" directory.
scripFile = open("../Input/stocksList.txt") 
scripArray = scripFile.read()
scripList = scripArray.split("\n")

#Print complete row from dataFrame created from raw CSV file
#print(dFA.loc[dFA['Symbol'].isin(scripList)])

#Print only scrip code and the board meeting date: dFB.Symbol = dFB['Symbol']
#print(dFB.loc[dFB['Symbol'].isin(scripList)])  #Prints with index number

#Print list without index number
dFF = dFB.loc[dFB['Symbol'].isin(scripList)]
print(dFF.to_string(index = False))

scripFile.close

# Write the scrip list and BM dates in a text file
fileName = 'boardMeetDates.txt'
f = open(fileName, "w")
f.write(dFF.to_string(index = False))
f.close()

'''  Sample input: stocksList.txt
AMARAJABAT
BATAINDIA
BEML
BHARTIARTL
CANFINHOME
FEDERALBNK
HDFC
HDFCBANK
INDUSINDBK
INFY
JSWENERGY
L&TFH
LICHSGFIN
LUPIN
M&MFIN
MARUTI
PFC
RELIANCE
SUNPHARMA
SUZLON
TCS
VGUARD
YESBANK
'''
''' Output from this program
Symbol BoardMeetingDate
AMARAJABAT      09-Nov-2018
 BATAINDIA      02-Nov-2018
BHARTIARTL      25-Oct-2018
CANFINHOME      22-Oct-2018
      HDFC      01-Nov-2018
  HDFCBANK      20-Oct-2018
 JSWENERGY      02-Nov-2018
     L&TFH      24-Oct-2018
 LICHSGFIN      29-Oct-2018
     LUPIN      31-Oct-2018
    M&MFIN      24-Oct-2018
    MARUTI      25-Oct-2018
       PFC      02-Nov-2018
    VGUARD      25-Oct-2018
   YESBANK      25-Oct-2018
'''