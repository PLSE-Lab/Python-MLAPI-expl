import csv as csv
from tabulate import tabulate
from operator import itemgetter

country = 'Switzerland' # Enter your desired country
year = '2015' # Enter the year. It can be 2014 or 2015 
max_rank = 100 # Enter the maximum ranking

readdata = csv.reader(open("../input/cwurData.csv")) # Opens the CSV file

alldata = [] # First we read and append the CSV file to an empty list
for data in readdata:
    alldata.append(data)

countrydata = alldata[1:] # The first row in our dataset is the table header, so we ignore it first

# Itemgetter helps to omit 'country'(2) for our selected country. So the selection will be 'world ranking' (0), 'institution'(1) and 'national ranking' (3) 
record = [(itemgetter(0,1,3)(record)) for record in countrydata if record[2] == country and record[-1] == year and int(record[0]) <= max_rank]

# Lets print a title for our table
print('\n\n', country, 'TOP UNIVERSITIES (<',max_rank,') (',year,')\n')

# Tabulate helps to create a table from our selected data
print(tabulate(record,headers=(itemgetter(0,1,3)(alldata[0]))))

#Lets print a title for our next table
print('\n\nWORLD TOP', max_rank, 'UNIVERSITIES (',year,')\n')

# We repeat the same code to get world top 100 universities, this time without country record 
data = [record[:3] for record in countrydata if int(record[0]) <= max_rank and record[-1] == year]
print(tabulate(data,headers=alldata[0]))