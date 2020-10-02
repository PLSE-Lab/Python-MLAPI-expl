import os
import pandas as pd
from zipfile import ZipFile

#Unzipping the file
# zip_file_directory = input('Enter the path to the zip file')
# zip_file_name = zip_file_directory[zip_file_directory.find()]
zip_location = os.getcwd()
zip_f = input("Enter the name of the file  ") #sas.zip
zip_file_name = zip_location + '\\' + zip_f
print(zip_file_name)

#taking only the filename from filename.zip
rootdir = zip_file_name[:zip_file_name.find(".")]
print(rootdir)

with ZipFile(zip_file_name, 'r') as zipObj:
    zipObj.extractall(rootdir)
print("successfully extracted")

#Searching through each folder and files inside the directory
dir_list = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        new_file = os.path.join(subdir,file)
        dir_list.append(new_file)
print(dir_list)

#Creating the csv foldername to store the csv file

path = rootdir+"_csv"
mode = 0o666
os.mkdir(path, mode) 
print(path)

#converting to csv with the save file name as the .csv file name
#C:\\Users\\sudip144856\\Desktop\\Python Practise\\sas\\airline.sas7bdat
csv_list = []
for ele in dir_list:
    df1 = pd.read_sas(ele)
    full_path_csv = path+"\\"+ele[ele.rfind("\\")+1:ele.find(".")]+".csv"
    csv_list.append(full_path_csv)
    df1.to_csv(full_path_csv,index = False) 
print(csv_list)
    #print(df1.head(5))

import csv , ast

for ele in csv_list:
    file = open(ele,'r')
    reader = csv.reader(file)

longest, headers, type_list = [], [], [] 
# longest - will give the longest varchar in the column 
# headers will gives the name of columns 
# type_list will give the possible type lists

import csv , ast



# longest - will give the longest varchar in the column 
# headers will gives the name of columns 
# type_list will give the possible type lists


#Searches for int , float : If the int is detected then check that the type is not varchar and float
#here type is divided into int, bigint and smallint
def dataType(val, current_type):
    try:
        # Evaluates numbers to an appropriate type, and strings an error
        t = ast.literal_eval(val) # if error c
    except ValueError:
        return 'varchar(max)' #varchar
    except SyntaxError:
        return 'varchar(max)'#varchar
    if type(t) in [int,  float]:
        if (type(t) in [int]) and current_type not in ['float', 'varchar']:
           # Use smallest possible int type
            if (-32768 < t < 32767) and current_type not in ['int', 'bigint']:
                return 'smallint'
            elif (-2147483648 < t < 2147483647) and current_type not in ['bigint']:
                return 'varchar(max)'
            else:
                return 'varchar(max)'
        if type(t) is float and current_type not in ['varchar']:
            return 'varchar(max)'
    else:
        return 'varchar(max)'
create_table_list = []
drop_table_list = []
    
for ele in csv_list:
    longest, headers, type_list = [], [], [] 
    file = open(ele,'r')
    reader = csv.reader(file)   
    for row in reader:
        if len(headers) == 0:
            headers = row
            for col in row:
                longest.append(100)
                type_list.append('')
        else:
            for i in range(len(row)):
                # NA is the csv null value
                if type_list[i] == 'varchar(max)' or row[i] == 'NA':
                    pass
                else:
                    var_type = dataType(row[i], type_list[i])
                    type_list[i] = var_type
            if len(row[i]) > longest[i]:
                longest[i] = len(row[i])
    file.close()

##WRITING CREATE TABLE SCRIPTS
    table_name =  ele[ele.rfind('\\')+1:ele.rfind('.')]
    print(headers)

    statement = 'create table ' + table_name + ' ('

    for i in range(len(headers)):
        if type_list[i] == 'varchar':
            statement = (statement + '[{}] varchar({}),').format(headers[i], str(longest[i])) #'\n[{}] varchar({}),'
        else:
            statement = (statement + '' + '[{}] {}' + ',').format(headers[i], type_list[i])

    statement = statement[:-1] + ');'
    create_table_list.append(statement)
   

    print(statement)
    
    ###INSERT
    #INSERT INTO employees
# (employee_id, last_name, first_name)
# VALUES
# (10, 'Anderson', 'Sarah'),
# (11, 'Johnson', 'Dale');

    
print("CREATION OF TABLE SUCCESSFUL _ NOW INSERTING")

# WRITING INSERT SCRIPT



    
#print(create_table_list)

print("Creation of table successfull")


import pyodbc
try:
    server = 'dbname'
    database = 'dbname' #test_sas_csv
    username = 'test'
    password = 'test123'
    conn = pyodbc.connect('DRIVER={SQL SERVER};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)
    cur = conn.cursor()
    
except Exception as e:
    print("The exception has occured" + str(e))


#Selecting the columns with the query
for ele in create_table_list:
    print(ele)
    cur.execute(ele)
    cur.commit()

print('success')
tracker_count = {}
try:
    for ele in csv_list:
        table_name =  ele[ele.rfind('\\')+1:ele.rfind('.')]
        insert_list = []
        df = pd.read_csv(ele, header = 0,delimiter = ',')
        insert_list = df.values.tolist()
        #print(insert_list)
        print(table_name)
        tracker_count[table_name] = len(insert_list) 
        print(len(insert_list))
        len_row = len(insert_list) 
        for e in range(len_row):
            list_data = insert_list[e]
            replace_square = str(list_data).replace('[','(').replace(']',')')
            insert_script = ("insert into "+table_name+ " values " + replace_square )
            cur.execute(insert_script)
            cur.commit()
except Exception as e:
    print("Exception occured while inserting to the db with the message ---> " + str(e))
cur.close()
print(tracker_count)
print(insert_script)