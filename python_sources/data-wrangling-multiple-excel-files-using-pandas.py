import pandas as pd
import xlrd
import numpy as np
import sys
import csv
import openpyxl
import os
reload(sys)

sys.setdefaultencoding('UTF8')
#Global variables-------------------------------------------------
data=[]
#rootdir="directory where your files are located"#
#outfile="name of your output file"#
idx=0
sheet_name="sheet name"
#read source files from directory and store in list-------------------------------------------------
for subdir, dirs, files in os.walk(rootdir):
	for file in files:
		excel_files=[]
		excel_files.append(os.path.join(subdir, file))
		
#iterate through list and read source data into dataframe
		for ind_file in excel_files:
			worksheet=pd.read_excel(io=ind_file,worksheet=sheet_name)
#create second dataframe and insert filename
			filename=pd.Series(file,excel_files)
			worksheet.insert(0,"Filename",file)
			
#insert content from spreadsheets into list object
			data.insert(idx,worksheet)					
##concatenate the content in the list into one dataframe	
results=pd.concat(data)
#write data to csv file
results.to_csv(outfile,encoding='utf-8')