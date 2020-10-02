import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
import csv
import re
import xlrd
import zipfile

filepath = "CityofLA/"

def create_db():
    with open('jobs_db.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        row_list = ["job name", "class code", "open date", "salary low 1", "salary high 1", "salary low 2", "salary high 2", "salary low 3", "salary high 3", "salary low 4", "salary high 4", "salary low 5", "salary high 5", "salary low 6", "salary high 6", "deadline", "requirements", "duties"]
        writer.writerow(row_list)

        for filename in os.listdir(pd.read_csv(filepath+"Job Bulletins/")):
            row_list.clear()

            file = open(pd.read_csv(filepath+"Job Bulletins/"+filename, 'r', errors="replace"))   
            text = file.read().strip()

            lines = text.splitlines()

            for l in lines:
                l = l.strip()

            if lines[0] == "CAMPUS INTERVIEWS ONLY":
                job_title = lines[1]
            else:
                job_title = lines[0]

            # initialize other vars
            class_code = ""
            open_date = ""
            salary_low1 = ""
            salary_high1 = ""
            salary_low2 = ""
            salary_high2 = ""
            salary_low3 = ""
            salary_high3 = ""
            salary_low4 = ""
            salary_high4 = ""
            salary_low5 = ""
            salary_high5 = ""
            salary_low6 = ""
            salary_high6 = ""
            deadline = ""
            requirements = ""
            duties = ""

            # initialize boolean flags
            code_found = False
            date_found = False
            salary_found = False
            deadline_found = False
            reqs_found = False
            duties_found = False

            for line in range(0, len(lines)):
                if(lines[line][0:11] == "Class Code:" and line < 15 and code_found == False):
                    class_code = lines[line][11:].strip()
                    code_found = True

                if(lines[line][0:10] == "Open Date:" and date_found == False):
                    open_date = lines[line][10:].strip()
                    date_found = True

                if(len(lines[line]) > 0 and lines[line][0] == '$' and salary_found == False):
                    salary_line = lines[line].strip()
                    salary_list = re.findall(r"[0-9]+,[0-9]+", salary_line)

                    if(len(salary_list) >= 1):
                        salary_low1 = salary_list[0]
                    if(len(salary_list) >= 2):
                        salary_high1 = salary_list[1]
                    if(len(salary_list) >= 3):
                        salary_low2 = salary_list[2]
                    if(len(salary_list) >= 4):
                        salary_high2 = salary_list[3]
                    if(len(salary_list) >= 5):
                        salary_low3 = salary_list[4]
                    if(len(salary_list) >= 6):
                        salary_high3 = salary_list[5]
                    if(len(salary_list) >= 7):
                        salary_low4 = salary_list[6]
                    if(len(salary_list) >= 8):
                        salary_high4 = salary_list[7]
                    if(len(salary_list) >= 9):
                        salary_low5 = salary_list[8]
                    if(len(salary_list) >= 10):
                        salary_high5 = salary_list[9]
                    if(len(salary_list) >= 11):
                        salary_low6 = salary_list[10]
                    if(len(salary_list) >= 12):
                        salary_high6 = salary_list[11]

                    salary_found = True

                if(lines[line][0:32] == "Applications must be received by" and deadline_found == False):
                    deadline = lines[line][32:].strip()
                    deadline_found = True

                if(lines[line][0:11] == "REQUIREMENT"):
                    l = line + 1
                    req_list = list()

                    if(lines[l] == ""):
                        l = l + 1

                    while(lines[l] != "" and lines[l][0:7].lstrip() != "PROCESS" and lines[l][0:4].lstrip() != "NOTE"):
                        req_list.append(lines[l])
                        l = l + 1

                    requirements = '\n'.join(req_list)
                    reqs_found = True  

                if(lines[line][0:6] == "DUTIES"):
                    if(lines[line].strip() == "DUTIES AND RESPONSIBILITIES"):
                        l = line + 1
                    else:
                        l = line + 2

                    duties = lines[l].strip()
                    duties_found = True

            row_list.append(job_title)
            row_list.append(class_code)
            row_list.append(open_date)
            row_list.append(salary_low1)
            row_list.append(salary_high1)
            row_list.append(salary_low2)
            row_list.append(salary_high2)
            row_list.append(salary_low3)
            row_list.append(salary_high3)
            row_list.append(salary_low4)
            row_list.append(salary_high4)
            row_list.append(salary_low5)
            row_list.append(salary_high5)
            row_list.append(salary_low6)
            row_list.append(salary_high6)
            row_list.append(deadline)
            row_list.append(requirements)
            row_list.append(duties)

            writer.writerow(row_list)


def get_headings(bulletin):      
    for filename in os.listdir(filepath+"Job Bulletins/"):
        f = open(filepath+"Job Bulletins/"+filename)
        data=f.read().replace('\t','').split('\n')
        data=[head for head in data if head.isupper()]
        f.close()
        return data

    # with open(filepath+"Job Bulletins/"+bulletins[bulletin]) as f:    ##reading text files 
    #     data=f.read().replace('\t','').split('\n')
    #     data=[head for head in data if head.isupper()]
    #     return data
        
def clean_text(bulletin):                                               
    with open(filepath+"Job Bulletins/"+bulletins[bulletin]) as f:
        data=f.read().replace('\t','').replace('\n','')
        return data

def main():
    from subprocess import check_output
    print(check_output(["ls", "../input/Data Science for Good: City of Los Angeles/data"]).decode("utf8"))
    
    with zipfile.ZipFile("../input/data/CityofLA.zip","r") as z:
        z.extractall(".")
    
    create_db()
    
    # files=[dir for dir in os.walk(filepath)]

    # for file in files:
    #     print(os.listdir(file[0]))
    #     print("\n")
    #     bulletins=os.listdir(filepath+'Additional data/job bulletins with annotations')
    #     additional=os.listdir(filepath+'Additional data/City Job Paths')

    # job_title=pd.read_csv(filepath+'Additional data/job_titles.csv')
    # sample_job=pd.read_csv(filepath+'Additional data/sample job class export template.csv')
    # kaggle_data=pd.read_csv(filepath+'Additional data/kaggle_data_dictionary.csv')

    # headings = get_headings(bulletins)
    # print(headings[0:5])
    
main()

