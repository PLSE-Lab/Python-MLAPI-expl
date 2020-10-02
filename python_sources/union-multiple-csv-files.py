# -*- coding: utf-8 -*-
"""
@author: Smit Mehta
task:    Combine multiple CSVs into one (Union)
"""

#importing libraries
import glob
import os

def get_dir():
    dir = input('Enter path: ')
    try:
        os.chdir(dir)
    except:
        print('Path Name was incorrect')
        get_dir()


def union_csvs(output, csvlist):
    with open(output, 'w') as output_file:
        for each_csv_file in csvlist:
            with open(each_csv_file) as input_file:
                for line in input_file:
                    output_file.write(line)

        
        
        
get_dir()
output_filename = input('Enter a name for the output file (with extension): ')

all_csv_files = [f for f in glob.glob('*.csv')]
union_csvs(output_filename, all_csv_files)
