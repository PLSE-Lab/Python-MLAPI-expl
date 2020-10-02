# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:29:31 2020

@author: YannCLAUDEL

The function ipynbtopy(input_dir) convert all files *.ipynb to *.py 
The targets files *.py are created in a directory pysrc
"""

import json
import os

def ipynbtopyParse(fileIn,fileOut):

    with open(fileIn) as json_file:
        data = json.load(json_file)
    
    out = open(fileOut,"w")
    
    for cell in data['cells']:    
        if (cell.get('cell_type')!=None and cell.get('cell_type')=='markdown'):
            lines = cell.get('source')
            if len(lines)>0:
                out.write("# "+str(lines[0]).replace("#", ''))
                out.write("\n")
        if (cell.get('cell_type')!=None and cell.get('cell_type')=='code'):
            for line in cell.get('source'):
                out.write(line)
            out.write("\n\n")
        # out.write("\n")
            
    out.close()


def ipynbtopy(input_dir):
    for dirname, dirnames, filenames in os.walk(input_dir):
        # print path to all subdirectories first.
        # for subdirname in dirnames:
        #     print(os.path.join(dirname, subdirname))
    
        # print path to all filenames.
        for filename in filenames:
            if filename.endswith(".ipynb"):
                newfilename = filename[:-6]+".py"
                newdirname = dirname+r"\pysrc"
                if not os.path.exists(newdirname):
                    os.mkdir(newdirname)
                ipynbtopyParse(os.path.join(dirname, filename),os.path.join(newdirname, newfilename))
                
        # Advanced usage:
        # editing the 'dirnames' list will stop os.walk() from recursing into there.
        if '.git' in dirnames:
            # don't go into any .git directories.
            dirnames.remove('.git')
        if '.ipynb_checkpoints' in dirnames:
            # don't go into any .git directories.
            dirnames.remove('.ipynb_checkpoints')

input_dir="."
ipynbtopy(input_dir)