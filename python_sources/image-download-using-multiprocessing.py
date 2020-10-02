#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import multiprocessing
from multiprocessing import Pool
import requests
import json, requests, shutil
from pandas.io.json import json_normalize
import os
import urllib3

"""
How to use this code
File structure
---------------
/data
/input
data.py



File description
-----------------
/data is originally empty. It will stored the data in test, train and validation folders.
/input contains all the required data input which include test.json, train.json, and validation.json

"""



def download(data_type, data):

    #create a directory if it is absent
    directory = 'data/'+data_type+'_images/'
    if not os.path.exists(directory):
        os.makedirs(directory)



    for i in range(len( data)):
        #response = requests.get(data.loc[i]['url'], timeout = 5, stream = True)


        try:
            response = requests.get(data.loc[i]['url'], timeout = 50000, stream = True)

            with open( directory+ str(data.loc[i]['imageId'])+ '.jpeg', 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
        except ( requests.exceptions.ConnectionError, urllib3.exceptions.ProtocolError ):
            print "Connection refused"

    return 1


def convertToParallelSolution (ind, data_type, data, numOfProcesses):
    totalnumberOfSamples = len( data )

    numOfBins = round ( totalnumberOfSamples / numOfProcesses ) + 1
    start =  int (ind * numOfBins )
    end = int (start + numOfBins )

    result = 0
    
    if end >= totalnumberOfSamples:
        end = totalnumberOfSamples

    if end <= start:
        return result 

    if end > start:
        result = download (data_type, data[start : end].reset_index( ) )

    print "Batch {} of {} is done so far!!!!! {}.json (in progress)".format (ind, numOfProcesses, data_type)
    #return result
    return 1


def parallel_solution (data_type, data, numOfProcesses=20 ):

    pool = Pool(processes=numOfProcesses)              # start 20 worker processes

    # launching multiple evaluations asynchronously *may* use more processes
    multiple_results = [pool.apply_async(convertToParallelSolution, args=(ind, data_type, data, numOfProcesses, )) for ind in range(numOfProcesses)]

    resultContainer =  [res.get() for res in multiple_results]

if __name__ == "__main__":

    dataTypeLst = ["test", "train", "validation" ]
    for data_type in dataTypeLst:
        data = json.load(open('input/' + data_type + '.json'))
        data = json_normalize(data["images"])
        print "{}.json is fully loaded".format (data_type)
        parallel_solution  (data_type, data )


# In[ ]:




