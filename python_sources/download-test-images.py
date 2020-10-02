# Note: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset.
#url that are forbidden are skipped.
# All downloaded images will be saved in the JPG format

import csv
import urllib

data_file = 'F:/.../test.csv'
csvfile = open(data_file, 'r')
csvreader = csv.reader(csvfile)
key_url_list = [line[:2] for line in csvreader]
key_url_list = key_url_list[1:]

url_data = []
for i in range(117703):
    url_data.append(key_url_list[i][1])

def download(url, directory):
    print('Downloading %s to %s' % (url, directory))
    try:
        urllib.request.urlretrieve(url, directory)
    except:
        print("Forbidden url")
  
n = 1
for row in url_data:
    filepath = "F:/.../test"
    filepath = filepath + '/' + str(n) + '.jpg'
    n = n + 1
    if(row==None or row=='None'):
        continue
    else:
        url = row
    download(url, filepath)