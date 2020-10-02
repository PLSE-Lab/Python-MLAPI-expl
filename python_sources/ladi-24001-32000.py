# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from skimage import io
from skimage.transform import resize
import tarfile

images_metadata = pd.read_csv('/kaggle/input/ladiannotation/ladi_images_metadata.csv')
aggregated_responses = pd.read_csv('/kaggle/input/ladiannotation/ladi_aggregated_responses.tsv', sep='\t')

c = images_metadata.loc[images_metadata['url'].isin(aggregated_responses['img_url'].unique())]

def size(x, y, new):
    if x > y:
        dv = x / new
        y = y // dv
        return (new, int(y))
    else:
        dv = y / new
        x = x // dv
        return (int(x), new)

_path = './'
try:
    os.remove(_path + 'images4.tar.gz')
except:
    print('No file')
          
tar = tarfile.open("images4.tar.gz", "w:gz")
p = len(c)
i = 1
batch = 1
filelist = []

for index,  row in c[['uuid', 'url']].iterrows():
    if i>24000:
        _str = str(i) + '/' + str(p) + ': '
        print(_str + 'Downloading...')
        os.system('wget -c -O ' + _path + row['uuid'] + '_1.jpg ' + row['url'])
        #print(_str + 'Downloaded!')
        filelist.append(row['uuid'])

        if i%25 == 0:
            print(str(batch) + ': Resizing...')

            for _uuid in filelist:
                filename = os.path.join(_path, _uuid + '_1.jpg')
                filename2 = os.path.join(_path, _uuid + '.jpg')            
                pic = io.imread(filename)            
                #pic_resized = resize(pic, size(pic.shape[0], pic.shape[1], 1024), preserve_range=True, anti_aliasing=False)
                pic_resized = cv2.resize(pic, size(pic.shape[1],pic.shape[0], 1024))
                io.imsave(filename2, pic_resized)            
                os.remove(filename)


            print('Add to archive...')        
            for _uuid in filelist:
                filename2 = os.path.join(_path, _uuid + '.jpg')
                tar.add(filename2)
                os.remove(filename2)

            filelist = []
            batch += 1
            
    if i==32000:
        break
    i += 1

tar.close()