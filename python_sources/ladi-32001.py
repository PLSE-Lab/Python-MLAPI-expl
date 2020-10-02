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
    os.remove(_path + 'images5.tar.gz')
except:
    print('No file')
          
tar = tarfile.open("images5.tar.gz", "w:gz")
p = len(c)
i = 1
resize_errors = 0
errors = []
arch_errors = 0

for index,  row in c[['uuid', 'url']].iterrows():
    _uuid = row['uuid']
    
    if i>32000:
        _str = str(i) + '/' + str(p) + ': '
        print(_str + 'Downloading...')
        os.system('wget -c -q -nv -O ' + _path + _uuid + '_1.jpg ' + row['url'].replace('(', '\('))

        print(_str + 'Resizing...')
        filename = os.path.join(_path, _uuid + '_1.jpg')
        filename2 = os.path.join(_path, _uuid + '.jpg') 

        try:         
            pic = io.imread(filename)            
            pic_resized = cv2.resize(pic, size(pic.shape[1],pic.shape[0], 1024))
            io.imsave(filename2, pic_resized)            
            os.remove(filename)
        except:
            print('Resizing error')
            resize_errors += 1
            errors.append(_uuid)


        print(_str + 'Add to archive...')        
        try:
            filename2 = os.path.join(_path, _uuid + '.jpg')
            tar.add(filename2)
            os.remove(filename2)
        except:
            print('Arch error')
            arch_errors += 1

    i += 1

tar.close()
print('resize errors: ' + str(resize_errors))
print('arch_errors: ' + str(arch_errors))
print(errors)