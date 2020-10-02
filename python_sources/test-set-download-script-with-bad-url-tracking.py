# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from urllib import request, error
import urllib
import pandas as pd
import csv

################
# Change here
output_dir = '//d02/data/google_recog2019/test'
test_file = './data/test.csv'
bad_ids_file = './data/bad_ids.csv'
################

test_df = pd.read_csv(test_file, header=0, dtype=str)

from PIL import Image
from io import BytesIO


def download(output_dir, url, filename):
    """Download to a directory structure similar to the train set.
    This permits you to rerun and not download already existing tar files from previous attempts."""
    d1 = filename[0]
    d2 = filename[1]
    basepath=os.path.join(output_dir,d1,d2)
    os.makedirs(basepath, exist_ok=True)
    filepath = os.path.join(basepath, filename+'.jpg')
    tmp_filepath = os.path.join(basepath, 'tmp_'+filename+'.jpg')

    # if the file is already present we don't want to do anything but ack its presence
    if os.path.isfile(filepath):
        return True, 'Already downloaded!'

    print('Downloading %s to %s' % (url, tmp_filepath))

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        return False, 'Warning: Could not download image {} from {}'.format(filename, url)

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        return False, 'Warning: Failed to parse image {}'.format(filename)

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        return False, 'Warning: Failed to convert image {} to RGB'.format(filename)

    try:
        pil_image_rgb.save(filepath, format='JPEG', quality=90)
    except:
        return False, 'Warning: Failed to save image {}'.format(filepath)

    return True, 'OK'

bad_ids = []
msgs = []

for row in range(len(test_df)):
        url = test_df.url[row]
        file_name = test_df.id[row]
        if url != 'None':
            status, msg =  download(output_dir, url, file_name)
            if not status:
                print(msg)
                bad_ids.append(file_name)
                msgs.append(msg)


print('Downloading completed  ...')

pd.DataFrame({'id':bad_ids, 'msg':msgs}).to_csv(bad_ids_file, header=True, index=False, quoting=csv.QUOTE_ALL)
