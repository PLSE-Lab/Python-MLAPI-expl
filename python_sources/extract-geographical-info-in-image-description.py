#!/usr/bin/env python
# coding: utf-8

# # Missing Exif Data
# The competition organizers [gave us permission][org] to use image `Exif` data and other metadata to identify landmarks.
# 
# If you used one of the [published image downloader kernels][imgk] to download your images, then you are probably missing out on this data.  The potentially useful meta data is dropped in the image download and conversion process.  You may want to redownload the images (~ 460GB) with the script shown at the end of this kernel if you want to use these nuggets.
# 
# [org]: https://www.kaggle.com/c/landmark-recognition-challenge/data
# [imgk]: https://www.kaggle.com/c/landmark-recognition-challenge/kernels

# # Details and Examples
# [`Exif` data][exif_tags] contains several image metadata fields, known as tags.  They range from GPS coordinates, to image description and type of camera used.  I couldn't find any GPS data in the test data set, but was able to find a lot of other useful metadata.
# 
# Here will explore the `ImageDescription` tag info that we can get from some images after downloading them with the script at the end of this kernel.
# 
# This is photo *68f7904482b7ef4c*  from the [`test.csv` data set][test].  It's a picture of [Bird's Nest stadium][bn] in Beijing, China.
# 
# ![](https://lh3.googleusercontent.com/--oQKH9vZtM0/WOuQnwwjaGI/AAAAAAAAAG0/QQ0IaOM2aIANLUuxizJ-nYkb9Iwgyk8YwCOcB/s1600/)
# 
# Let's read it's `Exif` `ImageDescription` tag we preserved.
# 
# [bn]: https://en.wikipedia.org/wiki/Beijing_National_Stadium
# [exif_tags]: https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
# [test]: https://www.kaggle.com/c/landmark-recognition-challenge/data

# In[ ]:


from PIL import Image
from io import BytesIO
import piexif
import warnings
warnings.filterwarnings('ignore')

exif_dict = piexif.load('../input/exif-data/68f7904482b7ef4c.jpg')
image_desc = exif_dict['0th'][piexif.ImageIFD.ImageDescription]
print(image_desc)


# Here we have the landmark name in the image description:  **Bird's Nest**

# This test set image *68f7904482b7ef4c* is of [Nero's Golden Palace][nero] in Rome, Italy, also known as *The Domus Aurea*.
# 
# ![](https://lh3.googleusercontent.com/-JnHJsWUy86U/WOFtIzUAgDI/AAAAAAAAFFM/Ti_HGa5EBX0B7Y_Y9vzOHyVxJiP11bPUgCOcB/s1600/)
# 
# And we have the landmark name listed in its `ImageDescription`.
# 
# [nero]: https://www.tickitaly.com/galleries/domus-aurea-rome.php

# In[ ]:


exif_dict = piexif.load('../input/exif-data/81cb67de72768b6f.jpg')
image_desc = exif_dict['0th'][piexif.ImageIFD.ImageDescription]
print(image_desc)


# And this one (test set: *6072c8a91b1a6f54*) is for [Abraj Al Bait Towers][abraj] in Mecca, Saudi Arabia.
# 
# ![](https://lh3.googleusercontent.com/-07QTjojZPhs/VrrbKXs4a3I/AAAAAAABugQ/mf54W9Ua2Y4/s1600/)
# 
# It's `ImageDescription` tag info has the landmark name.
# 
# [abraj]: https://en.wikipedia.org/wiki/Abraj_Al_Bait

# In[ ]:


exif_dict = piexif.load('../input/exif-data/6072c8a91b1a6f54.jpg')
image_desc = exif_dict['0th'][piexif.ImageIFD.ImageDescription]
print(image_desc)


# # Conclusion
# Not all images have `Exif` data, and not all those that have it will have a meaningful `ImageDescription` tag, or even have one.  However, I found out that a lot of the landmark images we have in this competition were taken by people who were willing to tag their images with descriptions to show them off.  I will use all the info we have to improve my recognition accuracy, and so should you.

# # Image Downloader with Exif Data
# 
# This script will not run as a kernel here.  You need to run it locally on your own machine.  It's a modification of [the image downloader kernel][down] by [anokas].
# 
# You need to install [`piexif` library][piexif] for this to work.
# 
# [piexif]: http://piexif.readthedocs.io/en/latest/
# [down]: https://www.kaggle.com/anokas/python3-dataset-downloader-with-progress-bar
# [anokas]: https://www.kaggle.com/anokas

# In[ ]:


# -*- coding: utf-8 -*-

# !/usr/bin/python

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import tqdm
import piexif


def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def download_image(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    filename = os.path.join(out_dir, '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
        # Read Exif data if the image has it.
        # Otherwise, create empty exif_bytes
        if pil_image.info.get('exif', None):
            exif_dict = piexif.load(pil_image.info['exif'])
            exif_bytes = piexif.dump(exif_dict)
        else:
            exif_bytes = piexif.dump({})
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        # Include exif data with saved file.
        pil_image_rgb.save(filename, format='JPEG', quality=90, exif=exif_bytes)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1
    
    return 0


def loader():
    if len(sys.argv) != 3:
        print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=20)  # Num of CPUs
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list), total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
    if os.environ['PYTHONPATH'] == '/kaggle/lib/kagglegym':
        print('This script does not run as intented on Kaggle.  Downlaod and run locally.')
    else:
        loader()

