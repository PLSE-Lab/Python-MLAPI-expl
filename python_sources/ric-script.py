# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import xml.etree.ElementTree as ET
import argparse
import os

# write your class names in your xml files
#classesname
classes = ["pet", "hdpf", "pvc", "peld", "pp", "ps", "other", "no_p"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)



def main(args):

	dataset_path = args.dataset_path

	annot_path = os.path.join(dataset_path, "annots")

	filenames = os.listdir(annot_path)


	if not os.path.exists(os.path.join(dataset_path, "labels")):
		os.makedirs(os.path.join(dataset_path, "labels"))
	for filename in filenames:
		img_id = filename[:-4]
		file_path = os.path.join(annot_path, filename)
		in_file = open(file_path)
		out_file = open(os.path.join(dataset_path, "labels", img_id + ".txt"), "w")
		tree=ET.parse(in_file)
		root = tree.getroot()
		size = root.find('size')
		w = int(size.find('width').text)
		h = int(size.find('height').text)

		for obj in root.iter('object'):
			difficult = obj.find('difficult').text
			cls = obj.find('name').text

			if cls not in classes or int(difficult) == 1:
				continue
			cls_id = classes.index(cls)
			xmlbox = obj.find('bndbox')
			b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
			bb = convert((w,h), b)
			out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
	

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument(
		'-d', '--dataset_path',
		help = 'The path of dataset',
		required = True)


    args = argparser.parse_args()
    
    main(args)