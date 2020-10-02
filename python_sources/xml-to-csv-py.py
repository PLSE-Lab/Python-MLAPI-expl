import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[5][0].text),
                     int(member[5][1].text),
                     int(member[5][2].text),
                     int(member[5][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def xml_to_csv_utility():
    os.chdir("/kaggle/input/face-data/homo_encrypt/image_data")
    for image_set in ['train_data','test_data']:
        image_path = os.path.join(os.getcwd(), image_set)
        print(image_path)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('{}.csv'.format(image_set.split("_")[0]), index=None)
        print('Successfully converted xml to csv.')

