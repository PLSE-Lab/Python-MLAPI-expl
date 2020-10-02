import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import json


class FoodDataset(Dataset):

    def __init__(self, data_folder, transform, task, json_path, train=False):
        """
        :param data_folder:
        :param train: load train input if train is true else load test input
        :param transform:
        """
        self.samples, self.label_names = self.load_data(data_folder, task, json_path, train)
        self.transform = transform
        print(self.label_names)

    def __getitem__(self, item):
        filename, label = self.samples[item]
        img = Image.open(filename)
        if img.getbands()[0] == 'L':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return self.label_names

    def load_data(self, data_folder, task, json_path, train):
        """
        load training input or testing input
        :param task_index:
        :param class_num:
        :param data_folder:
        :param train:
        :return:
        """
        image_folder = os.path.join(data_folder, 'images')
        meta_folder = os.path.join(data_folder, 'meta/meta')
        task_arr = json.load(open(json_path))
        if train:
            filename_dict = json.load(open(os.path.join(meta_folder, 'train.json')))
        else:
            filename_dict = json.load(open(os.path.join(meta_folder, 'test.json')))

        dict = task_arr[task]
        class_num = dict['class_num']
        label_names = dict['labels']

        samples = []

        for i in range(class_num):
            label_name = label_names[i]
            filenames = filename_dict[label_name]
            for filename in filenames:
                sample = (os.path.join(image_folder, filename + ".jpg"), i)
                samples.append(sample)

        return samples, label_names

