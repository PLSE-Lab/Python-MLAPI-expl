import os
import glob
import re
import csv
import numpy as np
import scipy.io.wavfile

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_folder = None
test_folder = None
train_folder = None
play_folder = None
submission_folder = None
to_float = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)


def set_data_path(option):
    global data_folder
    global test_folder
    global train_folder
    global play_folder
    global submission_folder
    if option == "local":
        data_folder = os.path.join(os.getcwd(), "Data")
        test_folder = os.path.join(data_folder, "Test")
        train_folder = os.path.join(data_folder, "Train")
        play_folder = os.path.join(data_folder, "Play")
        submission_folder = os.path.join(os.getcwd(), "Submission")
    elif option == "kaggle":
        data_folder = os.path.join("..", "input", "oeawai")
        test_folder = os.path.join(data_folder, "kaggle-test", "kaggle-test", "audio")
        train_folder = os.path.join(data_folder, "train", "kaggle-train", "audio")
        play_folder = os.path.join(data_folder, "train-small", "train-small", "audio")
        submission_folder = os.getcwd()
    else:
        raise NotImplementedError


class CustomDataSet(data.Dataset):

    def __init__(self, directory, transform=to_float):
        super().__init__()
        assert(isinstance(directory, str)), directory

        self.filenames = glob.glob(os.path.join(directory, "*.wav"))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        _, sample = scipy.io.wavfile.read(name)

        if self.transform is not None:
            sample = self.transform(sample)
        return torch.Tensor(sample)


class TestDataSet(CustomDataSet):

    def __init__(self, transform=to_float):
        super().__init__(test_folder, transform)
        # Needs to be sorted for test data, but don't do it elsewhere to save time
        self.filenames.sort(key=file_num)


def file_num(file_path):
    file_name = re.compile("\/").split(file_path)[-1]
    file_num = re.compile("\.").split(file_name)[0]
    return int(file_num)


class LabeledDataSet(CustomDataSet):

    def __init__(self, directory, transform=to_float, blacklist_patterns=None):
        if blacklist_patterns is None:
            blacklist_patterns = []
        super().__init__(directory, transform)

        assert(isinstance(blacklist_patterns, list))
        for pattern in blacklist_patterns:
            self.filenames = self.blacklist(self.filenames, pattern)

        self.build_label_encoder()

    def build_label_encoder(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(np.unique(self._instruments_family(self.filenames)))

    def _instruments_family(self, filenames):
        instruments = np.zeros(len(filenames), dtype=object)
        for i, file_name in enumerate(filenames):
            no_folders = re.compile("\/").split(file_name)[-1]
            instruments[i] = re.compile('_').split(no_folders)[0]
        return instruments

    def blacklist(self, filenames, pattern):
        return [filename for filename in filenames if pattern not in filename]

    def __getitem__(self, index):
        name = self.filenames[index]
        _, sample = scipy.io.wavfile.read(name)

        target = self._instruments_family([name])
        categorical_target = self.label_encoder.transform(target)[0]
        tensor_category = torch.from_numpy(np.array(categorical_target)).long()

        if self.transform is not None:
            sample = self.transform(sample)
        return [torch.Tensor(sample), tensor_category]

    def get_label(self, index):
        name = self.filenames[index]

        target = self._instruments_family([name])
        categorical_target = self.label_encoder.transform(target)[0]
        tensor_category = torch.from_numpy(np.array(categorical_target)).long()

        return tensor_category


class LabeledDataSetFromNames(LabeledDataSet):

    def __init__(self, filenames, transform=to_float):
        self.filenames = filenames
        self.transform = transform
        self.build_label_encoder()


class TrainDataSet(LabeledDataSet):

    def __init__(self, transform=to_float, blacklist_patterns=[]):
        super().__init__(train_folder, transform, blacklist_patterns)


class PlayDataSet(LabeledDataSet):

    def __init__(self, transform=to_float, blacklist_patterns=[]):
        super().__init__(play_folder, transform, blacklist_patterns)


def make_train_validate_loaders(data_set, batch_size=1, test_size=0.2):
    """ Takes a data set, and returns a loader for the training and validating.
    Both have the same proportions of the different classes
    Apply the transformation to everything already"""
    X_all = []
    y_all = []
    for data_entry in data_set:
        X_all.append(data_entry[0])
        y_all.append(data_entry[1])

    X_train, X_validate, y_train, y_validate = train_test_split(
                                                X_all, y_all, test_size=test_size, random_state=0)

    Xy_train = list(zip(X_train, y_train))
    Xy_validate = list(zip(X_validate, y_validate))

    train_loader = torch.utils.data.DataLoader(Xy_train, batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(Xy_validate, batch_size, shuffle=True)

    return train_loader, validate_loader


def make_low_memory_train_validate_loaders(data_set, batch_size=1, test_size=0.2):
    """Same as above, but doesn't apply transformation yet.
    Uses much less RAM and faster, but network will be faster"""
    y_all = [data_set.get_label(x) for x in range(len(data_set))]
    
    X_train, X_validate, _, _ = train_test_split(data_set.filenames, y_all, test_size=test_size, random_state=0)

    train_dataset = LabeledDataSetFromNames(X_train, data_set.transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    
    validate_dataset = LabeledDataSetFromNames(X_validate, data_set.transform)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size, shuffle=True)

    return train_loader, validate_loader


def make_low_memory_balanced_loaders(data_set, batch_size=1, test_size=0.2):
    """Same as above, but validation set has equal number of every class"""
    n_classes = len(data_set.label_encoder.classes_)
    names_by_label = [[] for _ in range(n_classes)]
    for k in range(len(data_set)):
        names_by_label[data_set.get_label(k)].append(data_set.filenames[k])
        
    test_filenames = []
    size = int(test_size * len(data_set) / n_classes)
    for k in range(n_classes):
        file_indices = np.random.choice(names_by_label[k], size=size, replace=False)
        test_filenames += file_indices.tolist()

    train_filenames = list(set(data_set.filenames).difference(set(test_filenames)))
    train_dataset = LabeledDataSetFromNames(train_filenames, data_set.transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    validate_dataset = LabeledDataSetFromNames(test_filenames, data_set.transform)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size, shuffle=True)
    
    bins = np.array([len(names) for names in names_by_label])
    weight = 1 / bins
    weight /= np.mean(weight) 

    return train_loader, validate_loader, torch.FloatTensor(weight)


def make_balanced_loaders_extended(data_set, batch_size=1, test_size=0.2, max_rep=5):
    """Same as above, but duplicates the entries of the least common classes in the trainer"""
    # Bin the file names by class
    n_classes = len(data_set.label_encoder.classes_)
    names_by_label = [[] for _ in range(n_classes)]
    for k in range(len(data_set)):
        names_by_label[data_set.get_label(k)].append(data_set.filenames[k])
    
    # Pick the file names for the validation set
    test_filenames = []
    size = int(test_size * len(data_set) / n_classes)
    for k in range(n_classes):
        file_indices = np.random.choice(names_by_label[k], size=size, replace=False)
        test_filenames += file_indices.tolist()
        
    # Pick remaining names (possibly multiple times) for training set
    largest_class = max(len(names) for names in names_by_label)
    train_filenames = []
    for names_of_label in names_by_label:
        remaining_names = list(set(names_of_label).difference(set(test_filenames)))
        multiple = min(largest_class // len(remaining_names), max_rep)
        train_filenames += (remaining_names * multiple)

    train_dataset = LabeledDataSetFromNames(train_filenames, data_set.transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    validate_dataset = LabeledDataSetFromNames(test_filenames, data_set.transform)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size, shuffle=True)

    return train_loader, validate_loader


def make_test_loader(data_set, batch_size):
    return torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)


def write_csv_submission(file_name, answers):
    """
    file name is the local name of the submission
    Answers is a list of tuples (file_id, predicted class name)
    """
    full_file_name = os.path.join(submission_folder, file_name+".csv")

    with open(full_file_name, 'w', newline='') as writeFile:
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(writeFile, fieldnames=fieldnames, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for answer in answers:
            writer.writerow({'Id': answer[0], 'Predicted': answer[1]})
