# https://github.com/Hsankesara/DeepResearch/blob/master/Prototypical_Nets/prototypicalNet.py
import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from time import sleep
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import multiprocessing as mp


class Net(nn.Module):
    """
    Image2Vector CNN which takes image of dimension (28x28x3) and return column vector length 64
    """

    def sub_block(self, in_channels, out_channels=64, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                            out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        return block

    def __init__(self):
        super(Net, self).__init__()
        self.convnet1 = self.sub_block(3)
        self.convnet2 = self.sub_block(64)
        self.convnet3 = self.sub_block(64)
        self.convnet4 = self.sub_block(64)

    def forward(self, x):
        x = self.convnet1(x)
        x = self.convnet2(x)
        x = self.convnet3(x)
        x = self.convnet4(x)
        x = torch.flatten(x, start_dim=1)
        return x


class PrototypicalNet(nn.Module):
    def __init__(self, use_gpu=False):
        super(PrototypicalNet, self).__init__()
        self.f = Net()
        self.gpu = use_gpu
        if self.gpu:
            self.f = self.f.cuda()

    def forward(self, datax, datay, Ns, Nc, Nq, total_classes):
        """
        Implementation of one episode in Prototypical Net
        datax: Training images
        datay: Corresponding labels of datax
        Nc: Number  of classes per episode
        Ns: Number of support data per class
        Nq:  Number of query data per class
        total_classes: Total classes in training set
        """
        k = total_classes.shape[0]
        K = np.random.choice(total_classes, Nc, replace=False)
        Query_x = torch.Tensor()
        if(self.gpu):
            Query_x = Query_x.cuda()
        Query_y = []
        Query_y_count = []
        centroid_per_class = {}
        class_label = {}
        label_encoding = 0
        for cls in K:
            S_cls, Q_cls = self.random_sample_cls(datax, datay, Ns, Nq, cls)
            centroid_per_class[cls] = self.get_centroid(S_cls, Nc)
            class_label[cls] = label_encoding
            label_encoding += 1
            # Joining all the query set together
            Query_x = torch.cat((Query_x, Q_cls), 0)
            Query_y += [cls]
            Query_y_count += [Q_cls.shape[0]]
        Query_y, Query_y_labels = self.get_query_y(
            Query_y, Query_y_count, class_label)
        Query_x = self.get_query_x(Query_x, centroid_per_class, Query_y_labels)
        return Query_x, Query_y

    def random_sample_cls(self, datax, datay, Ns, Nq, cls):
        """
        Randomly samples Ns examples as support set and Nq as Query set
        """
        data = datax[(datay == cls).nonzero()]
        perm = torch.randperm(data.shape[0])
        idx = perm[:Ns]
        S_cls = data[idx]
        idx = perm[Ns: Ns+Nq]
        Q_cls = data[idx]
        if self.gpu:
            S_cls = S_cls.cuda()
            Q_cls = Q_cls.cuda()
        return S_cls, Q_cls

    def get_centroid(self, S_cls, Nc):
        """
        Returns a centroid vector of support set for a class
        """
        return torch.sum(self.f(S_cls), 0).unsqueeze(1).transpose(0, 1) / Nc

    def get_query_y(self, Qy, Qyc, class_label):
        """
        Returns labeled representation of classes of Query set and a list of labels.
        """
        labels = []
        m = len(Qy)
        for i in range(m):
            labels += [Qy[i]] * Qyc[i]
        labels = np.array(labels).reshape(len(labels), 1)
        label_encoder = LabelEncoder()
        Query_y = torch.Tensor(
            label_encoder.fit_transform(labels).astype(int)).long()
        if self.gpu:
            Query_y = Query_y.cuda()
        Query_y_labels = np.unique(labels)
        return Query_y, Query_y_labels

    def get_centroid_matrix(self, centroid_per_class, Query_y_labels):
        """
        Returns the centroid matrix where each column is a centroid of a class.
        """
        centroid_matrix = torch.Tensor()
        if(self.gpu):
            centroid_matrix = centroid_matrix.cuda()
        for label in Query_y_labels:
            centroid_matrix = torch.cat(
                (centroid_matrix, centroid_per_class[label]))
        if self.gpu:
            centroid_matrix = centroid_matrix.cuda()
        return centroid_matrix

    def get_query_x(self, Query_x, centroid_per_class, Query_y_labels):
        """
        Returns distance matrix from each Query image to each centroid.
        """
        centroid_matrix = self.get_centroid_matrix(
            centroid_per_class, Query_y_labels)
        Query_x = self.f(Query_x)
        m = Query_x.size(0)
        n = centroid_matrix.size(0)
        # The below expressions expand both the matrices such that they become compatible to each other in order to caclulate L2 distance.
        # Expanding centroid matrix to "m".
        centroid_matrix = centroid_matrix.expand(
            m, centroid_matrix.size(0), centroid_matrix.size(1))
        Query_matrix = Query_x.expand(n, Query_x.size(0), Query_x.size(
            1)).transpose(0, 1)  # Expanding Query matrix "n" times
        Qx = torch.pairwise_distance(centroid_matrix.transpose(
            1, 2), Query_matrix.transpose(1, 2))
        return Qx


def train_step(optimizer, protonet, datax, datay, Ns, Nc, Nq):
    optimizer.zero_grad()
    Qx, Qy = protonet(datax, datay, Ns, Nc, Nq, np.unique(datay))
    pred = torch.log_softmax(Qx, dim=-1)
    loss = F.nll_loss(pred, Qy)
    loss.backward()
    optimizer.step()
    acc = torch.mean((torch.argmax(pred, 1) == Qy).float())
    return loss, acc


def test_step(protonet, datax, datay, Ns, Nc, Nq):
    Qx, Qy = protonet(datax, datay, Ns, Nc, Nq, np.unique(datay))
    pred = torch.log_softmax(Qx, dim=-1)
    loss = F.nll_loss(pred, Qy)
    acc = torch.mean((torch.argmax(pred, 1) == Qy).float())
    return loss, acc


def load_weights(filename, protonet, use_gpu):
    if use_gpu:
        protonet.load_state_dict(torch.load(filename))
    else:
        protonet.load_state_dict(torch.load(filename), map_location='cpu')
    return protonet



def image_rotate(img, angle):
    """
    Image rotation at certain angle. It is used for data augmentation
    """
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return np.expand_dims(dst, 0)


def read_alphabets(alphabet_directory):
    """
    Reads all the characters from alphabet_directory and augment each image with +-5 degrees of rotation.
    """
    datax = None
    datay = []
    characters = os.listdir(alphabet_directory)
    if len(characters) < 2:
        return [],[]
    for character in characters:
        #print(":",character)
        #images = os.listdir(alphabet_directory + character + '/')
        #for img in images:
        img = cv2.imread( alphabet_directory + character )
        if np.shape(img)==(): continue
        img = img[80:170,80:170] # crop inner 90x90
        image = cv2.resize(img, (28, 28))
        #image2 = image_rotate(image, -5)
        # image3 = image_rotate(image, 5)
        image = np.expand_dims(image, 0)
        if datax is None:
            datax = np.vstack([image])
            #datax = np.vstack([image, image2, image3])
            #datax = np.vstack([image, image2])
        else:
            datax = np.vstack([datax, image])
            #datax = np.vstack([datax, image, image2, image3])
            #datax = np.vstack([datax, image, image2])
        datay.append(alphabet_directory)
        #datay.append(alphabet_directory)
        #datay.append(alphabet_directory)
    return datax, np.array(datay)


def read_images(base_directory):
    """
    Used multithreading for data reading to decrease the reading time drastically
    """
    datax = None
    datay = []
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(read_alphabets, args=(
        base_directory + '/' + directory + '/', )) for directory in os.listdir(base_directory)]
    pool.close()
    for result in results:
        if result[0]==[]:
          continue
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.vstack([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    return datax, datay


#from google.colab.patches import cv2_imshow

def main():
    # i1 = np.ones((300,300,3),dtype=np.uint8)
    # print(i1.shape)
    # cv2_imshow(i1)

    # Reading the data
    trainx, trainy = read_images('lfw-deepfunneled/')
    #trainx, trainy = read_images('images_background/')
    #testx, testy = read_images('images_evaluation/')
    # Checking if GPU is available
    use_gpu = torch.cuda.is_available()
    # Converting input to pytorch Tensor
    trainx = torch.from_numpy(trainx).float()
    # testx = torch.from_numpy(testx).float()
    if use_gpu:
        trainx = trainx.cuda()
        # testx = testx.cuda()
    # Printing the data
    print(trainx.size(), len(np.unique(trainy)), use_gpu)
    # Set training iterations and display period
    num_episode = 75000
    frame_size = 500
    trainx = trainx.permute(0, 3, 1, 2)
    # testx = testx.permute(0, 3, 1, 2)

    # Initializing prototypical net
    protonet = PrototypicalNet(use_gpu)
    optimizer = optim.Adam(protonet.parameters(), lr = 0.005)#, momentum=0.99)

    # Training loop
    frame_loss = 0
    frame_acc = 0
    for i in range(num_episode):
        loss, acc = train_step(optimizer, protonet, trainx, trainy, 5, 60, 5)
        frame_loss += loss.data
        frame_acc += acc.data
        if((i+1) % frame_size == 0):
            print("Frame Number:", ((i+1) // frame_size), 'Frame Loss: ', frame_loss.data.cpu().numpy().tolist() /
                  frame_size, 'Frame Accuracy:', (frame_acc.data.cpu().numpy().tolist() * 100) / frame_size)
            frame_loss = 0
            frame_acc = 0

            input_names = [ "input1" ]
            output_names = [ "output1" ]
            outname = "lfw_proto_%d.onnx" %  ((i+1) // frame_size)
            torch.onnx.export(protonet.f, trainx, outname, verbose=False, input_names=input_names, output_names=output_names)


"""
    # Test loop
    num_test_episode = 2000
    avg_loss = 0
    avg_acc = 0
    for _ in range(num_test_episode):
        loss, acc = test_step(protonet, testx, testy, 5, 60, 15)
        avg_loss += loss.data
        avg_acc += acc.data
    print('Avg Loss: ', avg_loss.data.cpu().numpy().tolist() / num_test_episode,
          'Avg Accuracy:', (avg_acc.data.cpu().numpy().tolist() * 100) / num_test_episode)
"""
main()
