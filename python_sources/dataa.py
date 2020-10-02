import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import utils 
import numpy as np 

from PIL import Image

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, min_length, max_length, color=[0,0,0]):
        self.n_holes = n_holes
        self.length = (min_length, max_length)
        self.color = np.array(color)

    def __call__(self, img):
        #w,h = img.width, img.height
        data = np.asarray(img)
        w,h = data.shape[:2]
        
        mask = np.ones_like(img)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            length = np.random.randint( self.length[0], self.length[1])
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2, :] = self.color[None,None, :]
        data = mask * data
        
        return Image.fromarray(data, mode=img.mode)




class ImageDataset(object):
    def __init__(self, dataset, batch_size, test_batch_size, path='./data', mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768], preprocesses=[], num_workers=1):
        Dataset = dataset
        
        normalize = transforms.Normalize(mean, std)
        
        # preprocessing of training data
        transform = transforms.Compose(preprocesses + [
            transforms.ToTensor(),
            normalize,
        ])

        self.trainset = Dataset(root=path, train=True, transform=transform, download=True)
        
        self.train = t.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)

        self.testset = Dataset(root=path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    
        self.test = t.utils.data.DataLoader(
            self.testset,
            batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        

        #move to cuda
        self.train = utils.DeviceDataLoader(self.train)
        self.test = utils.DeviceDataLoader(self.test)


def augmentify( augment, params ):
    
    preprocesses = []
    if augment is None:
        pass
    elif 'flip-rot' in augment:
            preprocesses = [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(5),
            ]
    elif 'pad-crop' in augment:
            preprocesses = [
                    transforms.Pad(8, fill=0),
                    transforms.RandomCrop(32, pad_if_needed=True)
            ]
    elif 'cutout' in augment:
            preprocesses = [
                    Cutout(1, 3, 7),        
            ]
    else: assert False, "augmentation %s not valid" % augment
    params['preprocesses'] = preprocesses
    return params
    
def CIFAR10(augment=None, **params):
    params = augmentify( augment, params )
    return ImageDataset( datasets.CIFAR10, **params )
    
def CIFAR100(augment=None, **params):
    params = augmentify( augment, params )
    return ImageDataset( datasets.CIFAR100, **params )
