import os
import os.path as path
import json
import matplotlib.pyplot as plt
import cv2
import shutil

# Auxiliar functions
def _reescale(image, size=256):
    width = size
    height = size
    
    dimensions = (width, height)
    
    image = cv2.resize(image, dsize=dimensions, interpolation = cv2.INTER_AREA)
    
    return image


def _fix_rgb(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def _preprocess(image):
    image = _reescale(image, 256)
    
    image = _fix_rgb(image)
    
    return image

def provide_monet_dataset():
    # Dowload the script that will manage the data procurement
    # https://github.com/lucasdavid/wikiart
    os.system("git clone https://github.com/lucasdavid/wikiart.git")
    os.system("python3 wikiart/wikiart.py --datadir ./wikiart-saved/ fetch --only artists")

    # Metadata to indicate the artist to be downloaded
    entry = '''[{
        "artistName": "Claude Monet",
        "birthDay": "/Date(-4074969600000)/",
        "birthDayAsString": "November 14, 1840",
        "contentId": 211667,
        "deathDay": "/Date(-1359331200000)/",
        "deathDayAsString": "December 5, 1926",
        "dictonaries": [
            1221,
            316
        ],
        "image": "https://uploads0.wikiart.org/00115/images/claude-monet/440px-claude-monet-1899-nadar-crop.jpg!Portrait.jpg",
        "lastNameFirst": "Monet Claude",
        "url": "claude-monet",
        "wikipediaUrl": "http://en.wikipedia.org/wiki/Claude_Monet"
    }]'''

    with open('./wikiart-saved/meta/artists.json', 'w') as artists_file:
        artists_file.write(entry)

    # Download the art from the artist
    os.system("python3 wikiart/wikiart.py --datadir ./wikiart-saved/ fetch")

    # Filepath of all files retrieval
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk('wikiart-saved/images/claude-monet'):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))

    images = []

    for file in files:
        images.append(_preprocess(cv2.imread(file)))

    # An array with al the read fieles is returned
    return images

def clean_wikiart_files():
    os.system("rm -rf wikiart wikiart-saved")
    
def store_images(images, directory_path='monet_dataset'):
    os.makedirs(directory_path, exist_ok=True)
    
    for i, image in enumerate(images):
        plt.imsave(os.path.join(directory_path, "{:04d}.png".format(i)), image)
    
if __name__=='__main__':
    columns = 5
    rows = 6
    
    images = provide_monet_dataset()
    
    store_images(images)
    
    clean_wikiart_files()
    
    shutil.make_archive('monet_dataset', 'zip', './monet_dataset')