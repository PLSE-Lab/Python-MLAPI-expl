# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
encoded_pixel = "46337 1 46593 4 46849 5 47105 7 47361 9 47617 10 47873 11 48129 11 48385 11 48641 12 48897 12 49153 13 49409 12 49665 12 49921 11 50177 11 50433 10 50689 10 50945 9 51201 9 51457 7 51715 3"

def drawEnc(encoded, row = 256, column = 256):
    start_p = None
    counter = 0
    img = np.zeros((row, column, 3), np.uint8)
    for item in encoded.split():
        counter += 1
        item = int(item)
        if counter%2 != 0:
            start_p = item
        else:
            for point in range(start_p, start_p + item):
                loc = divmod((point-1), row)
                cv2.line(img, loc, loc, (255, 255, 255), 1)
    return cv2.imwrite('out.png', img)


rle = drawEnc(encoded_pixel)