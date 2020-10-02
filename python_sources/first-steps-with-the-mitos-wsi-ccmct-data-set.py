#!/usr/bin/env python
# coding: utf-8

# ## First steps with the MITOS WSI CCMCT data set
# 
# In this short tutorial, you will see how to use the database and the DICOM images of this data set.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import sqlite3
DB = sqlite3.connect('/kaggle/input/mitosis-wsi-ccmct-training-set/MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite')
cur = DB.cursor()


# In[ ]:


import numpy as np
from pydicom.encaps import decode_data_sequence
from PIL import Image
import io
import pydicom

class ReadableDicomDataset():
    def __init__(self, filename):
        self._ds = pydicom.dcmread(filename)
        self.geometry_imsize = (self._ds[0x48,0x6].value,self._ds[0x48,0x7].value)
        self.geometry_tilesize = (self._ds.Columns, self._ds.Rows)
        self.geometry_columns = round(0.5+(self.geometry_imsize[0]/self.geometry_tilesize[0]))
        self.geometry_rows = round(0.5 + (self.geometry_imsize[1] / self.geometry_tilesize[1] ))
        self._dsequence = decode_data_sequence(self._ds.PixelData)


    def imagePos_to_id(self, imagePos:tuple):
        id_x, id_y = imagePos
        return (id_x+(id_y*self.geometry_columns))
    
    def get_tile(self, pos):
        return np.array(Image.open(io.BytesIO(self._dsequence[pos])))
        

    def get_id(self, pixelX:int, pixelY:int) -> (int, int, int):

        id_x = round(-0.5+(pixelX/self.geometry_tilesize[1]))
        id_y = round(-0.5+(pixelY/self.geometry_tilesize[0]))

        return (id_x,id_y), pixelX-(id_x*self.geometry_tilesize[0]), pixelY-(id_y*self.geometry_tilesize[1]),

    @property
    def dimensions(self):
        return self.geometry_imsize
        
    def read_region(self, location: tuple, size:tuple):
        lu, lu_xo, lu_yo = self.get_id(*list(location))
        rl, rl_xo, rl_yo = self.get_id(*[sum(x) for x in zip(location,size)])
        # generate big image
        bigimg = np.zeros(((rl[1]-lu[1]+1)*self.geometry_tilesize[0], (rl[0]-lu[0]+1)*self.geometry_tilesize[1], self._ds[0x0028, 0x0002].value), np.uint8)
        for xi, xgridc in enumerate(range(lu[0],rl[0]+1)):
            for yi, ygridc in enumerate(range(lu[1],rl[1]+1)):
                if (xgridc<0) or (ygridc<0):
                    continue
                bigimg[yi*self.geometry_tilesize[0]:(yi+1)*self.geometry_tilesize[0],
                       xi*self.geometry_tilesize[1]:(xi+1)*self.geometry_tilesize[1]] = \
                       self.get_tile(self.imagePos_to_id((xgridc,ygridc)))
        # crop big image
        return bigimg[lu_yo:lu_yo+size[1],lu_xo:lu_xo+size[0]]

    


# In[ ]:


ds = ReadableDicomDataset('/kaggle/input/mitosis-wsi-ccmct-training-set/fff27b79894fe0157b08.dcm')
location=(69700,17100)
size=(500,500)
img = Image.fromarray(ds.read_region(location=location,size=size))
img


# In[ ]:


# Get the annotation coordinates, offset by the left upper coordinate (location)
# NOTE: We would actually have to check the label class - which we omit for the sake of simplicity here
cells = cur.execute(f"""SELECT coordinateX-{location[0]}, coordinateY-{location[1]}
                        from Annotations_coordinates where slide==7 and 
                        coordinateX>{location[0]} and coordinateX<{location[0]+size[0]} and 
                        coordinateY>{location[1]} and coordinateY<{location[1]+size[1]}""").fetchall()

from PIL import ImageDraw
draw = ImageDraw.Draw(img)
for (cx,cy) in cells:
    r=25
    draw.ellipse([(cx-r,cy-r),(cx+r,cy+r)],outline=(255,0,0,255))
img


# Et voila - here we have found two annotated cells, in this case mitotic figures. Note that we did not check for the class in this code.
