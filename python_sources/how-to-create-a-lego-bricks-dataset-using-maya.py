#!/usr/bin/env python
# coding: utf-8

# ![](https://raw.githubusercontent.com/JoostHazelzet/rawimages/master/HighRes_LEGO_bricksloose.jpg)
# *Copyright The LEGO Group*

# # How to create a LEGO bricks dataset using Maya

# ## Introduction
# In the summer of 2018 I decided to create a dataset containing computer rendered LEGO bricks. I used a small set of digital bricks and next made images of the brick while it was rotated in the x, y and z axis. At that time I used Blender as the rendering software. Based on the feedback I got from Kaggler users, thanks for this, I decided to create a version 2 dataset. This time I am going to use Maya software from Autodesk (I used the Maya 2020 trial version) which is in my opinion easier to use. In this notebook I explain step by step how I created the dataset.

# ## Creating the set of digital Lego bricks
# I decided to use [Mecabricks](https://www.mecabricks.com/) to selected and generate digital models of the LEGO bricks. Each brick is exported as Collada (.dae) file because this is file format recognized by Autodesk Maya. I decided to take middle gray tone as color (so centered approximately around half RGB range 0..255) because if color is required then this can be applied later on the finished rendered image.
# 
# A list is provided of all Collada files in the [Kaggle Lego Bricks dataset](https://www.kaggle.com/joosthazelzet/lego-brick-images/).

# ## Setting up the scene in Maya using 2 cameras
# The approach is to load a digital brick to a pivot point. The brick is rotated stepwise around the x, y and z axis while rendering and save an image at each step. 
# 
# <div>
# <img src="https://raw.githubusercontent.com/JoostHazelzet/rawimages/master/scene.png" align="left" width="400"/>
# </div>

# I decided to use two cameras. The reason is to solve an issue that occurs with one camera: it is not possible to determine the brick type correctly. See the following example where it is difficult to predict if this is a brick of a plate:
# <div>
# <img src="https://raw.githubusercontent.com/JoostHazelzet/rawimages/master/3001L.png" align="left" width="200"/>
# </div>
# 

# This issue especially happens if a brick image is taken from the top with one camera. However, using a second offset camera, it becomes clear that this is a brick:
# <div>
# <img src="https://raw.githubusercontent.com/JoostHazelzet/rawimages/master/3001R.png" align="left" width="200"/>
# </div>

# Finally, we need to take care of the lighting. I decided to use a softbox setup using a square light. A softbox diffuses the light into a pleasing soft, even light. The softbox is depicted as the raster:
# <div>
# <img src="https://raw.githubusercontent.com/JoostHazelzet/rawimages/master/perspective-view.png" align="left" width="600"/>
# </div>

# The scene setup in Maya is created using a Python script in order to be able to regenerate it.

# In[ ]:


'''
Run this code in the AutoDesk Maya Python Script Editor!
Start with a new empty scene. The script will setup the cameras and softbox lighting for you.
Copyright Joost Hazelzet, script is provided under MIT License.
'''

import maya.cmds as cmds
from mtoa.cmds.arnoldRender import arnoldRender
import mtoa.ui.arnoldmenu as arnoldmenu; 
import mtoa.utils as mutils
from os import listdir
from os.path import isfile, join
import random

#Setup the cameras
cmds.camera(centerOfInterest=5, focalLength=170, lensSqueezeRatio=1, cameraScale=1, horizontalFilmAperture=1.41732, 
        horizontalFilmOffset=0, verticalFilmAperture=0.94488, verticalFilmOffset=0, filmFit='fill', overscan=1, 
        motionBlur=0, shutterAngle=144, nearClipPlane=0.1, farClipPlane=10000, orthographic=0, orthographicWidth=30,
        panZoomEnabled=0, horizontalPan=0, verticalPan=0, zoom=1)
nameCameraRight = cmds.ls(selection=True)[0]
cmds.move( 19., 0., 19., r=False )
cmds.rotate( 0., 45., 0., r=False )

cmds.camera(centerOfInterest=5, focalLength=170, lensSqueezeRatio=1, cameraScale=1, horizontalFilmAperture=1.41732, 
        horizontalFilmOffset=0, verticalFilmAperture=0.94488, verticalFilmOffset=0, filmFit='fill', overscan=1, 
        motionBlur=0, shutterAngle=144, nearClipPlane=0.1, farClipPlane=10000, orthographic=0, orthographicWidth=30,
        panZoomEnabled=0, horizontalPan=0, verticalPan=0, zoom=1)
nameCameraLeft = cmds.ls(selection=True)[0]
cmds.move( -19., 0., 19., r=False )
cmds.rotate( 0., -45., 0., r=False )

print(nameCameraRight, nameCameraLeft)


#Setup the lighting
cmds.polyPlane(width=1, height=1, subdivisionsX=10, subdivisionsY=10, axis=[0,1,0], createUVs=2,constructionHistory=True)
mutils.createMeshLight()
nameLight = cmds.ls(selection=True)[0]
cmds.setAttr(nameLight+"Shape.intensity", 10)
cmds.setAttr(nameLight+"Shape.aiExposure", 5)
cmds.setAttr(nameLight+".scaleX", 20)
cmds.setAttr(nameLight+".scaleZ", 20)
cmds.setAttr(nameLight+".rotateX", -90)
cmds.setAttr(nameLight+".translateZ", 3)

print(nameLight)


# The folowing Python script set processes the digital bricks found from the indicated path. Each brick is 400 times rotated randomly in the x, y and z axis. This results in 400 unique frames and, because of the 2 cameras setup, a total of 800 images are rendered. Each frame is saved as png image and has a left (L) and right (R) indicator in the file.

# In[ ]:


'''
Run this code in the AutoDesk Maya Python Script Editor!
Be sure to run the scene setup first including the necessary import statements.
Copyright Joost Hazelzet, script is provided under MIT License.
'''

#Prepare the renderer
cmds.setAttr("defaultArnoldDriver.ai_translator", "png", type="string")
random.seed(1)

def RenderLegoBrick(pathBricks, frames, cameraRight, cameraLeft, test):
    if test:
        brickFiles = ['3001 brick 2x4.dae']
    else:
        brickFiles = [f for f in listdir(pathBricks) if isfile(join(pathBricks, f))]

    print('{0} brick models found.'.format(len(brickFiles)))
    processed=0
    images=0
    
    for brickFile in brickFiles:
        
        print('Process Collada file: '+brickFile)
        brickName = brickFile.split('.')[0]
       
        #Remove all former brick parts if any
        if cmds.ls('Part_*') != []:
            cmds.select('Part_*')
            cmds.delete()
        
        #Import the Collada file. This generates a Part_.. node in the scene
        daeFile = pathBricks+brickFile
        cmds.file(daeFile, type="DAE_FBX", i=True, ra=True, ignoreVersion=True, options="v=0;", 
                importTimeRange="combine", pr=True, mergeNamespacesOnClash=True )
  
        #Set the reflectivity, color and center the Part node
        cmds.select('Part_*', r=True)
        part = cmds.ls(sl=True,long=False)[0] 
        phong = cmds.defaultNavigation(defaultTraversal=True, destination=part+'*.surfaceShader')[0]
        if test:
            print('Internal Maya part: '+part)
            print('Phong of part: '+phong)
        cmds.setAttr(phong+'.reflectivity', 0.24)
        cmds.setAttr(phong+'.color', 0.5, 0.5, 0.5, type='float3')
        cmds.xform(centerPivots=True)
        cmds.move( 0., 0., 0., rpr=True ) #Move pivot to center of brick

        #Rotate the Part node randomly along the x, y and z axis, render via the 2 cameras and saves the result
        for i in range(frames):
            rotx = random.random()*360
            roty = random.random()*360
            rotz = random.random()*360
            cmds.rotate( rotx, roty, rotz, r=False )
            fileName = "{0} {1:03d}".format(brickName, i)
            if test:
                print(i, rotx, roty, rotz)
            else:
                cmds.setAttr("defaultArnoldDriver.pre", fileName+"R", type="string")
                arnoldRender(400, 400, False, False, cameraRight, ' -layer defaultRenderLayer')
                cmds.setAttr("defaultArnoldDriver.pre", fileName+"L", type="string")
                arnoldRender(400, 400, False, False, cameraLeft, ' -layer defaultRenderLayer')
        processed += 1
        images += frames*2
        
    print('{0} brick models processed.'.format(processed))
    print('{0} images generated.'.format(images))


# Use this command to start the rendering process. 
# The import statements, nameCameraRight and nameCameraLeft are defined in the previous scene setup script.
RenderLegoBrick('C:/Users/joost/Documents/LEGO Creations/Original Bricks/', 400, nameCameraRight, nameCameraLeft, False)


# ## Post processing
# Probably it is me but I had 2 issues with the rendering process I couldn't resolve in Maya itself:
# 1. The background turns black while I was expecting it to be transparent.
# 2. The files are saved with a \_1.png while I was expecting without \_1.   
# I except the first issue as given, the second issue is crrected using this small script.

# In[ ]:


#Remove _1 from file
import os, sys
from ipywidgets import IntProgress
from IPython.display import display

pathRename = 'C:/Users/joost/Documents/maya/projects/lego_bricks/images/tmp/'

files = [f for f in os.listdir(pathRename) if os.path.isfile(os.path.join(pathRename, f))]
pbar = IntProgress(min=0, max=len(files)) # instantiate the bar
display(pbar) # display the bar

for f in files:
    lstr = len(f)
    start = f[:lstr-6]
    purge = f[lstr-6:lstr-4]
    end = f[lstr-4:lstr]
    newName = f'{start}{end}'
    if purge == '_1':
        os.rename(f'{pathRename}{f}', f'{pathRename}{newName}')  
    pbar.value +=1


# ## Conclusion
# I used 50 digital Collada models of LEGO bricks and rendered each brick 800 times (400 Left camera images and 400 Left camera images). This resulted in a total of 40.000 images in the dataset.
# Because all images are randomly generated, you have quite some freedom in chosing the validation set. The following script creates a csv file with the 80 frames (=20% of 400 frames) selected of each brick.

# In[ ]:


import os, sys

# Path where all 50 Collada files are stored
path = 'C:/Users/joost/Documents/LEGO Creations/Original Bricks/processed'

txt= open('C:/Users/joost/Documents/LEGO Creations/Original Bricks/validation.txt','w')
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
for f in files:
    brickName = f.split('.')[0]
    for i in range(80):
        fileName = "{0} {1:03d}".format(brickName, i)
        txt.write(fileName+"R.png\n")
        txt.write(fileName+"L.png\n")
txt.close()


# 
