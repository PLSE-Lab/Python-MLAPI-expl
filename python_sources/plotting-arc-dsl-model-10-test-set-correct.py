#!/usr/bin/env python
# coding: utf-8

# # The Notebook
# This notebook takes the solution I had created for the ARC challenge and gives users an interface to select different tasks, to see how it coped with each one. Spoiler: not well.
# 
# Although I was scoring about 10% success in the train and task sets, I only achieved an LB score of 0.99 with this. I write "DSL" in quotation marks in the title because although my solution loosely alligns with DSL solutions, it may not be a strict example of a DSL. Essentially, I wrote classes to contain embeddings of the images, and functions to operate on these.
# 
# 
# 
# 
# # How it works
# I broke the whole challenge into 2 sections. Some tasks were related to patterns, some to the arrangement of objects. 
# 
# **Patterns:** I had trained a NN to detect tasks which involved filling in repeating patterns. Once these had been identified, I used Paulo Pinto's [+28 Tasks notebook](https://www.kaggle.com/paulorzp/28-tasks-tiles-and-symmetry) to get a solution for these.
# 
# **Objects:** This is where the heavy lifting was done. First I created a class which contained the whole task, then within that, an "imgpair" class comprising of an input + output: within each input and output I had a "singleimg" class comprising of any objects found in that image. Finally, each object listed had various properties found: size, height, x & y location, colour, amount of holes. These were used later by a model to work out what was important for deciding how a certain rule happened. This formed the basis of object manipulation. I then created functions which identified: 1. if a certain rule took place, 2. What objects that rule applied to, 3: the specifics of that rule. For example, if objects moved around from input -> output, we'd need to know what objects moved, and the specifics of where they moved and the rules for how they move. In total, I did rules for: object colour changes, object removal, object movements, logical operations between 2 objects (and, nand, or, nor, xor, xnor) and not operators for single objects. I guess, in the end, though, this approach didn't generalise very well.
# 
# 
# 
# 
# # Success Stories
# The following tasks in the test set were completed by this approach:
# * 009d5c81.json
# * 0a2355a6.json
# * 0bb8deee.json
# * 0c9aba6e.json
# * 195ba7dc.json
# * 1990f7a8.json
# * 1c0d0a4b.json
# * 1d0a4b61.json
# * 1e97544e.json
# * 332efdb3.json
# * 34b99a2b.json
# * 37d3e8b2.json
# 
# You'll need to click "edit" in the top right to get the dropdown to work and then scroll all the way to the bottom (ignore all code other than the dropdown).

# # The inner workings: ignore the next 6 cells unless you'd like to get into the guts of the code:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cloudpickle
from skimage import measure
from matplotlib import colors
from numpy.lib.stride_tricks import as_strided
import json
from pathlib import Path

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INITIALISE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# now get the tasks
def startup(dataset='train', printsample=False):

    data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge')

    # Any results you write to the current directory are saved as output.

    training_path = data_path / 'training'
    evaluation_path = data_path / 'evaluation'
    test_path = data_path / 'test'

    training_tasks = sorted(os.listdir(training_path))
    evaluation_tasks = sorted(os.listdir(evaluation_path))
    test_tasks = sorted(os.listdir(test_path))

    alltasks = []
    tasknames = []
    
    if dataset is 'train':       
        for i in range(len(training_tasks)):
            task_file = str(training_path / training_tasks[i])
            tasknames.append(training_tasks[i].split('.')[0])

            with open(task_file, 'r') as f:
                nexttask = json.load(f)
                alltasks.append(nexttask)
    elif dataset is 'eval':
        for i in range(len(evaluation_tasks)):
            task_file = str(evaluation_path / evaluation_tasks[i])
            tasknames.append(evaluation_tasks[i].split('.')[0])

            with open(task_file, 'r') as f:
                nexttask = json.load(f)
                alltasks.append(nexttask)
    elif dataset is 'test':
        for i in range(len(test_tasks)):
            task_file = str(test_path / test_tasks[i])
            tasknames.append(test_tasks[i].split('.')[0])

            with open(task_file, 'r') as f:
                nexttask = json.load(f)
                alltasks.append(nexttask)
    else:
        print('dataset assigned to non-existent string')
        alltasks = 0

    return alltasks, tasknames


# In[ ]:


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ARC CLASSES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import cv2
import scipy.ndimage
from copy import deepcopy


class SameColourObject:
    width = None
    height = None
    colour = None
    elementarr = None
    distrelotherobjs = None
    positionabsx = None
    positionabsy = None
    holecount = 0
    holes = None

    def findholes(self, obj):
        filledobj = scipy.ndimage.morphology.binary_fill_holes(obj).astype(int)
        holes = filledobj - obj
        holes = np.uint8(holes)

        if np.count_nonzero(holes) > 0:
            #  labels is the shape of the hole. Potentially could do something with this later
            retval, labels = cv2.connectedComponents(holes)
            self.holecount = retval - 1

    def __init__(self, labels, specificcolour):
        # cut away where object isn't in the image
        validcols = labels.sum(axis=0) > 0  # logical index col
        validrows = labels.sum(axis=1) > 0  # logical index row
        firstcol = min(np.where(validcols)[0])
        firstrow = min(np.where(validrows)[0])
        lastcol = max(np.where(validcols)[0])
        lastrow = max(np.where(validrows)[0])
        validcols[firstcol:lastcol] = True
        validrows[firstrow:lastrow] = True

        # assign the top left corner
        self.positionabsx = int(firstcol)
        self.positionabsy = int(firstrow)

        self.elementarr = labels[np.ix_(validrows, validcols)]

        self.colour = specificcolour

        self.height = np.size(self.elementarr, axis=0)
        self.width = np.size(self.elementarr, axis=1)

        self.findholes(self.elementarr)


class MultiColourObject:
    width = None
    height = None
    elementarr = None
    distrelotherobjs = None
    positionabsx = None
    positionabsy = None
    samecolourobjs = []
    multicolourobjs = []

    def __init__(self, labels):
        """ takes a full-sized image, finds the object in the image, finds the top
        left hand corner of the bounding box around that object within the image and
        saves that to positionAbs, then places the boxed obj into elementarr. 
        """

        # cut away where object isn't in the image
        validcols = labels.sum(axis=0) > 0  # logical index col
        validrows = labels.sum(axis=1) > 0  # logical index row
        firstcol = min(np.where(validcols)[0])
        firstrow = min(np.where(validrows)[0])
        lastcol = max(np.where(validcols)[0])
        lastrow = max(np.where(validrows)[0])
        validcols[firstcol:lastcol] = True
        validrows[firstrow:lastrow] = True

        # assign the top left corner
        self.positionabsx = firstrow
        self.positionabsy = firstcol

        self.elementarr = labels[np.ix_(validrows, validcols)]

        self.height = np.size(self.elementarr, axis=0)
        self.width = np.size(self.elementarr, axis=1)


class SingleImagePair:
    fullinputimg = None
    fulloutputimg = None
    fullpredimg = None
    inputsamecolourobjs = []
    predoutputsamecolobjs = []  # predicted same colour objects
    predoutputmulticolobjs = []
    predoutputcanvas = None  # output canvas
    outputsamecolourobjs = []
    backgroundcol = None

    def gridrefactorobjs(self):
        """looks for periodicity and shape similarity. If something there, refactor all objs so they conform
        with this pattern
        """
        objwidths, objheights, shortestx, shortesty = [], [], [], []
        furtheestleft, furthestup, shortestxt2, shortestyt2 = 100, 100, 100, 100
        for objno, obj1 in enumerate(self.predoutputsamecolobjs):
            if len(self.predoutputsamecolobjs[objno+1:]) != 0:
                for obj2 in self.predoutputsamecolobjs[objno+1:]:
                    shortestxt1 = obj2.positionabsx - obj1.positionabsx
                    shortestyt1 = obj2.positionabsy - obj1.positionabsy

                    if (shortestxt1 > 0) & (shortestxt1 < shortestxt2) & (obj2.positionabsy == obj1.positionabsy):
                        shortestxt2 = shortestxt1

                    if (shortestyt1 > 0) & (shortestyt1 < shortestyt2) & (obj2.positionabsx == obj1.positionabsx):
                        shortestyt2 = shortestyt1

                shortestx = shortestx + [shortestxt2]
                shortesty = shortesty + [shortestyt2]
            objwidths = objwidths + [obj1.elementarr.shape[1]]
            objheights = objheights + [obj1.elementarr.shape[0]]

            if obj1.positionabsx < furtheestleft:
                furtheestleft = obj1.positionabsx
                topleftx = obj1.positionabsx
                toplefty = obj1.positionabsy

            if obj1.positionabsy < furthestup:
                furthestup = obj1.positionabsy
                toplefty = obj1.positionabsy

        mostfreqwidth = max(set(objwidths), key=objwidths.count)
        mostfreqheight = max(set(objheights), key=objheights.count)
        mostfreqx = max(set(shortestx), key=shortestx.count)
        mostfreqy = max(set(shortesty), key=shortesty.count)

        # sense-check at this point
        if (mostfreqwidth >= mostfreqx) or (mostfreqheight >= mostfreqy):
            self.gridapplied = 0
            return

        # use these numbers to set your grid & rep obj size. If you can account for all pixels in each obj: good.
        # start at top left obj
        bwfullimg = (self.fullpredimg != self.backgroundcol) * 1
        pixelscounted = 0
        xpos = topleftx
        ypos = toplefty
        newobjrefactor = []
        rowsize = 0
        objlist = []
        counter = 0
        outputcanvas = np.zeros([bwfullimg.shape[0], bwfullimg.shape[1]])

        while (ypos + mostfreqheight) <= bwfullimg.shape[0]:
            while (xpos + mostfreqwidth) <= bwfullimg.shape[1]:
                bwarr1 = bwfullimg[ypos:ypos+mostfreqheight, xpos:xpos+mostfreqwidth]  # bw array for counting obj pixels
                bwarr = deepcopy(outputcanvas)
                bwarr[ypos:ypos+mostfreqheight, xpos:xpos+mostfreqwidth] = bwarr1
                colarr = self.fullpredimg[ypos:ypos+mostfreqheight, xpos:xpos+mostfreqwidth]  # colarr for making newobj
                pixelscounted = pixelscounted + bwarr.sum()  # count the pixels

                if len(np.delete(np.unique(colarr), np.where(np.unique(colarr) == 0))) == 1:  # rem backcol & chek 1 col left
                    specificcolour = np.delete(np.unique(colarr), np.where(np.unique(colarr) == 0))[0]
                    newobj = SameColourObject(bwarr, specificcolour)
                    newobjrefactor = newobjrefactor + [newobj]
                    objlist = objlist + [counter]
                else:
                    objlist = objlist + [None]
                xpos = xpos + mostfreqx
                counter += 1
            rowsize += 1
            xpos = topleftx
            ypos = ypos + mostfreqy

        objgrid = np.array(objlist).reshape([rowsize, int(counter/rowsize)])

        if pixelscounted == bwfullimg.sum():  # if all objs are accounted for, re-factor objs into grid format
            self.predoutputsamecolobjs = newobjrefactor
            self.gridapplied = 1
            # self.objgrid = objgrid
            print('refactored by seperating by obj grid')
        else:
            self.gridapplied = 0

    def findscobjects(self, side, forgroundcols):
        """Find same colour objects in either the input or output and place that into the image
        """

        for specificcolour in forgroundcols:
            # process so we can find connected components (individual obs) in image
            if side == 'output':
                cvimg = self.fulloutputimg == specificcolour
            elif side == 'input':
                cvimg = self.fullinputimg == specificcolour

            cvimg = cvimg * 1
            cvimg = np.uint8(cvimg)

            # find individual objects
            retval, labels = cv2.connectedComponents(cvimg)
            for objs in range(1, retval):
                newobj = SameColourObject((labels == objs) * 1, specificcolour)
                if side == 'output':
                    self.outputsamecolourobjs = self.outputsamecolourobjs + [newobj]
                elif side == 'input':
                    self.inputsamecolourobjs = self.inputsamecolourobjs + [newobj]

            self.predoutputsamecolobjs = deepcopy(self.inputsamecolourobjs)

    def findbackground(self):
        countforcols = []

        # rule 1: the colour needs to be shared in both the input and the output
        uniquesin = np.unique(self.fullinputimg)
        uniquesout = np.unique(self.fulloutputimg)
        uniques = list(set(uniquesin) & set(uniquesout))

        if 0 in uniques:  # just make back background: 99% of the time it is (bad but need to write rest of code!!)
            self.backgroundcol = 0
        elif not uniques:  # empty list (how can this ever happen?! 0 - black - is a col. So...
            self.backgroundcol = None
        else:
            # rule 2: the colour is the dominent colour of the input
            for specificcolour in uniques:
                countforcols.append(np.count_nonzero(self.fullinputimg == specificcolour))

            self.backgroundcol = uniques[countforcols.index(max(countforcols))]

    def extraobjattrs(self):
        # make a list of x co-ords & y co-ords
        xstartcoords = []
        ystartcoords = []

        for obj in self.inputsamecolourobjs:
            xstartcoords = xstartcoords + [obj.positionabsx]
            ystartcoords = ystartcoords + [obj.positionabsy]

        xstartcoords.sort()
        ystartcoords.sort()

        for obj in self.inputsamecolourobjs:
            obj.xorder = next(i for i, x in enumerate(xstartcoords) if x == obj.positionabsx)
            obj.yorder = next(i for i, y in enumerate(ystartcoords) if y == obj.positionabsy)

    def __init__(self, tip, traintest, backgroundcol=None):
        """Takes a task image pair and populates all
        properties with it
        tip ---   dict, where tip['input'] is the input and
                  taskImgPair['output'] is the output
        """
        # inputs first
        self.fullinputimg = np.array(tip["input"])

        if traintest == 'train':  # need to do this now as it's used later
            self.fulloutputimg = np.array(tip["output"])

        # find unique colours
        inuniques = np.unique(self.fullinputimg)

        # assuming background is the prominent colour: find background & assign the property
        if traintest == 'train':  # if test: we'll need to get it from a train imgpair, can't do it here
            self.findbackground()
        else:
            self.backgroundcol = backgroundcol

        # find all colours other than background colours
        inforgroundcols = inuniques.tolist()
        if self.backgroundcol in inforgroundcols:
            inforgroundcols.remove(self.backgroundcol)

        # find same colour invididual objects in image
        self.findscobjects('input', inforgroundcols)

        if traintest == 'train':
            outuniques = np.unique(self.fulloutputimg)

            outforgroundcols = outuniques.tolist()
            if self.backgroundcol in outforgroundcols:
                outforgroundcols.remove(self.backgroundcol)

            self.findscobjects('output', outforgroundcols)

        # add extra attributes
        self.extraobjattrs()

        # create the first predoutputimg
        self.fullpredimg = deepcopy(self.fullinputimg)

        try:
            self.gridrefactorobjs()
        except:
            None


class FullTask:
    trainsglimgprs = []    # list of single image pairs for train set
    testinputimg = []      # list of single image pairs for test set
    testpred = None        # list containing a numpy array for a final prediction, 1 array for each input in test

    def seperatinglinerefactorobjs(self):
        """looks for line(s) which run through the entire input, separating the input into equal sized smaller portions,
        which also equal the size of the output image, suggesting some combo of those input objs to be had. Need to
        do this on a fulltask scale
        """
        for imgpair in self.trainsglimgprs:
            if imgpair.gridapplied:  # the grid obj structure messes with this one
                del imgpair.gridapplied
                return
            else:
                del imgpair.gridapplied

            stillval = 0

            # look for lines
            linesx = []
            linesy = []
            horzorvert = []
            linecol = []
            for obj in imgpair.predoutputsamecolobjs:
                if (obj.height == imgpair.fullinputimg.shape[0]) & (obj.width == 1):
                    # vert line
                    linesx = linesx + [obj.positionabsx]
                    linesy = linesy + [obj.positionabsy]
                    linecol = linecol + [obj.colour]
                    horzorvert = horzorvert + ['vert']

                if (obj.width == imgpair.fullinputimg.shape[1]) & (obj.height == 1):
                    # horz line
                    linesx = linesx + [obj.positionabsx]
                    linesy = linesy + [obj.positionabsy]
                    linecol = linecol + [obj.colour]
                    horzorvert = horzorvert + ['horz']

            if len(horzorvert) > 0:  # there are lines that run through the whole img
                # find the objects created by the seperating lines
                if horzorvert.count(horzorvert[0]) == len(horzorvert):
                    # all the same vals: either all vert or all horz
                    linesx = linesx + [imgpair.fullpredimg.shape[1]]
                    linesy = linesy + [imgpair.fullpredimg.shape[0]]
                    subimgs = []
                    subimgspositions = []
                    colsinobjs = []
                    if horzorvert[0] == 'vert':
                        startx = 0
                        # see if all objs make by splitting full img up are the same
                        for xno in linesx:
                            subimgs = subimgs + [imgpair.fullpredimg[:, startx:xno]]
                            subimgspositions = subimgspositions + [(0, imgpair.fullpredimg.shape[0], startx, xno)]
                            colsinobjs = colsinobjs + list(np.unique(subimgs))
                            startx = xno + 1

                    else:
                        starty = 0
                        # see if all objs make by splitting full img up are the same
                        for yno in linesy:
                            subimgs = subimgs + [imgpair.fullpredimg[starty:yno, :]]
                            subimgspositions = subimgspositions + [(starty, yno, 0, imgpair.fullpredimg.shape[1])]
                            colsinobjs = colsinobjs + list(np.unique(subimgs))
                            starty = yno + 1

                else:
                    # combo of vert & horz lines
                    vertlines = np.array(horzorvert) == 'vert'
                    linesy = np.array(linesy)
                    linesx = np.array(linesx)
                    vertlinesx = np.append(linesx[vertlines], imgpair.fullpredimg.shape[0])
                    horzlinesy = np.append(linesy[vertlines == False], imgpair.fullpredimg.shape[1])
                    startx, starty = 0, 0
                    for vlines in vertlinesx:
                        for hlines in horzlinesy:
                            subimgs = subimgs + [imgpair.fullpredimg[starty:hlines, startx:vlines]]  # last one
                            subimgspositions = subimgspositions + [(starty, hlines, startx, vlines)]
                            colsinobjs = colsinobjs + list(np.unique(subimgs))
                            starty = hlines + 1
                        starty = 0
                        startx = vlines + 1

                # see if the objects can be used for some sort of comparison
                backgroundcol = max(set(colsinobjs), key=colsinobjs.count)

                stillval = 1
                multicolour = 0
                ipbackcanvas = np.zeros([imgpair.fullpredimg.shape[0], imgpair.fullpredimg.shape[1]])
                for objno, eachobj in enumerate(subimgs):
                    if not ((eachobj.shape[0] == imgpair.fulloutputimg.shape[0]) &
                            (eachobj.shape[1] == imgpair.fulloutputimg.shape[1])):
                        stillval = 0

                    if len(np.unique(eachobj)) > 2:
                        multicolour = 1
                    else:
                        cols = np.unique(imgpair.fulloutputimg)
                        col = np.delete(cols, np.argwhere(cols == backgroundcol))
                        imgpair.outputsamecolourobjs = [SameColourObject((imgpair.fulloutputimg == col)*1, col[0])]
                        newlabel = deepcopy(ipbackcanvas)
                        poss = subimgspositions[objno]
                        newlabel[poss[0]:poss[1], poss[2]:poss[3]] = eachobj
                        cols = np.unique(eachobj)
                        col = np.delete(cols, np.argwhere(cols == backgroundcol))
                        subimgs[objno] = SameColourObject(newlabel, col[0])

                if stillval & multicolour:
                    imgpair.predoutputmulticolobjs = subimgs
                elif stillval & (not multicolour):
                    imgpair.predoutputsamecolobjs = subimgs

        if stillval:
            for imgpair in self.testinputimg:
                ipbackcanvas = np.zeros([imgpair.fullpredimg.shape[0], imgpair.fullpredimg.shape[1]])
                subimgstest = deepcopy(subimgs)
                for objno, poss in enumerate(subimgspositions):
                    eachobj = imgpair.fullpredimg[poss[0]:poss[1], poss[2]:poss[3]]
                    newlabel = deepcopy(ipbackcanvas)
                    newlabel[poss[0]:poss[1], poss[2]:poss[3]] = eachobj
                    cols = np.unique(eachobj)
                    col = np.delete(cols, np.argwhere(cols == backgroundcol))
                    subimgstest[objno] = SameColourObject(newlabel, col[0])

                if stillval & multicolour:
                    imgpair.predoutputmulticolobjs = subimgstest
                elif stillval & (not multicolour):
                    imgpair.predoutputsamecolobjs = subimgstest
                    print('seperating by lines, samecolobj, succeeded')

    def createoutputcanvas(self):
        """creates the 'canvas' for the test output: i.e. size of output & background col
        """
        # see if the output image is a certain scale / size relative to the input
        stillvalid = 1
        for transpose in [0, 1]:
            rm = 0
            for imgpair in self.trainsglimgprs:
                outrow = imgpair.fulloutputimg.shape[0]
                outcol = imgpair.fulloutputimg.shape[1]
                inrow = imgpair.fullinputimg.shape[transpose]
                incol = imgpair.fullinputimg.shape[1 - transpose]

                if rm == 0:  # this is the first image
                    rm = outrow // inrow
                    rc = outrow % inrow
                    cm = outcol // incol
                    cc = outcol % incol
                else:
                    if not (((inrow * rm + rc) == outrow) & ((incol * cm + cc) == outcol)):
                        stillvalid = 0

            if stillvalid:
                for trainortest in [self.trainsglimgprs, self.testinputimg]:
                    for eachtask in trainortest:
                        if eachtask.backgroundcol is None:
                            # set to 0
                            eachtask.backgroundcol = 0

                        inrow = eachtask.fullinputimg.shape[transpose]
                        incol = eachtask.fullinputimg.shape[1 - transpose]
                        eachtask.predoutputcanvas =                             np.ones([int(inrow * rm + rc), int(incol * cm + cc)]) * eachtask.backgroundcol
                return

        # see if it's a fixed size:
        stillvalid = 1
        for ii, imgpair in enumerate(self.trainsglimgprs):
            if ii == 0:
                outputshape = imgpair.fulloutputimg.shape
            else:
                if imgpair.fulloutputimg.shape != outputshape:
                    stillvalid = 0

        if stillvalid:
            print('refactored by seperating by line')
            for trainortest in [self.trainsglimgprs, self.testinputimg]:
                for eachtask in trainortest:
                    if eachtask.backgroundcol is None:
                        # set to 0
                        eachtask.backgroundcol = 0

                        eachtask.predoutputcanvas = np.ones([outputshape[0], outputshape[1]])

            return

    def findtestbackground(self):
        backgroundcols = []
        # make a list of background cols for all train sets
        for trainimgpair in self.trainsglimgprs:
            backgroundcols = backgroundcols + [trainimgpair.backgroundcol]

        # if the background is the same in all sets, assign this to the test
        if len(backgroundcols) == backgroundcols.count(backgroundcols[0]):
            return backgroundcols[0]
        else:
            return None

    def __init__(self, task_file):
        import json

        if isinstance(task_file, str):
            with open(task_file, 'r') as f:
                task = json.load(f)  # tasks is a dict
        else:  # assume we've entered the task from alltasks: dict
            task = task_file

        trainset = task['train']  # trainset is a list
        for ii in range(len(trainset)):
            ntpis = SingleImagePair(trainset[ii], 'train')
            self.trainsglimgprs = self.trainsglimgprs + [ntpis]

        testset = task['test']  # testnset is a list
        for ii in range(len(testset)):
            backgroundcol = self.findtestbackground()
            ntpis = SingleImagePair(testset[ii], 'test', backgroundcol=backgroundcol)

            self.testinputimg = self.testinputimg + [ntpis]

            self.createoutputcanvas()

            try:
                self.seperatinglinerefactorobjs()
            except:
                print('seperating objs by line failed')


class FullTaskFromClass(FullTask):
    def __init__(self, fulltask):
        # need to pack back into tips
        for imgpair in fulltask.trainsglimgprs:
            trainset = {'input': imgpair.fullpredimg.astype(int), 'output': imgpair.fulloutputimg}
            ntpis = SingleImagePair(trainset, 'train')
            ntpis.fullinputimg = imgpair.fullinputimg
            self.trainsglimgprs = self.trainsglimgprs + [ntpis]

        for imgpair in fulltask.testinputimg:
            backgroundcol = self.findtestbackground()
            testset = {'input': imgpair.fullpredimg.astype(int), 'output': imgpair.fulloutputimg}
            ntpis = SingleImagePair(testset, 'test', backgroundcol=backgroundcol)
            ntpis.fullinputimg = imgpair.fullinputimg

            self.testinputimg = self.testinputimg + [ntpis]

            self.createoutputcanvas()


# ~~~~~~~~~~~~~~~~~~~~~~~~ PATTERN CLASSES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SinglePatImagePair(SingleImagePair):
    def __init__(self, tip, test_pred_one=0):
        # input - the un-processed input pattern
        self.fullinputimg = np.array(tip["input"])

        # output - the processed input
        if len(test_pred_one) != 1:   # got a testset
            self.fulloutputimg = test_pred_one


class FullPatTask(FullTask):
    def __init__(self, task_file, test_pred_list):
        task = task_file

        trainset = task['train']  # trainset is a list
        for ii in range(len(trainset)):
            ntpis = SinglePatImagePair(trainset[ii])
            self.trainsglimgprs = self.trainsglimgprs + [ntpis]

        testset = task['test']  # testnset is a list
        for ii in range(len(testset)):
            backgroundcol = self.findtestbackground()
            ntpis = SinglePatImagePair(testset[ii], test_pred_one=test_pred_list[ii])

            self.testinputimg = self.testinputimg + [ntpis]


# In[ ]:


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PAT DETECT  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def initialisedataset():
    """This creates the X & Y dataset that we'll use to train the CNN on whether the task is pattern related or not.
    X - 2 channel image, first channel is the input picture. Second channel is the output picutre
    Y - List of booleans stating whether the problem is pattern related or not (this was manually labelled)"""
    import initialdefs
    import math

    task_file, alltasks = initialdefs.starup()

    X = np.array([[np.zeros([32, 32]), np.zeros([32, 32])]])
    Y = [0]

    # make prelim Y's - labels for which problems are patterns. Prelim because we'll make more samples from each problem
    # so we'll only use these to inform us what label we should use
    Yprelim = [0] * 400

    # from manually going through and seeing what tasks were filling in repeating patterns / mosaics
    for i in [16, 60, 73, 109, 241, 286, 304, 312, 350, 399]:
        Yprelim[i] = 1

    for taskno in range(len(alltasks)):
        print(taskno)
        task = alltasks[taskno]
        train = task['train']

        # check the input & output are the same size: if not, don't use (too different, would cause too many problems)
        check = train[0]
        checkinput = np.array(check['input'])
        checkoutput = np.array(check['output'])

        # if they are the same, we can use as sample for the model.
        if checkoutput.shape == checkinput.shape:
            for trainno in range(len(train)):
                # dim0: samples dim1: channels (2: input, out), dim3: x dim4: y
                imagepair = train[trainno]
                imageinput = imagepair['input']
                imageoutput = imagepair['output']
                sz0l = math.floor((32 - np.size(imageinput, 0))/2)  # padding for the left of dimension 0
                sz0r = math.ceil((32 - np.size(imageinput, 0))/2)  # padding for the right of dimension 0
                sz1l = math.floor((32 - np.size(imageinput, 1))/2)  # padding for the left of dimension 1
                sz1r = math.ceil((32 - np.size(imageinput, 1))/2)  # padding for the right of dimension 1
                ippad = np.pad(imageinput, ((sz0l, sz0r), (sz1l, sz1r)), constant_values=(0, 0))
                oppad = np.pad(imageoutput, ((sz0l, sz0r), (sz1l, sz1r)), constant_values=(0, 0))

                newsample = np.array([[ippad, oppad]])

                X = np.concatenate((X, newsample), axis=0)
                Y.append(Yprelim[taskno])

                # create more images from the rotated versions
                for i in range(3):
                    ippad = np.rot90(ippad)
                    oppad = np.rot90(oppad)

                    newsample = np.array([[ippad, oppad]])

                    X = np.concatenate((X, newsample), axis=0)
                    Y.append(Yprelim[taskno])

                # create more images from the transposed & rotated versions
                ippad = ippad.T
                oppad = oppad.T

                newsample = np.array([[ippad, oppad]])

                X = np.concatenate((X, newsample), axis=0)
                Y.append(Yprelim[taskno])

                for i in range(3):
                    ippad = np.rot90(ippad)
                    oppad = np.rot90(oppad)

                    newsample = np.array([[ippad, oppad]])

                    X = np.concatenate((X, newsample), axis=0)
                    Y.append(Yprelim[taskno])

    X = np.delete(X, 0, axis=0)
    Y.__delitem__(0)

    #  make channel the last dim
    X = np.moveaxis(X, 1, -1)

    return X, Y


def modelbuildtrain(X, Y):
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

    buildcomplex = True

    if buildcomplex:
        #  build model - complex
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=(32, 32, 2)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    else:
        # Build the model - simple
        model = Sequential([
            Conv2D(8, 3, input_shape=(32, 32, 2)),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(1, activation='sigmoid'),
        ])

    model.compile(
        'adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    # make data
    Xtrain = X[0:5500, :, :, :]
    Xtest = X[5501:, :, :, :]
    Ytrain = Y[0:5500]
    Ytest = Y[5501:]

    # Train the model.
    model.fit(
        Xtrain,
        Ytrain,
        epochs=3,
        validation_data=(Xtest, Ytest)
    )

    return model


def preparetaskformodel(task):
    """takes one image pair and transforms it into the correct format to be presented to model
    imagepair --    an image pair example from a task
    newsample --    a sample to be presented to the model
    """
    import math

    tasktrain = task['train']
    imagepair = tasktrain[0]

    imageinput = np.array(imagepair['input'])
    imageoutput = np.array(imagepair['output'])

    #  check that these are the same size
    if not imageinput.shape == imageoutput.shape:
        #  print('Input and output not the same size so we know its not pattern')
        return 0

    sz0l = math.floor((32 - np.size(imageinput, 0)) / 2)  # padding for the left of dimension 0
    sz0r = math.ceil((32 - np.size(imageinput, 0)) / 2)  # padding for the right of dimension 0
    sz1l = math.floor((32 - np.size(imageinput, 1)) / 2)  # padding for the left of dimension 1
    sz1r = math.ceil((32 - np.size(imageinput, 1)) / 2)  # padding for the right of dimension 1
    ippad = np.pad(imageinput, ((sz0l, sz0r), (sz1l, sz1r)), constant_values=(0, 0))
    oppad = np.pad(imageoutput, ((sz0l, sz0r), (sz1l, sz1r)), constant_values=(0, 0))

    newsample = np.array([[ippad, oppad]])

    #  make channel the last dim
    newsample = np.moveaxis(newsample, 1, -1)

    return newsample


# def modelpresrecall(model, X, Y):
#     """gives metrics on the precision and recall of our model designed to label mosaic/symmetry tasks
#     """
#     from sklearn.metrics import classification_report
#
#     #  get precision & recall
#     y_pred = model.predict(X, batch_size=64, verbose=1)
#     y_pred_bool = np.argmax(y_pred, axis=1)
#
#     print(classification_report(Y, y_pred_bool))


def makepredictions(task, model):
    newsample = preparetaskformodel(task)

    if newsample is 0:
        return False
    else:
        prediction = float(model.predict(newsample))
        return prediction > 0.5


def checktranssymmetry(image, repunit):
    """Once a translational repeating unit has been created, this checks whether the repeating unit can be used
    to describe the whole image
    image   --      output image
    repunit --      repeating unit
    return  --      boolean whether repunit creates full pattern or not
    """
    # raster-scan in any possible increments for repeating unit
    for rasterrow in range(1, np.size(repunit, axis=0)):
        for rastercol in range(1, np.size(repunit, axis=1)):
            newrepunit = image[0:rasterrow, 0:rastercol]

            if checktranssymmetry(image, newrepunit):
                #  found it!
                foundsol = 1
                return foundsol, newrepunit

    return 1


def findtranssymmetries(imageoutput):
    """There may be a repeating unit which is translated (raster scanned) across the image. This finds that symmetry
    task    --      full task
    return:
    testout --      output for the test pattern
    cache   --      parameters for how the task was solved
    """

    foundsol = 0

    # create a repeating pattern
    for reprow in range(2, np.size(imageoutput, axis=0) / 2):
        for repcol in range(2, np.size(imageoutput, axis=1) / 2):
            newrepunit = imageoutput[0:reprow, 0:repcol]

            if checktranssymmetry(imageoutput, newrepunit):
                #  found it!
                foundsol = 1
                return foundsol, newrepunit

    return foundsol, newrepunit


def findrotsymmetries(imageoutput):
        """Othe type of possible symmetry is rotational symmetry. This finds any rotational symmetry
    task    --      full task
    return:
    testout --      output for the test pattern
    cache   --      parameters for how the task was solved
    """


def findsymmetries(task):
    """Once a task has been ascertained as a pattern task, this is how to solve it
    task    --      full task
    return:
    testout --      output for the test pattern
    cache   --      parameters for how the task was solved
    """

    import numpy as np

    tasktrain = task['train']
    imagepair = tasktrain[0]

    imageinput = np.array(imagepair['input'])
    imageoutput = np.array(imagepair['output'])

    # find translational symmetries
    foundsol, newrepunit = findtranssymmetries(imageoutput)

    #  find rotational symmetries if no translational symmetries are present
    if foundsol is not 1:
        foundsol, newrepunit = findrotsymmetries(imageoutput)

    #  if a pattern has been found, see if it's the same for the others in the set


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from skimage import measure
from matplotlib import colors
from numpy.lib.stride_tricks import as_strided


def in_out_diff(t_in, t_out):
    x_in, y_in = t_in.shape
    x_out, y_out = t_out.shape
    diff = np.zeros((max(x_in, x_out), max(y_in, y_out)))
    diff[:x_in, :y_in] -= t_in
    diff[:x_out, :y_out] += t_out
    return diff


def check_symmetric(a):
    try:
        sym = 1
        if np.array_equal(a, a.T):
            sym *= 2  # Check main diagonal symmetric (top left to bottom right)
        if np.array_equal(a, np.flip(a).T):
            sym *= 3  # Check antidiagonal symmetric (top right to bottom left)
        if np.array_equal(a, np.flipud(a)):
            sym *= 5  # Check horizontal symmetric of array
        if np.array_equal(a, np.fliplr(a)):
            sym *= 7  # Check vertical symmetric of array
        return sym
    except:
        return 0


def bbox(a):
    try:
        r = np.any(a, axis=1)
        c = np.any(a, axis=0)
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    except:
        return 0,a.shape[0],0,a.shape[1]


def cmask(t_in):
    cmin = 999
    cm = 0
    for c in range(10):
        t = t_in.copy().astype('int8')
        t[t==c],t[t>0],t[t<0]=-1,0,1
        b = bbox(t)
        a = (b[1]-b[0])*(b[3]-b[2])
        s = (t[b[0]:b[1],b[2]:b[3]]).sum()
        if a>2 and a<cmin and s==a:
            cmin=a
            cm=c
    return cm


def mask_rect(a):
    r,c = a.shape
    m = a.copy().astype('uint8')
    for i in range(r-1):
        for j in range(c-1):
            if m[i,j]==m[i+1,j]==m[i,j+1]==m[i+1,j+1]>=1:m[i,j]=2
            if m[i,j]==m[i+1,j]==1 and m[i,j-1]==2:m[i,j]=2
            if m[i,j]==m[i,j+1]==1 and m[i-1,j]==2:m[i,j]=2
            if m[i,j]==1 and m[i-1,j]==m[i,j-1]==2:m[i,j]=2
    m[m==1]=0
    return (m==2)


def crop_min(t_in):
    try:
        b = np.bincount(t_in.flatten(),minlength=10)
        c = int(np.where(b==np.min(b[np.nonzero(b)]))[0])
        coords = np.argwhere(t_in==c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return t_in[x_min:x_max+1, y_min:y_max+1]
    except:
        return t_in


def call_pred_train(t_in, t_out, pred_func):
    import inspect

    feat = {}
    feat['s_out'] = t_out.shape
    if t_out.shape==t_in.shape:
        diff = in_out_diff(t_in,t_out)
        feat['diff'] = diff
        feat['cm'] = t_in[diff!=0].max()
    else:
        feat['diff'] = (t_in.shape[0]-t_out.shape[0],t_in.shape[1]-t_out.shape[1])
        feat['cm'] = cmask(t_in)
    feat['sym'] = check_symmetric(t_out)
    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]])
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    feat['sizeok'] = len(t_out)==len(t_pred)
    t_pred = np.resize(t_pred,t_out.shape)
    acc = (t_pred==t_out).sum()/t_out.size
    return t_pred, feat, acc


def call_pred_test(t_in, pred_func, feat):
    import inspect

    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]])
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    return t_pred


# from: https://www.kaggle.com/nagiss/manual-coding-for-the-first-10-tasks
def get_data(task_filename):
    import json
    with open(task_filename, 'r') as f:
        task = json.load(f)
    return task

# from: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
color2num = {c: n for n, c in enumerate(num2color)}


def plot_one(ax, input_matrix, title_text):
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title_text)


def check_p(task, pred_func):
    import matplotlib.pyplot as plt

    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fnum = 0
    t_acc = 0
    t_pred_test_list = []
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred, feat, acc = call_pred_train(t_in, t_out, pred_func)
        plot_one(axs[0,fnum],t_in,f'train-{i} input')
        plot_one(axs[1,fnum],t_out,f'train-{i} output')
        plot_one(axs[2,fnum],t_pred,f'train-{i} pred')
        t_acc+=acc
        fnum += 1
    for i, t in enumerate(task["test"]):
        # removed t_out as there should be no output in the test
        t_in= np.array(t["input"]).astype('uint8')
        t_pred_test = call_pred_test(t_in, pred_func, feat)
        plot_one(axs[0,fnum],t_in,f'test-{i} input')
        #plot_one(axs[1,fnum],t_out,f'test-{i} output')
        plot_one(axs[2,fnum],t_pred_test,f'test-{i} pred')
        t_pred = np.resize(t_pred,t_in.shape) # assume same shape. used to be: t_pred = np.resize(t_pred,t_out.shape)
        # if len(t_out)==1:
        #     acc = int(t_pred==t_out)
        # else:
        #     acc = (t_pred==t_out).sum()/t_out.size
        # t_acc += acc
        # fnum += 1
        t_pred_test_list.append(t_pred_test)
    # plt.show()
    return t_acc/fnum, t_pred_test_list


def get_tile(img ,mask):
    try:
        m,n = img.shape
        a = img.copy().astype('int8')
        a[mask] = -1
        r=c=0
        for x in range(n):
            if np.count_nonzero(a[0:m,x]<0):continue
            for r in range(2,m):
                if 2*r<m and (a[0:r,x]==a[r:2*r,x]).all():break
            if r<m:break
            else: r=0
        for y in range(m):
            if np.count_nonzero(a[y,0:n]<0):continue
            for c in range(2,n):
                if 2*c<n and (a[y,0:c]==a[y,c:2*c]).all():break
            if c<n:break
            else: c=0
        if c>0:
            for x in range(n-c):
                if np.count_nonzero(a[:,x]<0)==0:
                    a[:,x+c]=a[:,x]
                elif np.count_nonzero(a[:,x+c]<0)==0:
                    a[:,x]=a[:,x+c]
        if r>0:
            for y in range(m-r):
                if np.count_nonzero(a[y,:]<0)==0:
                    a[y+r,:]=a[y,:]
                elif np.count_nonzero(a[y+r,:]<0)==0:
                    a[y,:]=a[y+r,:]
        return a[r:2*r,c:2*c]
    except:
        return a[0:1,0:1]


def patch_image(t_in,s_out,cm=0):
    try:
        t = t_in.copy()
        ty,tx=t.shape
        if cm>0:
            m = mask_rect(t==cm)
        else:
            m = (t==cm)
        tile = get_tile(t ,m)
        if tile.size>2 and s_out==t.shape:
            rt = np.tile(tile,(1+ty//tile.shape[0],1+tx//tile.shape[1]))[0:ty,0:tx]
            if (rt[~m]==t[~m]).all():
                return rt
        for i in range(6):
            m = (t==cm)
            t -= cm
            if tx==ty:
                a = np.maximum(t,t.T)
                if (a[~m]==t[~m]).all():t=a.copy()
                a = np.maximum(t,np.flip(t).T)
                if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.flipud(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.fliplr(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            t += cm
            m = (t==cm)
            lms = measure.label(m.astype('uint8'))
            for l in range(1,lms.max()+1):
                lm = np.argwhere(lms==l)
                lm = np.argwhere(lms==l)
                x_min = max(0,lm[:,1].min()-1)
                x_max = min(lm[:,1].max()+2,t.shape[0])
                y_min = max(0,lm[:,0].min()-1)
                y_max = min(lm[:,0].max()+2,t.shape[1])
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                if i==1:
                    sy//=2
                    y_max=y_min+sx
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                allst = as_strided(t, shape=(ty,tx,sy,sx),strides=2*t.strides)
                allst = allst.reshape(-1,sy,sx)
                allst = np.array([a for a in allst if np.count_nonzero(a==cm)==0])
                gm = (gap!=cm)
                for a in allst:
                    if sx==sy:
                        fpd = a.T
                        fad = np.flip(a).T
                        if i==1:gm[sy-1,0]=gm[0,sx-1]=False
                        if (fpd[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fpd)
                            t[y_min:y_max,x_min:x_max] = gap
                            break
                        if i==1:gm[0,0]=gm[sy-1,sx-1]=False
                        if (fad[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fad)
                            t[y_min:y_max,x_min:x_max] = gap
                            break
                    fud = np.flipud(a)
                    flr = np.fliplr(a)
                    if i==1:gm[sy-1,0]=gm[0,sx-1]=gm[0,0]=gm[sy-1,sx-1]=False
                    if (a[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,a)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (fud[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,fud)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (flr[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,flr)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
        if s_out==t.shape:
            return t
        else:
            m = (t_in==cm)
            return np.resize(t[m],crop_min(m).shape)
    except:
        return np.resize(t_in, s_out)
    
 #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MISC FROM KAGGLE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

def Defensive_Copy(A):
    n = len(A)
    k = len(A[0])
    L = np.zeros((n, k), dtype=int)
    for i in range(n):
        for j in range(k):
            L[i, j] = 0 + A[i][j]
    return L.tolist()


def Create(task, task_id=0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output


def Recolor(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    N = len(Input)

    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1

    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1, 8):
            for Q2 in range(1, 8):
                if Q1 + Q2 == t:
                    Pairs.append((Q1, Q2))

    for Q1, Q2 in Pairs:
        for v in range(4):

            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}

            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v == 2:
                            p1 = i % Q1
                        else:
                            p1 = (n - 1 - i) % Q1
                        if v == 0 or v == 3:
                            p2 = j % Q2
                        else:
                            p2 = (k - 1 - j) % Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)
                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            if possible:

                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v == 2:
                                p1 = i % Q1
                            else:
                                p1 = (n - 1 - i) % Q1
                            if v == 0 or v == 3:
                                p2 = j % Q2
                            else:
                                p2 = (k - 1 - j) % Q2

                            color1 = x[i][j]
                            rule = (p1, p2, color1)

                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False
                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v

    if Best_Dict == -1:
        return -1  # meaning that we didn't find a rule that works for the traning cases

    # Otherwise there is a rule: so let's use it:
    n = len(Test_Picture)
    k = len(Test_Picture[0])

    answer = np.zeros((n, k), dtype=int)

    for i in range(n):
        for j in range(k):
            if Best_v == 0 or Best_v == 2:
                p1 = i % Best_Q1
            else:
                p1 = (n - 1 - i) % Best_Q1
            if Best_v == 0 or Best_v == 3:
                p2 = j % Best_Q2
            else:
                p2 = (k - 1 - j) % Best_Q2

            color1 = Test_Picture[i][j]
            rule = (p1, p2, color1)
            if (p1, p2, color1) in Best_Dict:
                answer[i][j] = 0 + Best_Dict[rule]
            else:
                answer[i][j] = 0 + color1

    return answer.tolist()


def toplevel1(task):
    Function = Recolor

    basic_task = Create(task, 0)
    a = Function(basic_task)

    return a        


# In[ ]:


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ARCRULES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from copy import deepcopy
import datetime
import time
import xgboost as xgb

# ~~~~~~~~~~~~~~~~~~~~ functions used by entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ functions used by entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ functions used by entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ functions used by entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def colouriner(imgpair):
    """if the input & output is the same other than objects being a different colour, then yes
    imgpair is a SingleImagePair object
    """
    # look for multi-colour objs against a background
    if imgpair.backgroundcol is not None:
        inputbw = (imgpair.fullpredimg == imgpair.backgroundcol) * 1
        outputbw = (imgpair.fulloutputimg == imgpair.backgroundcol) * 1

        if np.array_equal(inputbw, outputbw) & (not np.array_equal(imgpair.fullpredimg, imgpair.fulloutputimg)):
            # this says that the objects are the same... but:
            # is one object a multicolour object?
            # is the background incorrect?

            # easy win:
            if len(imgpair.predoutputsamecolobjs) != len(imgpair.outputsamecolourobjs):
                return 2

    # look for same-colour objs
    if len(imgpair.predoutputsamecolobjs) == len(imgpair.outputsamecolourobjs):
        identitycount = 0
        for objin in imgpair.outputsamecolourobjs:
            for objout in imgpair.predoutputsamecolobjs:
                if (np.array_equal(objin.elementarr, objout.elementarr)) & (objin.positionabsx == objout.positionabsx)                         & (objin.positionabsy == objout.positionabsy):
                    identitycount += 1

        if identitycount == len(imgpair.outputsamecolourobjs):
            # all objects in input can be found in output in the same location
            return 1
        else:
            return 2
    else:
        return 0


def zoominer(imgpair, returnsz=1):
    """if the output is a zoomed in version of the input, return 1
    imgpair is a SingleImagePair object
    """
    inimg = imgpair.fullinputimg
    outimg = imgpair.fulloutputimg

    # raster-scan an "outimg" sized image across inimg, see if any of the segments are equal to outimg
    if (inimg.shape[0] > outimg.shape[0]) & (inimg.shape[1] > outimg.shape[1]):
        for ii in range(inimg.shape[0] - outimg.shape[0]):
            for jj in range(inimg.shape[1] - outimg.shape[1]):
                rows = np.repeat(list(range(ii, ii + outimg.shape[0])), outimg.shape[1])
                cols = list(range(jj, jj + outimg.shape[1])) * outimg.shape[0]
                inzoom = inimg[rows, cols]
                if np.array_equal(inzoom, outimg.flatten()):
                    if returnsz == 1:
                        return 1
                    else:
                        return 1, ii, jj

    return 0


def zoomonobject(imgpair):
    for obj in imgpair.inputsamecolourobjs:
        if np.array_equal(obj.elementarr * obj.colour, imgpair.fulloutputimg):
            return 1

    return 0


def objremer(imgpair):
    """finds if an object(s) from the input is removed from the output
    """
    if len(imgpair.inputsamecolourobjs) <= len(imgpair.outputsamecolourobjs):
        return 0

    if len(imgpair.inputsamecolourobjs) > 100:
        return 0

    outobjcount = [0] * len(imgpair.outputsamecolourobjs)
    for outobjno, outobj in enumerate(imgpair.outputsamecolourobjs):
        for inobj in imgpair.inputsamecolourobjs:
            if np.array_equal(inobj.elementarr, outobj.elementarr):
                outobjcount[outobjno] = 1

    if len(outobjcount) == outobjcount.count(1):
        return 1
    else:
        return 0


def listofuniqueshapes(objlist):
    objswiththatshape = []  # list of [list for each shape: which contains obj nos associated with that shape]
    listofshapes = []  # list of np arrays, each of which is a unique shape
    for objno, eachobj in enumerate(objlist):
        newshape = 1
        for shapeno, shape in enumerate(listofshapes):  # list of all unique symbols for this task
            if np.array_equal(eachobj.elementarr, shape):
                newshape = 0
                objswiththatshape[shapeno] = objswiththatshape[shapeno] + [objno]
                break

        if newshape:  # made it to the end of listofshapes, not in there, add it
            listofshapes = listofshapes + [eachobj.elementarr]
            objswiththatshape = objswiththatshape + [[objno]]

    return {'shapes': listofshapes, 'objswithshape': objswiththatshape}


def symbolser(fulltask):
    """look for re-occuring symbols across all the tasks
    """
    # question is: what constitutes a symbol, what constitutes just a normal object?
    # very basic: let's turn everything into a symbol
    listofsymbols = []
    allsymbolnumbers = []  # unique number for each symbol found
    for traintest in [fulltask.trainsglimgprs, fulltask.testinputimg]:
        for eachtest in traintest:
            symbolsinimg = []
            for eachobj in eachtest.predoutputsamecolobjs:
                stillvalid = 1
                counter = 0
                for symbolno, symbol in enumerate(listofsymbols):  # list of all symbols for this task
                    if np.array_equal(eachobj.elementarr, symbol):
                        stillvalid = 0
                        symbolsinimg = symbolsinimg + [symbolno]
                        break

                    counter += 1

                if stillvalid:  # made it to the end of listofsymbols, not in there, add it
                    listofsymbols = listofsymbols + [eachobj.elementarr]

                    # add this symbol number to symbols in this image:
                    allsymbolnumbers = allsymbolnumbers + [counter]
                    symbolsinimg = symbolsinimg + [counter]

            # when we've gone through each object in the set, we should add these to every obj
            for eachobj in eachtest.predoutputsamecolobjs:
                for symbolno, eachsymbol in enumerate(symbolsinimg):
                    setattr(eachobj, 'symbol' + str(symbolno), eachsymbol)

    return fulltask


def booleannoter(fulltask):
    boolnottask = 0
    for imgpair in fulltask.trainsglimgprs:
        toomanyobjs = len(imgpair.predoutputsamecolobjs) > 40
        if not toomanyobjs:
            for objpred in imgpair.predoutputsamecolobjs:
                newelemarr = 1 - objpred.elementarr
                objnottoosmall = newelemarr.sum() > 1
                if objnottoosmall:
                    # remove cols/rows of all zeros
                    validcols = newelemarr.sum(axis=0) > 0  # logical index col
                    validrows = newelemarr.sum(axis=1) > 0  # logical index row
                    firstcol = min(np.where(validcols)[0])
                    firstrow = min(np.where(validrows)[0])
                    lastcol = max(np.where(validcols)[0])
                    lastrow = max(np.where(validrows)[0])
                    validcols[firstcol:lastcol] = True
                    validrows[firstrow:lastrow] = True
                    newelemarr = newelemarr[np.ix_(validrows, validcols)]

                    for objout in imgpair.outputsamecolourobjs:

                        # if any one obj is the same, should apply a not to at least 1 of these objects
                        if np.array_equal(newelemarr, objout.elementarr):
                            boolnottask = 1

    return boolnottask


def booleanlogicer(imgpair):
    """need only 2 input shapes and they both need to be the same size as the output img
    """
    stillvalid = 1
    if len(imgpair.predoutputsamecolobjs) != 2:
        stillvalid = 0

    for obj in imgpair.predoutputsamecolobjs:
        if (obj.elementarr.shape[0] != imgpair.fulloutputimg.shape[0]) or                 (obj.elementarr.shape[1] != imgpair.fulloutputimg.shape[1]):
            stillvalid = 0

    return stillvalid


def movingobjectser(imgpair):
    """Prelim requirements for moving objs around. check that all objs can be mapped from in to out and they're
    in different locations at the in than they are at the out
    """
    intooutobjs, warnings = linkinobjtooutobj(imgpair)
    if len(warnings) > 0:  # we need 1:1 mapping from in:out, as in test, won't know what to map
        return 0

    inobjs = intooutobjs['inobjs']

    if (len(inobjs) == len(imgpair.predoutputsamecolobjs)) and (len(inobjs) == len(imgpair.outputsamecolourobjs)):
        return 1
    else:
        return 0


# ~~~~~~~~~~~~~~~~~~~~~~~~ rules to apply if passes entry requirements ~~~~~~~~~~~~~~~~~~~~
def accbyinputpixtooutput(fulltask):
    """returns a pixel-wise comparison of matching pixels, comparing input to output
    """
    allaccs = []

    for imgpair in fulltask.trainsglimgprs:
        if imgpair.fullpredimg.shape == imgpair.fulloutputimg.shape:
            # can compare on a pixel-wise basis
            samepix = np.equal(imgpair.fullpredimg, imgpair.fulloutputimg) * 1
            unique, count = np.unique(samepix, return_counts=True)
            if (1 in unique) and (0 in unique):
                oneidx = np.where(unique == 1)[0][0]
                acc = count[oneidx] / (sum(count))
            elif (1 in unique) and (0 not in unique):
                acc = 1
            else:
                acc = 0
        else:
            # should compare on an object-wise basis
            linkedobjs, warning = linkinobjtooutobj(imgpair)  # outobjs in dict are outobjs linked
            # if there are same amount of linked objs to output objs: all objects are accounted for. max acc = 0.9
            acc = 0.9 - np.tanh(abs(len(linkedobjs['outobjs']) - len(imgpair.outputsamecolourobjs)) +
                                abs(len(linkedobjs['outobjs']) - len(imgpair.predoutputsamecolobjs)))

        allaccs = allaccs + [acc]

    acc = sum(allaccs) / len(allaccs)

    return acc


def subtaskvalidation(fulltaskold, fulltasknew, taskname):
    fulltasknew = placeobjsintofullimg(fulltasknew)

    accnew = accbyinputpixtooutput(fulltasknew)
    accold = accbyinputpixtooutput(fulltaskold)
    print('{} - acc before: {}, acc after: {}'.format(taskname, accold, accnew))

    if accnew == 1:
        fulltasknew = placeobjsintofullimg(fulltasknew)
        fulltasknew.testpred = []
        for testimgpairs in fulltasknew.testinputimg:
            fulltasknew.testpred = fulltasknew.testpred + [testimgpairs.fullpredimg]

    if accnew > accold:
        return accnew, fulltasknew
    else:
        return accold, fulltaskold


def findintattrs(fulltask):
    """returns the attributes of the fulltask which are int, as these can be unpacked easily as
    features for an NN
    """
    attrvals = vars(fulltask.trainsglimgprs[0].predoutputsamecolobjs[0])
    samecolobjattrs = list(vars(fulltask.trainsglimgprs[0].predoutputsamecolobjs[0]).keys())
    isint = []

    for attr in samecolobjattrs:
        if isinstance(attrvals[attr], int):
            isint.append(attr)

    return isint


def resultsfrommodel(xtest, ylabels, model):
    """passes xtest through a model to return a set of y predictions
    """
    model, modeltype = model

    if modeltype == 'nn':
        predictions = model.predict(xtest)
        results = np.dot(np.round(predictions), ylabels)  # put them back into an array which holds the colours
    elif modeltype == 'otomap':
        otomap, col = model
        results = []
        for ii in range(xtest.shape[0]):
            xval = str(int(xtest[ii, col]))  # sometimes leave .0 on so need to do int
            results = results + [otomap[xval]]
    elif modeltype == 'xgb':
        predictions = model.predict(xtest)
        predictions2, _ = ylabelstooh(predictions)
        results = np.dot(np.round(predictions2), ylabels)  # put them back into an array which holds the colours

    return results


def createxfeatures(imgpair, objno, isint, maxobjno):
    """creates all the x features for one sample. Many samples in an xsamples, which calls this fun.
    """
    features = [0] * maxobjno * len(isint)
    attrcount = 0

    objlist = imgpair.predoutputsamecolobjs

    # features for this obj go at the beginning
    for attr in isint:
        features[attrcount] = getattr(objlist[objno], attr)
        attrcount += 1

    # make a list of numbers for all objects other than main obj
    otherobjs = list(range(len(objlist)))
    otherobjs.remove(objno)

    # loop through list, each loop looping through attrs like above
    for otherobj in otherobjs:
        for attr in isint:
            features[attrcount] = getattr(objlist[otherobj], attr)
            attrcount += 1

    return np.array(features).reshape(1, maxobjno * len(isint))


def findmaxobjno(fulltask):
    """finds the max no of objs in both train & test so that the correct size for xtrain is given
    """
    maxobjno = 0

    # make the maxobjno size the size of imgpair with the most objs
    for traintest in [fulltask.trainsglimgprs, fulltask.testinputimg]:
        for imgpair in traintest:
            objlist = imgpair.predoutputsamecolobjs

            if len(objlist) > maxobjno:
                maxobjno = len(objlist)

    return maxobjno


def createxsamples(imgpairlist, isint, maxobjno):
    """creates an array which creates x samples to train the NN on.
    Input (imgpairlist): testinputimg or trainsglimgprs
    samples is np array. each row is 1 sample. Each col is 1 feature.
    """
    for imgpair in imgpairlist:
        objlist = imgpair.predoutputsamecolobjs

        for objno in range(len(objlist)):
            if 'samples' in locals():
                samples = np.vstack((samples, createxfeatures(imgpair, objno, isint, maxobjno)))
            else:
                samples = createxfeatures(imgpair, objno, isint, maxobjno)

    # remove cols which don't have any variation
    # colsallsame = np.where(np.all(samples == samples[0, :], axis=0))[0]

    # if colsallsame.size != 0:
    #     samples = np.delete(samples, colsallsame, 1)

    return samples


def makesimplemodel(xtrain, ytrain, ylabels, isint, xtest):
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.metrics import accuracy_score

    # try a one-to-one mapping first
    revertedy = np.dot(ytrain, ylabels)

    symbolcols = [i for i, name in enumerate(isint) if not name.find('symbol')]
    acc = 0

    for allcols in [symbolcols, range(len(isint))]:
        for cols in allcols:

            otomap = {}
            otomaprev = {}  # make sure there ins't just a list of unique xvals mapping to same val of y's
            for rows in range(xtrain.shape[0]):
                xval = str(xtrain[rows, cols])
                yval = str(revertedy[rows])
                if xval in otomap:
                    if (otomap[xval] != revertedy[rows]) or (otomaprev[yval] != xtrain[rows, cols]):
                        break
                else:
                    otomap[xval] = revertedy[rows]
                    otomaprev[yval] = xtrain[rows, cols]

                if rows == xtrain.shape[0] - 1:
                    acc = 1
                    break

            if acc:
                break
        if acc:
            break

    if acc == 1:  # first if acc == 1: need to now check that this oto mapping is still valid for test
        for rows in range(xtest.shape[0]):
            if not str(xtest[rows, cols]) in otomap.keys():
                # oto map will err
                acc = 0

    if acc == 1:
        model = otomap, cols  # the correct mapping & the col from x train that it was from
        modeltype = 'otomap'
        print(isint[cols])
    else:
        yclasses = np.size(ytrain, axis=1)
        usexgboost = 1

        if usexgboost:
            model = xgb.XGBClassifier(max_depth=3, eta=1, reg_lambda=5)
            model.fit(xtrain, revertedy)
            prediction = model.predict(xtrain)
            prediction2, _ = ylabelstooh(prediction)
            ypred = np.dot(np.round(prediction2), ylabels)
            acc = accuracy_score(revertedy, ypred)
            modeltype = 'xgb'
        else:
            # Neural network
            model = Sequential()
            model.add(Dense(16, input_dim=np.size(xtrain, axis=1), activation='relu'))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(yclasses, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(xtrain, ytrain, epochs=800, verbose=0)

            # evaluate the keras model
            _, acc = model.evaluate(xtrain, ytrain)

            modeltype = 'nn'

    return (model, modeltype), acc


def ylabelstooh(ylabel):
    from sklearn.preprocessing import OneHotEncoder

    # turn into a ohe
    ohe = OneHotEncoder()
    ylabel = np.array(ylabel)
    ylabelunique = np.unique(ylabel)
    ylabel = ylabel.reshape(-1, 1)
    y = ohe.fit_transform(ylabel).toarray()

    return y, ylabelunique


def createcolysamples(imgpairlist):
    ylabel = []
    for imgpair in imgpairlist:
        inobjs = imgpair.predoutputsamecolobjs
        outobjs = imgpair.outputsamecolourobjs

        for inobj in inobjs:
            xpos = inobj.positionabsx
            ypos = inobj.positionabsy

            # the same object in output might not be in the same index so will need to find it
            for outobj in outobjs:
                if (outobj.positionabsx == xpos) & (outobj.positionabsy == ypos):
                    break

            ylabel = ylabel + [outobj.colour]

    y, ylabelunique = ylabelstooh(ylabel)

    return y, ylabelunique


def makecolourpredictions(fulltask, model, isint, ylabels, maxobjno):
    altfulltask = deepcopy(fulltask)

    for traintest in [altfulltask.trainsglimgprs, altfulltask.testinputimg]:
        for testno, eachtest in enumerate(traintest):
            # find predictions from model
            xtest = createxsamples([eachtest], isint, maxobjno)
            # print(xtest.shape)

            results = resultsfrommodel(xtest, ylabels, model)

            # put the predictions into a final image
            for objno in range(len(eachtest.predoutputsamecolobjs)):
                eachtest.predoutputsamecolobjs[objno].colour = results[objno]

    return altfulltask


def placeobjsintosingleimgpair(imgpair):
    # make a blank canvas
    outputimg = np.zeros([imgpair.fullinputimg.shape[0], imgpair.fullinputimg.shape[1]])

    for objno, obj in enumerate(imgpair.inputsamecolourobjs):
        rowsalt = np.repeat(list(range(obj.positionabsx, obj.positionabsx + obj.height)), obj.width)
        colsalt = list(range(obj.positionabsy, obj.positionabsy + obj.width)) * obj.height
        vals = obj.elementarr.flatten() * obj.colour

        outputimg[rowsalt, colsalt] = vals

    return outputimg


def placeobjsintofullimg(fulltask):
    """takes objects from predoutputsamecolobjs and places them back into a fullpredimg image.
    We do this after every sub-task so that we can see if we've completed the task.
    """
    fulltask.testpred = []

    for traintest in [fulltask.trainsglimgprs, fulltask.testinputimg]:
        for testno, eachtest in enumerate(traintest):
            outputimg = np.copy(eachtest.predoutputcanvas)
            # need another incase obj locations are out of bounds for new canvas (need to move objs around for new
            # canvas, still, but still wanna keep previous changes)
            outputimgorig = np.zeros([eachtest.fullpredimg.shape[0], eachtest.fullpredimg.shape[1]])
            returnorigimgsize = 0

            for objno, obj in enumerate(eachtest.predoutputsamecolobjs):
                # colsalt = list(np.repeat(list(range(obj.positionabsx, obj.positionabsx + obj.height)), obj.width))
                # rowsalt = list(range(obj.positionabsy, obj.positionabsy + obj.width)) * obj.height
                colsalt = list(range(obj.positionabsx, obj.positionabsx + obj.width)) * obj.height
                rowsalt = list(np.repeat(list(range(obj.positionabsy, obj.positionabsy + obj.height)), obj.width))

                vals = list(obj.elementarr.flatten() * obj.colour)

                # remove values which are 0, so the background val is maintained  rowsalt
                indices = [i for i, x in enumerate(vals) if x == obj.colour]
                rowsalt = [d for (i, d) in enumerate(rowsalt) if i in indices]
                colsalt = [d for (i, d) in enumerate(colsalt) if i in indices]
                vals = [d for (i, d) in enumerate(vals) if i in indices]

                outputimgorig[rowsalt, colsalt] = vals
                try:
                    outputimg[rowsalt, colsalt] = vals
                except IndexError:
                    # got an err therefore we should return the other canvas
                    returnorigimgsize = 1

            if returnorigimgsize:
                eachtest.fullpredimg = outputimgorig
            else:
                eachtest.fullpredimg = outputimg

    return fulltask


def colourchange(fulltaskin):
    print('colourchange accessed')
    fulltask = deepcopy(fulltaskin)

    # unroll all the features & samples & put into array
    isint = findintattrs(fulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(fulltask)
    xtrain = createxsamples(fulltask.trainsglimgprs, isint, maxobjno)  # np array with samples=rows, features=cols
    # print(xtrain.shape)
    ytrain, ylabels = createcolysamples(fulltask.trainsglimgprs)  # makes y_set. one-hot np.
    # print(ytrain.shape)

    # special case: only 1 y output type
    ytrain2 = ytrain.tolist()
    if ytrain2.count(ytrain2[0]) == len(ytrain2):  # all 1's & will err out
        for traintest in [fulltask.trainsglimgprs, fulltask.testinputimg]:
            for imgno, imgpair in enumerate(traintest):
                for objno, obj in enumerate(imgpair.predoutputsamecolobjs):
                    obj.colour = fulltask.trainsglimgprs[0].outputsamecolourobjs[0].colour
        fulltask = placeobjsintofullimg(fulltask)
        acc, fulltaskfinal = subtaskvalidation(fulltaskin, fulltask, 'colourchange')
        return acc, fulltaskfinal

    # make & train a model on prorperties from train set
    xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)
    model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

    # use trained NN to predict colours for test set(s) + change the colours
    fulltask = makecolourpredictions(fulltask, model, isint, ylabels, maxobjno)

    # find acc of new iteration
    acc, fulltaskfinal = subtaskvalidation(fulltaskin, fulltask, 'colourchange')

    return acc, fulltaskfinal


def multicolourchange(fulltask):
    """this could be so complicated & could come in so many guises, I wouldn't know where to start with a
    hard-coded solutions. I'm just going to throw a NN at it.
    """
    print('multicolourchange accessed')
    acc = 0
    return acc, fulltask


def createyzoomsamples(imgpairlist):
    """Outputs a list of booleans. 1 if the object in question was what was zoomed in on. 0 if not
    """
    ytrain = []

    for imgpair in imgpairlist:
        for obj in imgpair.predoutputsamecolobjs:
            if np.array_equal(obj.elementarr * obj.colour, imgpair.fulloutputimg):
                ytrain = ytrain + [1]
            else:
                ytrain = ytrain + [0]

    y, ylabelunique = ylabelstooh(ytrain)

    return y


def makezoomobjpredictions(fulltask, model, isint, maxobjno):
    fulltask.testpred = []

    for testno, eachtest in enumerate(fulltask.testinputimg):
        # find predictions from model
        xtest = createxsamples([eachtest], isint, maxobjno)
        ylabels = [0, 1]

        results = resultsfrommodel(xtest, ylabels, model)

        objno = np.where(results == 1)[0][0]

        predobj = fulltask.testinputimg[testno].predoutputsamecolobjs[objno].elementarr *                   fulltask.testinputimg[testno].predoutputsamecolobjs[objno].colour

        fulltask.testpred = fulltask.testpred + [predobj]

    return fulltask


def zoomspecialrulecheck(fulltask):
    objsinimg = []
    for imgpairno in range(len(fulltask.trainsglimgprs)):
        objsinimg = objsinimg + [len(fulltask.trainsglimgprs[imgpairno].predoutputsamecolobjs)]

    if (len(objsinimg) == objsinimg.count(objsinimg[0])) & (objsinimg[0] == 1):
        return 1
    else:
        return 0


def zoomobjrules(fulltask):
    print('zoomobjrules accessed')

    # special, easy rule if there's only one obj to choose from:
    if zoomspecialrulecheck(fulltask):
        fulltask.testpred = []
        for istrain, traintest in enumerate([fulltask.trainsglimgprs, fulltask.testinputimg]):
            for testno, eachtest in enumerate(traintest):
                colour = eachtest.predoutputsamecolobjs[0].colour
                objimg = eachtest.predoutputsamecolobjs[0].elementarr
                eachtest.fullpredimg = objimg * colour
                if not istrain:
                    fulltask.testpred = fulltask.testpred + [objimg * colour]

        return 1, fulltask

    isint = findintattrs(fulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(fulltask)
    xtrain = createxsamples(fulltask.trainsglimgprs, isint, maxobjno)  # np array with samples=rows, features=cols
    ytrain = createyzoomsamples(fulltask.trainsglimgprs)

    # train NN on prorperties from train set
    ylabels = [0, 1]  # boolean of "was this zoomed in on"
    xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)
    model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

    # use trained NN to predict colours for test set(s)
    fulltask = makezoomobjpredictions(fulltask, model, isint, maxobjno)

    return acc, fulltask


def zoomnoobjrules(fulltask):
    acc = 0
    return acc, fulltask


def zoomrules(fulltask):
    """looks for rules determining what section to zoom in on, in the input, and why
    """
    print('zoomrules accessed')
    if checkforallimages(fulltask, zoomonobject):
        acc, fulltask = zoomobjrules(fulltask)  # zoomed on a specific object we have in input
    else:
        acc, fulltask = zoomnoobjrules(fulltask)  # zoomed on a specific area in input but not exactly around an obj

    return acc, fulltask


def createyobjremsamples(imgpairlist):
    """outputs a list of booleans. 1 for if the input img exists in the output. 0 if not
    """
    ytrain = []

    for imgpair in imgpairlist:
        for inobj in imgpair.predoutputsamecolobjs:
            objexists = 0
            for outobj in imgpair.outputsamecolourobjs:
                if np.array_equal(inobj.elementarr, outobj.elementarr):
                    objexists = 1

            ytrain = ytrain + [objexists]

    y, ylabelunique = ylabelstooh(ytrain)

    return y


def makeobjrempredictions(fulltask, model, isint, maxobjno):
    """predict which objects are to be removed in the train & test input image(s) & remove them, then check
    the prediction with the train images
    """
    # make a copy of the fulltask - gonna make some deletions
    newfulltask = deepcopy(fulltask)

    # remove the objects from inputsamecolobjs in train
    for traintest in [newfulltask.trainsglimgprs, newfulltask.testinputimg]:
        for testno, eachtest in enumerate(traintest):
            xtrain = createxsamples([eachtest], isint, maxobjno)  # np arr with samples=rows, features=cols
            # print('xtrain no {} has {} rows and {} cols'.format(testno, xtrain.shape[0], xtrain.shape[1]))
            ylabels = [0, 1]

            results = resultsfrommodel(xtrain, ylabels, model)

            # this is the first obj manipulation task
            if len(eachtest.predoutputsamecolobjs) == 0:
                eachtest.predoutputsamecolobjs = deepcopy(eachtest.fullinputimg)

            objs = eachtest.predoutputsamecolobjs

            noofobjs = len(objs)
            for objno in range(noofobjs - 1, -1, -1):  # go backwards as if we del, all idxs will shift down one
                if results[objno] == 1:  # del this
                    del (objs[objno])

    # let's see if that's been positive
    acc, finalfulltask = subtaskvalidation(fulltask, newfulltask, 'objrem')

    return acc, finalfulltask


def objremrules(fulltask):
    """Looks for rules determining what objects are removed from the input and why
    """
    print('object remove rules accessed')
    isint = findintattrs(fulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(fulltask)
    xtrain = createxsamples(fulltask.trainsglimgprs, isint, maxobjno)  # np array with samples=rows, features=cols
    ytrain = createyobjremsamples(fulltask.trainsglimgprs)

    # train NN on prorperties from train set
    ylabels = [1, 0]  # objects we want to del
    xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)
    model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

    # use trained NN to predict colours for test set(s)
    acc, fulltask = makeobjrempredictions(fulltask, model, isint, maxobjno)

    return acc, fulltask


def linkinobjtooutobj(imgpair, maptype='oto'):
    """for each in object, see it exists as an output. If so, return it's array list number. Return
    the output object's x position and y position"""
    warning = ''

    inshapesall = listofuniqueshapes(imgpair.predoutputsamecolobjs)
    outshapesall = listofuniqueshapes(imgpair.outputsamecolourobjs)
    inshapes = inshapesall['shapes']
    outshapes = outshapesall['shapes']
    inobjswshapes = inshapesall['objswithshape']
    outobjswshapes = outshapesall['objswithshape']
    if len(inobjswshapes) != len(outobjswshapes):
        warning = 'different shapes in as out /n'

    inobj = []
    outobj = []

    for inshapeno, eachinshape in enumerate(inshapes):
        for outshapeno, eachoutshape in enumerate(outshapes):
            if np.array_equal(eachinshape, eachoutshape):  # got a matching pair, make sure they're okay
                if len(inobjswshapes[inshapeno]) == len(outobjswshapes[outshapeno]):  # same no of ins to outs:
                    for objno in range(len(inobjswshapes[inshapeno])):
                        if maptype == 'oto':  # one to one
                            # just assign x in to x out so we get a 1:1 mapping
                            inobj = inobj + [inobjswshapes[inshapeno][objno]]
                            outobj = outobj + [outobjswshapes[outshapeno][objno]]

                        if maptype == 'otm':  # one to many
                            None  # for now

                else:
                    warning = warning + 'different number of ins to outs /n'

    if len(inobj) != len(imgpair.predoutputsamecolobjs):
        warning = warning + 'not all in shapes accounted for: need to remove some first'

    return {'inobjs': inobj, 'outobjs': outobj}, warning


def createymovesamples(imgpair, axis):
    positionlist = []
    for obj in imgpair.outputsamecolourobjs:
        positionlist = positionlist + [getattr(obj, 'positionabs' + axis)]

    return positionlist


def creatingmovingobjsxset(imgpairlist, isint, maxobjno, traintestno, xyaxis):
    for imgpairno, imgpair in enumerate(imgpairlist):
        intooutobjs, warnings = linkinobjtooutobj(imgpair)
        if len(warnings) > 0:  # we need 1:1 mapping from in:out, as in test, won't know what to map
            return 0, 0

        inobjs = intooutobjs['inobjs']
        outobjs = intooutobjs['outobjs']

        xtrainraw = createxsamples([imgpair], isint, maxobjno)  # np array with samples=rows, features=cols

        if imgpairno == 0:
            xtrain = xtrainraw[inobjs[0], :]
            for objno in inobjs[1:]:
                xtrain = np.vstack([xtrain, xtrainraw[objno, :]])

            if traintestno == 0:
                ytrainraw = createymovesamples(imgpair, xyaxis)
                ytrainraw2 = [ytrainraw[outobjs[0]]]
                for objno in outobjs[1:]:
                    ytrainraw2 = ytrainraw2 + [ytrainraw[objno]]
            else:
                ytrainraw2 = 0
        else:
            for objno in inobjs:
                xtrain = np.vstack([xtrain, xtrainraw[objno, :]])

            if traintestno == 0:
                ytrainraw = createymovesamples(imgpair, xyaxis)
                for objno in outobjs:
                    ytrainraw2 = ytrainraw2 + [ytrainraw[objno]]

    return xtrain, ytrainraw2


def movingobjects(fulltask):
    """looks to determine rules for where to move each object if they need moving
    """
    print('movingobjects accessed')
    newfulltask = deepcopy(fulltask)

    isint = findintattrs(newfulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(newfulltask)

    for xyaxis in ['x', 'y']:
        xtrain, ytrainraw2 = creatingmovingobjsxset(newfulltask.trainsglimgprs, isint, maxobjno, 0, xyaxis)
        xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)

        # train
        ytrain, ylabels = ylabelstooh(ytrainraw2)
        model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

        for traintestno, traintest in enumerate([newfulltask.trainsglimgprs, newfulltask.testinputimg]):
            for imgpairno, imgpair in enumerate(traintest):
                # now make predictions with the model
                xset = createxsamples([imgpair], isint, maxobjno)
                results = resultsfrommodel(xset, ylabels, model)

                # assign the new val
                for objno in range(len(imgpair.predoutputsamecolobjs)):
                    setattr(imgpair.predoutputsamecolobjs[objno], 'positionabs' + xyaxis, int(results[objno]))

    acc, finalfulltask = subtaskvalidation(fulltask, newfulltask, 'moveobjs')

    return acc, finalfulltask


def booleannot(fulltask):
    """applies a boolean not to each object in turn. If accuracy goes up, keep the not
    """
    print('booleannot accessed')
    newfulltask = deepcopy(fulltask)

    # make the y's
    boolnotobjs = []
    for imgpair in newfulltask.trainsglimgprs:
        for obj in imgpair.predoutputsamecolobjs:
            newelemarr = 1 - obj.elementarr

            # check this elemarr against output img
            y1 = obj.positionabsy
            y2 = obj.positionabsy + obj.elementarr.shape[0]
            x1 = obj.positionabsx
            x2 = obj.positionabsx + obj.elementarr.shape[1]
            outputimg = (imgpair.fulloutputimg[y1:y2, x1:x2] != imgpair.backgroundcol) * 1

            # add to list saying if it is or ins't a not
            boolnotobjs = boolnotobjs + [np.array_equal(newelemarr, outputimg)]

    isint = findintattrs(newfulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(newfulltask)

    xtrain = createxsamples(newfulltask.trainsglimgprs, isint, maxobjno)  # np array with samples=rows, features=cols
    ytrain, ylabels = ylabelstooh(boolnotobjs)

    xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)
    model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

    for traintest in [newfulltask.trainsglimgprs, newfulltask.testinputimg]:
        for imgpair in traintest:
            xtrain = createxsamples([imgpair], isint, maxobjno)
            results = resultsfrommodel(xtrain, ylabels, model)
            for objno, obj in enumerate(imgpair.predoutputsamecolobjs):
                if results[objno]:
                    obj.elementarr = 1 - obj.elementarr

    # turn this into a new class as we might have gained/lost objects
    newfulltask = placeobjsintofullimg(newfulltask)
    newfulltask = FullTaskFromClass(newfulltask)

    acc, finalfulltask = subtaskvalidation(fulltask, newfulltask, 'booleannot')

    return acc, finalfulltask


def booltests(newfulltask, test):
    for traintest in [newfulltask.trainsglimgprs, newfulltask.testinputimg]:
        for imgpair in traintest:
            if test == 0:  # logical and
                taskname = 'logical and'
                logicalarr = np.logical_and(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr) * 1
                imgpair.predoutputsamecolobjs = [SameColourObject(logicalarr, 1)]
                imgpair.fullpredimg = logicalarr
            elif test == 1:  # logical or
                taskname = 'logical or'
                logicalarr = np.logical_or(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr) * 1
                imgpair.predoutputsamecolobjs = [SameColourObject(logicalarr, 1)]
                imgpair.fullpredimg = logicalarr
            elif test == 2:  # logical nand
                taskname = 'logical nand'
                logicalarr = np.logical_not(np.logical_or(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr)) * 1
                imgpair.predoutputsamecolobjs = [SameColourObject(logicalarr, 1)]
                imgpair.fullpredimg = logicalarr
            elif test == 3:  # logical nor
                taskname = 'logical nor'
                logicalarr = np.logical_not(np.logical_or(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr)) * 1
                imgpair.predoutputsamecolobjs = [SameColourObject(logicalarr, 1)]
                imgpair.fullpredimg = logicalarr
            elif test == 4:  # logical xor
                taskname = 'logical xor'
                logicalarr = np.logical_xor(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr) * 1
                imgpair.predoutputsamecolobjs = [SameColourObject(logicalarr, 1)]
                imgpair.fullpredimg = logicalarr
            elif test == 5:  # logical xnor
                taskname = 'logical xnor'
                logicalarr = np.logical_not(np.logical_xor(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr)) * 1
                imgpair.predoutputsamecolobjs = [SameColourObject(logicalarr, 1)]
                imgpair.fullpredimg = logicalarr

    # elif test == 6:
    #     for traintest in [newfulltask.trainsglimgprs, newfulltask.testinputimg]:
    #         for imgpair in traintest:
    return newfulltask, taskname


def booleanlogic(fulltask):
    print('boolean logic accessed')
    for imgpair in fulltask.trainsglimgprs:
        imgpair.fullpredimg = imgpair.predoutputsamecolobjs[0].elementarr
    accold = accbyinputpixtooutput(fulltask)
    accbest = accold
    for test in range(5):
        newfulltask = deepcopy(fulltask)
        newfulltask, taskname = booltests(newfulltask, test)
        accnew, fulltasknew = subtaskvalidation(fulltask, newfulltask, taskname)
        if accnew > accbest:
            accbest = accnew
            toptest = test

    if accbest > accold:
        newfulltask = deepcopy(fulltask)
        newfulltask, taskname = booltests(newfulltask, toptest)

        # to re-order the class
        accnew, newfulltask = subtaskvalidation(fulltask, newfulltask, taskname)
        return accbest, newfulltask
    else:
        # no luck, return the old, unchanged fulltask
        return 0, fulltask


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ looping through entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~
def checkforallimages(fulltask, function):
    truthlist = []
    for ii in range(len(fulltask.trainsglimgprs)):
        truthlist.append(function(fulltask.trainsglimgprs[ii]))

    if truthlist.count(truthlist[0]) == len(truthlist):  # all are the same
        return truthlist[0]
    else:
        return 0


def timeout_handler(signum, frame):
    raise TimeoutException


class TimeoutException(Exception):
    pass


def findnextrule(fulltask, subtaskdonelist, symbols=True):
    """This loops through all rule entry requirements and looks for a rule which satisfies requirements. If requirements
    are met, it then looks for rules to define that type of behaviour. e.g. see if objects just need to be coloured in
    (entry requirements). If so: what determines the rule of colouring in?
    fulltask is a FullTask: if it has been input as an argument to second or greater depth calls for findnextrule, this
    may be adjusted from original fulltask.
    subtaskdonelist is a list of subtasks done, in case we keep recognising a task as needing that transform done on it,
    so we don't end up in an endless loop
    """
    startprocess = datetime.datetime.now()
    if (0 not in subtaskdonelist) & symbols:
        fulltask = symbolser(fulltask)
        subtaskdonelist = subtaskdonelist + [0]

    if (checkforallimages(fulltask, colouriner) == 1) & (1 not in subtaskdonelist):
        # do colour stuff
        acc, fulltask = colourchange(fulltask)
        subtaskdonelist = subtaskdonelist + [1]
    elif (checkforallimages(fulltask, colouriner) == 2) & (2 not in subtaskdonelist):
        # do multicolour stuff
        acc, fulltask = multicolourchange(fulltask)
        subtaskdonelist = subtaskdonelist + [2]
    elif (checkforallimages(fulltask, zoominer) == 1) & (3 not in subtaskdonelist):
        acc, fulltask = zoomrules(fulltask)
        subtaskdonelist = subtaskdonelist + [3]
    elif (checkforallimages(fulltask, objremer) == 1) & (4 not in subtaskdonelist):
        acc, fulltask = objremrules(fulltask)
        subtaskdonelist = subtaskdonelist + [4]
    elif (booleannoter(fulltask) == 1) & (5 not in subtaskdonelist):
        acc, fulltask = booleannot(fulltask)
        subtaskdonelist = subtaskdonelist + [5]
    elif (checkforallimages(fulltask, booleanlogicer) == 1) & (6 not in subtaskdonelist):
        acc, fulltask = booleanlogic(fulltask)
        subtaskdonelist = subtaskdonelist + [6]
    elif (checkforallimages(fulltask, movingobjectser) == 1) & (7 not in subtaskdonelist):
        acc, fulltask = movingobjects(fulltask)
        subtaskdonelist = subtaskdonelist + [7]
    else:
        # no more rules to apply
        acc = 0
        return acc, fulltask

    endprocess = datetime.datetime.now()
    print('Time spent on this process was: {}'.format(endprocess - startprocess))

    if acc == 1:
        for testno, onetestpred in enumerate(fulltask.testpred):
            if isinstance(onetestpred, list):
                fulltask.testpred[testno] = [int(x) for x in onetestpred]
            else:  # assume numpy array
                fulltask.testpred[testno] = onetestpred.astype(int)
                fulltask.testpred[testno] = fulltask.testpred[testno].tolist()

        return acc, fulltask
    else:
        # go again to see if we can find the next step
        acc, fulltask = findnextrule(fulltask, subtaskdonelist, symbols)
        return acc, fulltask


# In[ ]:


############################# PLOTTING FUNCTIONS ###############################
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_one(ax, task, i, traintest, input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    input_matrix = task[traintest][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(traintest + ' ' + input_or_output)


def plot_one_class(ax, task, i, traintest, inoutpred):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    if traintest == 'train':
        if inoutpred == 'input':
            input_matrix = task.trainsglimgprs[i].fullinputimg
        elif inoutpred == 'output':
            input_matrix = task.trainsglimgprs[i].fulloutputimg
        else:
            if task.trainsglimgprs[i].fullpredimg is None:
                return

            input_matrix = task.trainsglimgprs[i].fullpredimg
    else:
        if inoutpred == 'input':
            input_matrix = task.testinputimg[i].fullinputimg
        elif inoutpred == 'pred':
            if task.testinputimg[i].fullpredimg is None:
                return

            input_matrix = task.testinputimg[i].fullpredimg

        else:
            return

    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(traintest + ' ' + inoutpred)


def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    if isinstance(task, FullTask) or isinstance(task, FullTaskFromClass):
        plotarcclass(task)
        return

    num_train = len(task['train'])
    fig, axs = plt.subplots(2, num_train, figsize=(3 * num_train, 3 * 2))
    for i in range(num_train):
        plot_one(axs[0, i], task, i, 'train', 'input')
        plot_one(axs[1, i], task, i, 'train', 'output')
    plt.tight_layout()
    plt.show()

    num_test = len(task['test'])
    fig, axs = plt.subplots(2, num_test, figsize=(3 * num_test, 3 * 2))
    if num_test == 1:
        plot_one(axs[0], task, 0, 'test', 'input')
        plot_one(axs[1], task, 0, 'test', 'output')
    else:
        for i in range(num_test):
            plot_one(axs[0, i], task, i, 'test', 'input')
            plot_one(axs[1, i], task, i, 'test', 'output')
    plt.tight_layout()
    plt.show()


def print_numpy_arr(task):
    num_train = len(task['train'])
    for i in range(num_train):
        print()


def plotarcclass(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    num_train = len(task.trainsglimgprs)
    fig, axs = plt.subplots(3, num_train, figsize=(3 * num_train, 3 * 2))
    for i in range(num_train):
        plot_one_class(axs[0, i], task, i, 'train', 'input')
        plot_one_class(axs[1, i], task, i, 'train', 'output')
        plot_one_class(axs[2, i], task, i, 'train', 'pred')
    plt.tight_layout()
    plt.show()

    num_test = len(task.testinputimg)
    fig, axs = plt.subplots(3, num_test, figsize=(3 * num_test, 3 * 2))
    if num_test == 1:
        plot_one_class(axs[0], task, 0, 'test', 'input')
        plot_one_class(axs[1], task, 0, 'test', 'output')
        plot_one_class(axs[2], task, 0, 'test', 'pred')
    else:
        for i in range(num_test):
            plot_one_class(axs[0, i], task, i, 'test', 'input')
            plot_one_class(axs[1, i], task, i, 'test', 'output')
            plot_one_class(axs[2, i], task, i, 'test', 'pred')
    plt.tight_layout()
    plt.show()


# In[ ]:


def singlesolution(task, patmodel):
    acc = 0
    patterntask = makepredictions(task, patmodel)

    if patterntask:
        # t_pred_test_list is a list containing numpy array, 1 element for each input in test
        acc, t_pred_test_list = check_p(task, patch_image)

        if acc != 1:
            None  # make a pattern class here and see if we can get results from that
        else:
            sol = 'pattern'

    if acc != 1:
        subtaskdonelist = []
        taskclass = FullTask(task)
        try:
            acc, fulltask = findnextrule(taskclass, subtaskdonelist)
            t_pred_test_list = fulltask.testpred

            if acc == 1:
                sol = 'arcrule'
        except Exception as inst:
            print('errored with symbols')
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)
            acc = 0

    if acc != 1:
        subtaskdonelist = []
        taskclass = FullTask(task)
        try:
            taskclass = FullTask(task)  # reload as symbols will stay
            acc, fulltask = findnextrule(taskclass, subtaskdonelist, symbols=False)
            t_pred_test_list = testpred

            if acc == 1:
                sol = 'arcrule'
            else:
                fulltask.testpred = []
                for testno, onetestpred in enumerate(fulltask.testinputimg):
                    if isinstance(onetestpred.fullpredimg, list):
                        fulltask.testpred = fulltask.testpred + [int(x) for x in onetestpred.fullpredimg]
                    else:  # assume numpy array
                        ontestpredlist = onetestpred.fullpredimg.astype(int)
                        fulltask.testpred = fulltask.testpred + [ontestpredlist.tolist()]
                        
        except Exception as inst:
            print('errored without symbols')
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)
            acc = 0

    if acc != 1:
        try:
            a = toplevel1(task)
        except Exception as inst:
            print('misckaggle errored')
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)
            a = -1
            acc = 0

        if a != -1:
            print('misc kaggle: {} to 1'.format(acc))
            acc = 1
            t_pred_test_list = [a]
            fulltask.testinputimg[0].fullpredimg = np.array(t_pred_test_list[0])
            sol = 'misckaggle'
        else:
            sol = 'arcrule'

    return fulltask, t_pred_test_list, sol

def testatask(task):
    # grab the pattern model
    f = open('/kaggle/working/patternmodel.pckl', 'rb')
    model = cloudpickle.load(f)
    
    try:
        fulltask, t_pred_test_list, sol = singlesolution(task, model)
    except Exception as inst:
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args
        print(inst)
        acc = 0

    # if acc != 1:
    #     taskclass = arcclasses.FullTask(task)
    #     acc, taskclass = arcrules.findnextrule(taskclass, [], symbols=False)  # trying again without symbols

    try:
        if (not sol == 'pattern') or (not sol == 'misckaggle'):
            plot_task(fulltask)
    except Exception as inst:
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args
        print(inst)
        acc = 0
        
        print('Something went wrong with plotting... sorry!')
        
# save the pattern detect NN model:
if os.path.isfile('/kaggle/input/patmodel/patternmodel.pckl'):
    f = open('/kaggle/input/patmodel/patternmodel.pckl', 'rb')
    model = cloudpickle.load(f)

    fl = open('/kaggle/working/patternmodel.pckl', 'wb')
    cloudpickle.dump(model, fl)


# # The fun interactive dropdown:

# In[ ]:


from ipywidgets import Layout, Button, VBox, Label, Box, Output

output_task = Output()

alltasks, tasknames = startup(dataset='test')

buttonstyle = ['danger'] * len(tasknames)
for taskcomplete in [1, 12, 14, 17, 39, 40, 46, 48, 86, 89]:
    buttonstyle[taskcomplete] = 'success'

def btn_eventhandler(obj):
    with output_task:
        print(obj.description)
        output_task.clear_output()
        tnindex = tasknames.index(obj.description)
        print(tnindex)
        testatask(alltasks[tnindex])

item_layout = Layout(height='50px', min_width='490px')
items = [Button(layout=item_layout, description=str(tasknames[taskno]), button_style=buttonstyle[taskno]) for taskno in range(len(tasknames))]
for eachbutton in items:
    eachbutton.on_click(btn_eventhandler)
    
box_layout = Layout(overflow_y='auto',
                    border='3px solid black',
                    width='500px',
                    height='500px',
                    flex_flow='column',
                    display='block')
carousel = Box(children=items, layout=box_layout)

VBox([Label('Tasks to select:'), carousel])


# In[ ]:


display(output_task)

