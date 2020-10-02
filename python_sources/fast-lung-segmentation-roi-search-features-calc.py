#!/usr/bin/env python
# coding: utf-8

# #Some snippets for better lung/faster segmentation, OpenCV ROI based search & nodule feature extraction#
# 
# -first try with python, use with caution! :) improvements or tips on python best practices & performance hugely appreciated)
# 
# -ROI search forked/inspired from https://www.kaggle.com/twanmal/data-science-bowl-2017/nodules-detection-with-opencv by AntoineMal
# 
# Hope it can help someone,
# Good luck!
# 
# \#Data4good
# 
# DevScope team
# [www.devscope.net][1]
# 
# 
#   [1]: http://www.devscope.net

# In[ ]:


#Check data
get_ipython().system('ls ../input/sample_images')


# In[ ]:


import matplotlib.pyplot as plt
import sys,os
import matplotlib.mlab as mlab
import scipy
from skimage import measure
import numpy as np 
import pandas as pd 
import argparse 
import cv2  
import dicom
import pdb
from tqdm import tqdm
import glob
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
import warnings

#TODO: Some warnings to fix
warnings.filterwarnings("ignore")


# In[ ]:


#Config/Settings

DATA_FOLDER="../input/sample_images/"
SHOW_PLOTS=False
SEGMENT_ONLY=False
CROP_SIZE=50
LOG_LEVEL=0

#Util functions
def log(message, level=0):
    if level>=LOG_LEVEL:
        print(message)

def error(message):
    log(message,9999)
        
def readDicom(file):
    return readDicomLocal(file)

def readDicomLocal(file):
    return dicom.read_file(DATA_FOLDER+"/"+file)

def show(image, level=0, title=""):
    if not SHOW_PLOTS:
            return
    if level>=LOG_LEVEL:
        plt.suptitle(title)
        plt.imshow(image)
        plt.show()


# In[ ]:


#Lung Segmentation 
#Rui Barbosa, v4

def dicomtoGray(dicom_pixel_array,bt,wt):    
    pixel_array = np.copy(dicom_pixel_array)
    image = np.zeros((pixel_array.shape[0], pixel_array.shape[1], 1), np.uint8)

    pa = np.copy(pixel_array)
    w = np.where((pa > bt) & (pa < wt))
    pixel_array[pa > 255] = 255
    pixel_array[pa < 0] = 0

    pixel_array[w] = np.multiply(pa[w],255 / (wt - bt) - 255 * bt / (wt - bt))

    pixel_array[pixel_array < 0] = 0
    pixel_array[pixel_array > 255] = 255

    image[:,:,0] = pixel_array
    return image

def sliceDicomPixelArray(dicom_file):
    #dicom_file = readDicom(caseSlice) ## original dicom File
    dicom_pixelarray = dicom_file.pixel_array

    if (dicom_file.RescaleSlope != 1):                    
        dicom_pixelarray = dicom_pixelarray * dicom_file.RescaleSlope
    if (dicom_file.RescaleIntercept != 1024):
        dicom_pixelarray = dicom_pixelarray + (dicom_file.RescaleIntercept + 1024)
    return dicom_pixelarray

def sliceRead(caseSlice, bt=0, wt=1400):
    dicom_pixelarray = sliceDicomPixelArray(caseSlice)
    image = dicomtoGray(dicom_pixelarray,bt,wt)
    return image



def ellipseAnglesNearPoints(ellipse, pts, Nb=50):
	# ra - major axis length
	# rb - minor axis length
	# ang - angle
	# x0,y0 - position of centre of ellipse
	# Nb - No.  of points that make an ellipse
	   		
	# parse cv2.fitEllipse data
	center, axis, angle = ellipse
	xpos,ypos = center
	radm,radn = axis

	# convert to angle to radians
	ang = angle * np.pi / 180.0

	# axis must be half width, half height
	radm = radm * 0.5
	radn = radn * 0.5

	# calculate all the ellipse points
	co,si = np.cos(ang),np.sin(ang)
	the = np.linspace(0.87,2 * np.pi,Nb)
	X = radm * np.cos(the) * co - si * radn * np.sin(the) + xpos
	ax = np.int0(X)
	Y = radm * np.cos(the) * si + co * radn * np.sin(the) + ypos
	ay = np.int0(Y)

	ptAngles = []
	for pt in enumerate(pts):
		ptx = pt[1][0]
		pty = pt[1][1]
		dist = np.sqrt((X - ptx) ** 2 + (Y - pty) ** 2)
		iMinDist = np.argmin(dist)
		ptAngles.append(np.int(np.round(the[iMinDist] / np.pi * 180.0)))

	return ptAngles

def ellipse2PointsStartEnd(ellipse, startMiddleEndPoints , Nb=180):
	# ra - major axis length
	# rb - minor axis length
	# ang - angle
	# x0,y0 - position of centre of ellipse
	# Nb - No.  of points that make an ellipse
	   		
	# parse cv2.fitEllipse data
	center, axis, angle = ellipse
	xpos,ypos = center
	radm,radn = axis

	#print(axis)

	# convert to angle to radians
	ang = angle * np.pi / 180.0

	# axis must be half width, half height
	radm = radm * 0.5
	radn = radn * 0.5

	# calculate all the ellipse points
	co,si = np.cos(ang),np.sin(ang)
	the = np.linspace(0,2 * np.pi,Nb - 1)
	X = radm * np.cos(the) * co - si * radn * np.sin(the) + xpos	
	Y = radm * np.cos(the) * si + co * radn * np.sin(the) + ypos	

	# calculate the start, middle and end points indexes
	# start angle index
	ptsx = startMiddleEndPoints[0][0]
	ptsy = startMiddleEndPoints[0][1]
	dist = np.sqrt((X - ptsx) ** 2 + (Y - ptsy) ** 2)
	istart = np.argmin(dist)

	#middle angle index
	ptmx = startMiddleEndPoints[1][0]
	ptmy = startMiddleEndPoints[1][1]
	dist = np.sqrt((X - ptmx) ** 2 + (Y - ptmy) ** 2)
	imiddle = np.argmin(dist)

	#end angle index
	ptex = startMiddleEndPoints[2][0]
	ptey = startMiddleEndPoints[2][1]
	dist = np.sqrt((X - ptex) ** 2 + (Y - ptey) ** 2)
	iend = np.argmin(dist)

	#if iend > istart:
	#	ax = np.int0(X[istart:iend + 1])
	#	ay = np.int0(Y[istart:iend + 1])
	#else:
	#	ax = np.int0(X[iend:])
	#	ay = np.int0(Y[iend:])
	#	ax=np.append(ax,np.int0(X[0:istart + 1]))
	#	ay=np.append(ay,np.int0(Y[0:istart + 1]))
			
	#1 5 10
	if istart < imiddle and imiddle < iend:
		ax = np.int0(X[istart:iend + 1])
		ay = np.int0(Y[istart:iend + 1])
	else:
		#10 5 1
		if istart > imiddle and iend < imiddle:
			ax = np.int0(X[iend:istart + 1])
			ay = np.int0(Y[iend:istart + 1])
		else:
			if istart > imiddle and iend > imiddle:
				if istart > iend:	
					if (istart - imiddle) + (len(X) - istart) > len(X) - istart + iend : 
						ax = np.int0(X[istart:])
						ay = np.int0(Y[istart:])
						ax = np.append(ax,np.int0(X[:iend]))
						ay = np.append(ay,np.int0(Y[:iend]))
					else:	
						ax = np.int0(X[imiddle:istart])
						ay = np.int0(Y[imiddle:istart])
						ax = np.append(ax,np.int0(X[iend:]))
						ay = np.append(ay,np.int0(Y[iend:]))
				else:
					ax = np.int0(X[iend:])
					ay = np.int0(Y[iend:])
					ax = np.append(ax,np.int0(X[0:istart]))
					ay = np.append(ay,np.int0(Y[0:istart]))
			else:
				if istart < iend and iend < imiddle:
					ax = np.int0(X[imiddle:])
					ay = np.int0(Y[imiddle:])
					ax = np.append(ax,np.int0(X[0:istart]))
					ay = np.append(ay,np.int0(Y[0:istart]))
				else:
					ax = np.int0(X[istart:])
					ay = np.int0(Y[istart:])
					ax = np.append(ax,np.int0(X[0:iend]))
					ay = np.append(ay,np.int0(Y[0:iend]))
					
	#if len(ax) > (Nb / 2 + 10):
	#	print("problems : %d" % len(ax))
	#	print("")

	# generate the contour
	pts = np.array((ax,ay)).T

	return pts


def ellipse2Points(ellipse,Nb=50):
	# ra - major axis length
	# rb - minor axis length
	# ang - angle
	# x0,y0 - position of centre of ellipse
	# Nb - No.  of points that make an ellipse
	   		
	# parse cv2.fitEllipse data
	center, axis, angle = ellipse
	xpos,ypos = center
	radm,radn = axis

	# convert to angle to radians
	ang = angle * np.pi / 180.0

	# axis must be half width, half height
	radm = radm * 0.5
	radn = radn * 0.5

	# calculate all the ellipse points
	co,si = np.cos(ang),np.sin(ang)
	the = np.linspace(0,2 * np.pi,Nb)
	X = radm * np.cos(the) * co - si * radn * np.sin(the) + xpos
	ax = np.int0(X)
	Y = radm * np.cos(the) * si + co * radn * np.sin(the) + ypos
	ay = np.int0(Y)

	while ax[0] == ax[-1] and ay[0] == ay[-1]:
		ax = ax[:-1]
		ay = ay[:-1]

	# generate the contour
	pts = np.array((ax,ay)).T

	return pts

def toInt(a):
	ai = np.int0(np.round(a))
	return ai

def retraceOuterContour(img, contour, centroid, isOnLeft):
	hull = cv2.convexHull(contour,returnPoints = True)
			
	ih = img.shape[0]
	iw = img.shape[1]
		
	# calculate extreme contour points
	indexPtLeft = hull[:, :, 0].argmin()
	ptLeft = tuple(hull[indexPtLeft][0])
	indexPtRight = hull[:, :, 0].argmax()
	ptRight = tuple(hull[indexPtRight][0])
	indexPtTop = hull[:, :, 1].argmin()
	ptTop = tuple(hull[indexPtTop][0])
	indexPtBottom = hull[:, :, 1].argmax()
	ptBottom = tuple(hull[indexPtBottom][0])

	imcolor = np.zeros((ih,iw,3),np.uint8)
	
	if isOnLeft :
		# render left part segment
		# calculate left countour segment
		leftContour = hull[range(indexPtBottom,indexPtTop)]
		if len(leftContour) >= 5:
			e = cv2.fitEllipse(leftContour)
			epts = ellipse2PointsStartEnd(e,[ptTop,ptLeft, ptBottom])
			cv2.polylines(img,[epts],False,(255,255,255),2)
			#cv2.polylines(imcolor,[epts],False,(0,0,255),2)
	else:			
		# render right part segment
		# calculate right countour segment
		rightContour = hull[sorted(np.r_[indexPtTop:len(hull) - 1,0:indexPtBottom])]			
		if len(rightContour) >= 5:
			e = cv2.fitEllipse(rightContour)
			epts = ellipse2PointsStartEnd(e,[ptTop,ptRight,ptBottom])
			cv2.polylines(img,[epts],False,(255,255,255),2)	
			#cv2.polylines(imcolor,[epts],False,(0,0,255),2)
	
	return epts

def rectContains(ra,rb):
	rax = ra[0]
	ray = ra[1]
	raw = ra[2]
	rah = ra[3]
	rbx = rb[0]
	rby = rb[1]
	rbw = rb[2]
	rbh = rb[3]
	return  rbx >= rax and rby >= ray and rbx + rbw <= rax + raw and rby + rbh <= ray + rah


def segmentLungsGetMask(dicom_pixel_array):
	ih = dicom_pixel_array.shape[0]
	iw = dicom_pixel_array.shape[1]

	# create a tube mask with a inner black border
	tubeMask = np.zeros((ih, iw, 1), np.uint8)
	tubeMask[dicom_pixel_array >= 0] = 255
	tubeMask[0:5,0:iw] = 0
	tubeMask[0:ih,0:5] = 0
	tubeMask[ih - 5:,0:iw] = 0
	tubeMask[0:ih:,iw - 5:] = 0		
		
	# now get a slice to extract contour
	imageSlice = dicomtoGray(dicom_pixel_array,600,3000)
	imageSlice[imageSlice > 1] = 255
	imageSlice[tubeMask == 0] = 255				
	#imageSlice = cv2.morphologyEx(imageSlice, cv2.MORPH_CLOSE,
	#np.ones((2,2),np.uint8))
				
	# find contours
	_, contours, hier = cv2.findContours(imageSlice,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
		
	# process contours and get the contour with max area at the same time
	maxia = (ih * iw) * 0.95
	maxContourArea = -1
	maxContourIndex = 0
	minContourValidArea = 500	
	
	ctinfo = []
	for i,c in enumerate(contours):
		
		cX = 0
		cY = 0

		# contour area
		ca = cv2.contourArea(c)
		# contour box
		cbox = cv2.boundingRect(c) # bx,by,bw,bh
		cbox = np.int0(cbox)

		# calculate extreme contour points of valid contours only
		extremePoints = []
		if ca > minContourValidArea : 					
			# determine contour centroid
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])			

		# save the ct info
		ctinfo.append((i,ca,(cX,cY),cbox))

		# determine the max contour index
		if ca > maxContourArea and ca < maxia:
			maxContourIndex = i
			maxContourArea = ca
	
	# center exclusion box
	cebw = 140
	cebh = 250
	cebox = [(iw - cebw) / 2, (ih - cebh) / 2, cebw,cebh] 

	# ctscan table exclusion
	maxcycenter = ih / 4 * 3.25

	# get valid contours
	validContoursInfo = [ctinfo[i] for i,ch in enumerate(hier[0]) if ch[3] == maxContourIndex and ctinfo[i][1] > minContourValidArea and not (ctinfo[i][1] < 2000 and rectContains(cebox,ctinfo[i][3])) and ctinfo[i][2][1] < maxcycenter]

	# create lungs mask
	lungsMask = np.zeros(imageSlice.shape, np.uint8)

	# determine retraceable items
	retraceBorderMinArea = 5000
	retraceBorderMaxWidth = 220
	#elementsToRetrace = validContoursInfo[np.where(x[1] >
	#minAreaToRetraceBorder)]

	#close imperfections/holes using detecting and filling patches
	for vci in validContoursInfo:
		# get contour
		ct = contours[vci[0]]
		
		# get contour area
		ca = vci[1]

		# centroid and relative position
		ctCentroid = vci[2]
		ctIsOnLeft = ctCentroid[0] < (iw * 0.5) # x center of image
		
		#get contour bounding box x,y
		cpx = vci[3][0]
		cpy = vci[3][1]

		#get contour bounding box w,h
		ctw = vci[3][2]
		cth = vci[3][3]			
		
		# shift contour points relative to blob, considering a 30px margin
		ct = ct - (cpx - 30,cpy - 30)

		# create an image for holding the blob mask for this contour
		maskc = np.zeros((cth + 60,ctw + 60,1), np.uint8)
		
		# draw contours and close gaps allong the contour
		cv2.drawContours(maskc, [ct], -1, (255,255,255), 1)
		maskc = cv2.morphologyEx(maskc, cv2.MORPH_CLOSE, np.ones((10,1),np.uint8))
		maskc = cv2.morphologyEx(maskc, cv2.MORPH_CLOSE, np.ones((1,10),np.uint8))
		
		#retrace outer contour

		#cv2.imshow('before tracer',maskc)
		retracePts = []
		if ctw <= retraceBorderMaxWidth and ca > retraceBorderMinArea:
			retracePts = retraceOuterContour(maskc, ct, ctCentroid, ctIsOnLeft)
		#cv2.imshow('after tracer',maskc)
		#cv2.waitKey(0)

		#fill blob holes
		maskcfl = maskc.copy()	
		maskff = np.zeros((cth + 62, ctw + 62), np.uint8)	# must be +2,+2
		cv2.floodFill(maskcfl, maskff, (0,0), 255)	# Floodfill from point (0, 0)
		maskc[maskcfl == 0] = 255

		# erase outer line of retracer
		if len(retracePts) > 0:
			cv2.polylines(maskc,[retracePts],False,(0,0,0),6)	

		#remove maskc border
		maskc = maskc[30:cth + 30,30:ctw + 30]
		# get lungs mask roi and add it with mask
		lungsMaskRoi = lungsMask[cpy:cpy + cth,cpx:cpx + ctw]			
		lungsMaskRoi[maskc > 0] = 255
		#set roi path back to lungsmask
		lungsMask[cpy:cpy + cth,cpx:cpx + ctw] = lungsMaskRoi

    #Final erode
	lungsMask=cv2.erode(lungsMask, None, iterations=3)
	return lungsMask


# In[ ]:


#From: https://github.com/jrosebr1/imutils/blob/master/imutils/contours.py
#imutils not loaded on kaggle

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


# In[ ]:


#Forked from https://www.kaggle.com/twanmal/data-science-bowl-2017/nodules-detection-with-opencv by AntoineMal
#Changed & tuned for better performance

#### a dicom monochrome file has pixel value between approx -2000 and +2000, opencv doesn't work with it#####
#### in a first step we transform those pixel values in (R,G,B)
### to have gray in RGB, simply give the same values for R,G, and B, 
####(0,0,0) will be black, (255,255,255) will be white,


## the threeshold to be automized with a proper quartile function of the pixel distribution
black_threeshold=0###pixel value below 0 will be black,
white_threeshold=1400###pixel value above 1400 will be white
wt=white_threeshold
bt=black_threeshold


def DicomtoRGB(dicom_pixel_array,bt,wt):
    
    pixel_array=np.copy(dicom_pixel_array)
    image = np.zeros((pixel_array.shape[0], pixel_array.shape[1], 3), np.uint8)

    pa=np.copy(pixel_array)
    w=np.where((pa>bt)  &( pa<wt))
    pixel_array[pa>255]=255
    pixel_array[pa<0]=0

    pixel_array[w]=np.multiply(pa[w],255/(wt-bt)-255*bt/(wt-bt))
        
    image[:,:,0]=pixel_array
    image[:,:,1]=pixel_array
    image[:,:,2]=pixel_array
    return image


def getLungSearchRegion(image,mask,caseSlice):
    im=image
    show(im,1,"original")
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts=cnts[1]
    cnts = sort_contours(cnts)[0]
    log(len(cnts))
    
    bigcnts=[]
    
    #filter small countours
    log(len(cnts))
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        
        #((cX, cY), radius)= cv2.minEnclosingCircle(c)
        area=cv2.contourArea(c)
        log("%d: %d"%(i,area))
        if (area>5000):
            bigcnts.append(c)
        
    cnts=  bigcnts
        
    
    if (len(cnts)==2):
        (x1, y1, w1, h1)=cv2.boundingRect(cnts[0])
        (x2, y2, w2, h2)=cv2.boundingRect(cnts[1])
        (x,y,radius)= (int(x2-w1*.1), int(y1+(h1/2)),int(w1/2))
        
        maskPosition=(len(cnts),(x1, y1, w1, h1), (x2, y2, w2, h2),(x,y,radius))
    if (len(cnts)==1):
        (x1, y1, w1, h1)=cv2.boundingRect(cnts[0])
      
        (x,y,radius)= (int(x1+w1/2), int(y1+(h1/2)),int(w1*.2))
        maskPosition=(len(cnts),(x1, y1, w1, h1), None,(x,y,radius))
    
    
    if (len(cnts)==0 or len(cnts)>2):
        error("warning cnts count<> 1,2: %s"%len(cnts))
        show(image,3)
        show(mask,3)
        return
    
    #Fit ellipse/check center & rotation
    
    if (len(cnts)==1):
        ellipse = cv2.fitEllipse(cnts[0])
    else:
        ellipse = cv2.fitEllipse(np.concatenate((cnts[0],cnts[1]), axis=0))
        
    (elx,ely),(elMA,elma),elangle=ellipse
    log(((elx,ely),(elMA,elma),elangle))
    #cv2.ellipse(mask,ellipse,60,1)
    #show(im,True,"mask ellipse")
    
    #TODO: Rotate image & mask, not working 100%, skip for now
    (h, w) = mask.shape[:2]
    M = cv2.getRotationMatrix2D((elx,ely),-(90-elangle), 1.0)
    #mask = cv2.warpAffine(mask, M, (w, h))
    #show(mask,1,"mask rotated")
    
    #im = cv2.warpAffine(im, M, (w, h))
    #show(im,1,"image rotated")
    
    
    log((x,y,radius))
    
    #holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30)))
    show(mask,1,"closing")
    
    
    masked_data = cv2.bitwise_and(im, im, mask=mask)
    show(im,True,"image")
    show(mask,True,"mask")    
    show(masked_data,True,"mask applied")
    
    return (im,mask,maskPosition,masked_data,(elx,ely))

def searchROIs(image,maskPosition,masked_data,caseSlice,zIndex,zIndexPerc):
    im=image
    
    #---------------------------------
    ###################################################################
    ## trying to find nodules as bright spot###########################
    ###################################################################


    ### in a first step, look at the first lung region, this should be improved to look at both lung regions

    lung_region = masked_data

    show(lung_region)

    
    w,h,bpp = np.shape(lung_region)
    sumRegion=np.sum(im)
    
    moyenne = sumRegion/(h*w*bpp)
    
    # load the image, convert it to grayscale, and blur it
    gray = cv2.cvtColor(lung_region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    show(blurred)
    thresh = cv2.threshold(blurred, moyenne + 20, 255, cv2.THRESH_BINARY)[1]## please try several value for moyenne+20  
 

    show(thresh)
    dilate = cv2.erode(thresh, None, iterations=3)## number of iterations to be automized

    show(dilate)
    thresh=dilate
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 50 & numPixels < 100:## to be automized
            mask = cv2.add(mask, labelMask)

    # find the contours in the mask, then sort them from left to
    # right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    cnts=cnts[1]
    if (len(cnts)==0):
        log("no countours found")
        return []
    
    cnts = sort_contours(cnts)[0]

    # loop over the contours
    a = np.matrix([])# liste of radius
    diff = np.array([0])
    df = pd.DataFrame({'cX':[0], 'cY': [0],'radius':[0]})
    j=0
    cropSize=50
    rois=[]
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image

        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        area = cv2.contourArea(c)
        log("candidate %s %s %s %s %s" % (cX, cY, radius,area,area/radius))
        if int(radius)<2 or int(area)<12 or area/radius<3:
            continue
        cropROI=np.copy(image[ int(cY-cropSize):int(cY+cropSize),int(cX-cropSize):int(cX+cropSize)])
        show(cropROI,1)
        rois.append((cY,cX,area,radius,zIndex,zIndexPerc,maskPosition))
        
        cv2.circle(image, (int(cX), int(cY)), int(radius),
            (255, 0, 0), 3)
        cv2.putText(image, "r{0:.1f},a{1:.1f}".format(radius,area), (x, y - 15),
            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        #cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        df.loc[j,'cX']=cX 
        df.loc[j,'cY'] = cY
        df.loc[j,'radius'] = radius
        df.loc[j,'area']=area
        j=j+1

  

    if (j>0):
        log ("Found %d: %s"%(j,caseSlice))
        show(dilate,2,"pre findings: %s"%j)
        log(df)
        show(image,2,"%s: %s"%(caseSlice,j))                               
        
    else:
        log("No findings")
    
    return rois


# In[ ]:


#Code for case/slice ROI workflow

#search case slice by Slice Number, ex: check with Radiant 
def segmentdAndSearchInstanceNumber(case,index=1):
    path=DATA_FOLDER+"/"+case
    
    slices = [{"file":os.path.basename(s), "dicom":dicom.read_file(s,stop_before_pixels=True)} for s in glob.glob(path+"/*.dcm")]    
    
    slices.sort(key=lambda x: int(x["dicom"].InstanceNumber))
    
    log("%d=%s/%s"%(index,case,slices[index-1]["file"]))
    
    path=case+"/"+slices[index]["file"]
    
    rois=segmentdAndSearchSlice(path)
    return rois



#search case slice
def segmentdAndSearchSlice(caseSlice,zIndex=-1,zIndexPerc=-1,roisDataFrame=None):
    try:
        log(caseSlice)
        dicom_file = readDicom(caseSlice) ## original dicom File
        dicom_pixelarray=dicom_file.pixel_array
        show(dicom_pixelarray)
       
        # get hu's pixel array, apply slope/offset
        dicom_pixel_array2 = sliceDicomPixelArray(dicom_file)
        
        # Nodule functions needs RGB/3 channels for now
        image=DicomtoRGB(dicom_pixel_array2,0,1400)   
        
        # get segmentation mask
        lungsMask = segmentLungsGetMask(dicom_pixel_array2)
        
        lungsMask=np.reshape(lungsMask,(lungsMask.shape[0],lungsMask.shape[1]))
        if (SEGMENT_ONLY):
            show(lungsMask,3,"lungsMask")
            return
        else:
            show(lungsMask,0,"lungsMask")
        
        lungsArea=len(lungsMask[lungsMask>0])/(lungsMask.shape[0]*lungsMask.shape[1])
        log("Lungs area %f" % lungsArea)
        if (lungsArea<.05):
            log("Skipping, small lung area")
            return

       
        #Get cut lungs regions
        lungSearchRegionResult=getLungSearchRegion(image,lungsMask,caseSlice)
        if (lungSearchRegionResult is None):
            return
        
        (image,lungsMask,maskPosition,masked_data,lungsCenter)=lungSearchRegionResult
        
        #Do ROI search
        foundRois=searchROIs(image,maskPosition,masked_data,caseSlice,zIndex,zIndexPerc)  
        i=0
        if (roisDataFrame is None):
            roisDataFrame=pd.DataFrame()
        
        #Confirm/Exclude rois, gather some stats
        for sliceRoi in foundRois:
            i=i+1
            (cY,cX,area,radius,zIndex,zIndexPerc,maskPosition)=sliceRoi
            
            (nLungs,maskPosLung1, maskPosLung2,(lungsx,lungsy,lungsRadius))=maskPosition
            (lung1x, lung1y, lung1w, lung1h)=maskPosLung1
            if (maskPosLung2!=None):
                (lung2x, lung2y, lung2w, lung2h)=maskPosLung1
            else:
                (lung2x, lung2y, lung2w, lung2h)=(0,0,0,0)
            
            cropROI=np.copy(masked_data[ cY-CROP_SIZE:cY+CROP_SIZE,cX-CROP_SIZE:cX+CROP_SIZE])
            cropMask=np.copy(lungsMask[ cY-CROP_SIZE:cY+CROP_SIZE,cX-CROP_SIZE:cX+CROP_SIZE])
            
            show(cropROI,1,"%s checking roi #%d"%(caseSlice,i))

            roiInfo=pd.DataFrame()
            
            #Confirm ROI, extract features
            includeRoi=addRoiFeatures(cropROI,sliceRoi,cropMask,lungsMask.copy(),cY,cX,lungsCenter,roiInfo)
            if (not includeRoi):
                continue
                
            show(masked_data,3,"Roi confirmed: %s #%d"%(caseSlice, i))
            
            #Add slice features/stats
            dfIndex=len(roisDataFrame)+1
            roisDataFrame.loc[dfIndex,"caseSlice"]=caseSlice  
    
            roisDataFrame.loc[dfIndex,"cX"]=cX
            roisDataFrame.loc[dfIndex,"cY"]=cY
            roisDataFrame.loc[dfIndex,"area"]=area
            roisDataFrame.loc[dfIndex,"radius"]=radius
            roisDataFrame.loc[dfIndex,"zIndex"]=zIndex
            roisDataFrame.loc[dfIndex,"zIndexPerc"]=zIndexPerc
            roisDataFrame.loc[dfIndex,"lung1x"]=lung1x
            roisDataFrame.loc[dfIndex,"lung1y"]=lung1y
            roisDataFrame.loc[dfIndex,"lung1w"]=lung1w
            roisDataFrame.loc[dfIndex,"lung1h"]=lung1h
            roisDataFrame.loc[dfIndex,"lung2x"]=lung2x
            roisDataFrame.loc[dfIndex,"lung2y"]=lung2y
            roisDataFrame.loc[dfIndex,"lung2w"]=lung2w
            roisDataFrame.loc[dfIndex,"lung2h"]=lung2h
            roisDataFrame.loc[dfIndex,"lungsx"]=lungsx
            roisDataFrame.loc[dfIndex,"lungsy"]=lungsy
            roisDataFrame.loc[dfIndex,"lungsRadius"]=lungsRadius
            
            #Add features from roi
            for col in roiInfo.columns.tolist():
                roisDataFrame.loc[dfIndex,col]=roiInfo.loc[1,col]
            
            log(roisDataFrame.iloc[0,:],1)
        
        return roisDataFrame
    except KeyboardInterrupt:
        raise
    except:
        raise
        error("Error on %s: %s"%(caseSlice, sys.exc_info()))
    
    return
        
def segmentdAndSearchCase(case,minPerc=.15,maxPerc=.55):
        
    path=DATA_FOLDER+"/"+case
    
    slices = [{"file":os.path.basename(s), "dicom":dicom.read_file(s,stop_before_pixels=True)} for s in glob.glob(path+"/*.dcm")]    
    
    slices.sort(key=lambda x: -int(x["dicom"].ImagePositionPatient[2]))
    slices=[s["file"] for s in slices]
    log("Dicoms for %s: %d" %(case,len(slices)))
    
    #Init data frame
    roisDf=pd.DataFrame()
    i=0
    for slice in tqdm(slices):
        i=i+1
        file=slice
        path=case+"/"+file
        try:
            zIndex=i
            zIndexPerc=i/len(slices)
            if (zIndexPerc<minPerc or zIndexPerc>maxPerc):
                log("%f not in perc range [%f,%f]"%(zIndexPerc,minPerc,maxPerc))
                continue
            
            segmentdAndSearchSlice(path,zIndex,zIndexPerc,roisDf)
           
        except KeyboardInterrupt:
            raise
        except:
            raise
            error("Error on %s: %s"%(path, sys.exc_info()))
   
       
    log("Rois for %s: %d"%(case,len(roisDf)),3)
    return roisDf


# In[ ]:


#Needs lots of cleaning & further tests
def addRoiFeatures(cropROI,sliceRoi,cropMask,lungsMask,roiY,roiX,lungsCenter,roisDataFrame=None):
    if (roisDataFrame is None):
        roisDataFrame=pd.DataFrame()
        
    dfIndex=len(roisDataFrame)+1
    (lungCenterX,lungCenterY)=lungsCenter
    
    #Show lungs center
    cv2.circle(lungsMask, (int(lungCenterX), int(lungCenterY)), int(5),
            (255, 0, 0), 3)
    
    #Force two lungs separation for distance
    lungsMask[:,lungCenterX-5:lungCenterX+5]=0
    
    (cY,cX,area,radius,zIndex,zIndexPerc,maskPosition)=sliceRoi
    
    img_gray = cv2.cvtColor(cropROI,cv2.COLOR_BGR2GRAY)
    original=img_gray.copy()
    
    show(img_gray,2,"gray")
    show(cropMask,1,"segmentRoiMask")
    
    img=img_gray
    original=img
   
    middle = img
    
    distxr=None
   
    #Distances to lung walls, to discard center false ROIs
    directions=[(-1,-1),(-1,0),(-1,1),(1,-1),(1,0),(1,1),(0,1),(0,-1)]
    
    wallDists=[]
    iDist=0
    maxX=roiX
    minX=roiX
    
    #TODO: For probably not the best/fastest way
    for (stepX,stepY) in directions:
        iDist=iDist+1
        yi=roiY
        xi=roiX
        dist=0
        for i in range(0,400):       
            yi=yi+stepY
            xi=xi+stepX
            if (yi>=lungsMask.shape[0] or yi<=0
               or xi>=lungsMask.shape[1] or xi<=0): 
                break;

            if lungsMask.item(yi,xi)==0:        
                dist=np.sqrt((xi-roiX)**2+(yi-roiY)**2)
                break;
                
            #Draw the walk
            lungsMask[yi,xi]=20
            
            #Store min and max X
            if (xi>maxX):
                    maxX=xi
            if (xi<minX):
                    minX=xi
                    
        wallDists.append(dist)
        roisDataFrame.loc[dfIndex,"dist_%s"%iDist]=dist
    
    log("dists1: %f dist2: %f"%(roisDataFrame.loc[dfIndex,"dist_2"],roisDataFrame.loc[dfIndex,"dist_5"]),1)
    roiLungCenterX=(maxX+minX)/2
    log("maxX: %f minX: %f avg:%f"%(maxX,minX,roiLungCenterX),1)
    log("roiX %d"%roiX)
    log(999)
    log("wall dists %s" % wallDists,0)
    wallDistsMax=np.max(wallDists)
    log("wall dists max %s" % wallDistsMax)
    
    #TODO
    if (wallDistsMax<40):
        log("wall dist max error")
        roisDataFrame.loc[dfIndex,:]=0
        return False
    
    #Exclude center ROIs, distance to inner border is greater than outer border (for left/right cases)
    log(roisDataFrame.loc[dfIndex])
    if (roisDataFrame.loc[dfIndex,"dist_2"]<=roisDataFrame.loc[dfIndex,"dist_5"]*1.1 and roiLungCenterX>lungCenterX ):
        log("skip more close to center (right)")
        roisDataFrame.loc[dfIndex,:]=0
        return False
    if (roisDataFrame.loc[dfIndex,"dist_2"]*1.1>=roisDataFrame.loc[dfIndex,"dist_5"] and  roiLungCenterX<lungCenterX):
        log("skip more close to center (left)")
        roisDataFrame.loc[dfIndex,:]=0
        return False
    
    #Detecting ROI contour
    
    kmeans = KMeans(n_clusters=2,random_state=42).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    
    #Allocate each x,y to one of the clusters (reshape, has to be one dimensional array)
    thresh_img=kmeans.predict(np.reshape(middle,[np.prod(middle.shape),1]))*10+10
    
    #Reshape again to needed shape
    thresh_img=np.reshape(thresh_img,middle.shape)
    

    thresh_img=thresh_img.astype(np.uint8)
   
    show(thresh_img,3,"kmeans")
    
    allRois=thresh_img.copy()

    #label connected components
    labels = measure.label(thresh_img, neighbors=8, background=0)
    mask = np.zeros(thresh_img.shape, dtype="uint8")

    # loop over the unique components
    i=0
    for label in np.unique(labels):
        i=i+1
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh_img.shape, dtype="uint8")
        
        # Color different labels
        labelMask[labels == label] = i*10+10
        numPixels = cv2.countNonZero(labelMask)        

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 50:
            mask = cv2.add(mask, labelMask)
    show(mask,0,"Labels")
    
    #Get the ROI label color (at center)
    centerLabel=mask[CROP_SIZE-1,CROP_SIZE-1]
    roiCenter=mask[CROP_SIZE-5:CROP_SIZE+5,CROP_SIZE-5:CROP_SIZE+5]
    
    #get label with most pixels at center
    modals, counts = np.unique(roiCenter, return_counts=True)
    show(roiCenter,0,"Roi Center")
    
    index = np.argmax(counts)
    centerLabel=modals[index]
    
    #TODO:hole in center? bug? best way needed to detect ROI
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    
    #Now clear everything else this is now ROI mask
    mask[mask!=centerLabel]=0
    roiArray = original[np.where(mask!=0)]
    roiMask=mask.copy()
    show(mask,3,"ROI")
    
    #ROI stats
    roiMean = np.mean(roiArray)
    roiPixels = len(roiArray)
    roiMedian = np.median(roiArray)
    roiMax = np.max(roiArray)
    roiHighCount = np.count_nonzero(roiArray>230)
    
    log("Max:  %d roiHighCount %d"%(roiMax,roiHighCount),1)
    
    #Calcifications, kinda...
    if (roiMax>=250):
            log("Max > 250: %d, skip."%roiMax)
            return False
    
    
    roiMin = np.min(roiArray)
    
    log("roiStats")
    log((roiPixels,roiMean,roiMedian,roiMin,roiMax))
    
    roisDataFrame.loc[dfIndex,"roiMean"]=roiMean
    roisDataFrame.loc[dfIndex,"roiMedian"]=roiMedian
    roisDataFrame.loc[dfIndex,"roiPixels"]=roiPixels
    roisDataFrame.loc[dfIndex,"roiMax"]=roiMax
    roisDataFrame.loc[dfIndex,"roiMin"]=roiMin
    
    #Get contours from ROI mask, should be only one
    im2, contours, _ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    
    #other except ROI, how many & area
    
    #Get roi color
    #show(allRois,0,"All Rois")
    roiCenter=allRois[CROP_SIZE-10:CROP_SIZE+10,CROP_SIZE-10:CROP_SIZE+10]
    modals, counts = np.unique(roiCenter, return_counts=True)
    index = np.argmax(counts)
    roiColor=modals[index]
    
    
    log("roiColor2 %d"%roiColor,0)
    
    #Check ROI contour
    cv2.drawContours(allRois, contours, -1, 100, 1)
    show(allRois,3,"Roi Contour")
    
    #Fill ROI contour with empty
    cv2.fillPoly(allRois, pts =contours, color=0)
    show(allRois,2,"remove ROI")
    
    #Now filter only for other locations with same color as ROI
    allRois[allRois!=roiColor]=0
    show(allRois,3,"clear everything but ROI (other ROIs)")
    _, otherRoisContours, _ = cv2.findContours(allRois,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(allRois, otherRoisContours, -1, 100, 1)
    show(allRois,1,"other ROI countours")
    otherRoisCount=len(otherRoisContours)
    
    log("other rois count %d"%len(otherRoisContours))
    
    roiCenterX=CROP_SIZE
    roiCenterY=CROP_SIZE
    
    #Check other/near ROIs areas & distance to ROI
    otherRoisAreas=[0]
    otherRoisDists=[0]
    for otherCnt in otherRoisContours:        
        otherRoisAreas.append(cv2.contourArea(otherCnt))
        ((otherX, otherY), radius)=cv2.minEnclosingCircle(otherCnt)
        otherDist=np.sqrt((otherX-roiCenterX)**2+(otherY+roiCenterY)**2)
        otherRoisDists.append(otherDist)
    
    otherRoisAreas=np.array(otherRoisAreas)
    otherRoisDists=np.array(otherRoisDists)
    
    log("otherRoisAreas %s"%otherRoisAreas)    
    log("otherRoisDists %s"%otherRoisDists)
    otherRoisWeigthDists=otherRoisAreas/(otherRoisDists/2+.00001) 
    log("otherRoisDists %s"%(otherRoisWeigthDists))
   
    otherRoisAreasAvg=np.mean(otherRoisAreas)
    otherRoisAreasMedian=np.median(otherRoisAreas)
    otherRoisAreasMax=np.max(otherRoisAreas)
    otherRoisAreasTotal=np.sum(otherRoisAreas)
    
    roisDataFrame.loc[dfIndex,"otherRoisCount"]=len(otherRoisContours)
    roisDataFrame.loc[dfIndex,"otherRoisAreasAvg"]=otherRoisAreasAvg
    roisDataFrame.loc[dfIndex,"otherRoisAreasMedian"]=otherRoisAreasMedian
    roisDataFrame.loc[dfIndex,"otherRoisAreasMax"]=otherRoisAreasMax
    roisDataFrame.loc[dfIndex,"otherRoisAreasTotal"]=otherRoisAreasTotal
        
    otherRoisWeigthDistsAvg=np.mean(otherRoisWeigthDists)
    otherRoisWeigthDistsMedian=np.median(otherRoisWeigthDists)
    otherRoisWeigthDistsMax=np.max(otherRoisWeigthDists)
    otherRoisWeigthDistsTotal=np.sum(otherRoisWeigthDists)
    
    roisDataFrame.loc[dfIndex,"otherRoisWeigthDistsAvg"]=otherRoisWeigthDistsAvg
    roisDataFrame.loc[dfIndex,"otherRoisWeigthDistsMedian"]=otherRoisWeigthDistsMedian
    roisDataFrame.loc[dfIndex,"otherRoisWeigthDistsMax"]=otherRoisWeigthDistsMax
    roisDataFrame.loc[dfIndex,"otherRoisWeigthDistsTotal"]=otherRoisWeigthDistsTotal
    
    log("other rois stats:")
    log((otherRoisAreasMedian,otherRoisAreasAvg,otherRoisAreasMax,otherRoisAreasTotal))
    
    
    #Erode/reduce
    eroded = morphology.erosion(allRois,np.ones([4,4]))
    show(eroded,2,"other ROIs eroded")
    
    #Now, select ROI contour (top 1 by area desc) 
    cv2.drawContours(mask, contours, -1, 100, 1)
    show(mask,1,"ROI countours")
    log("countours: %s"%len(contours))
    
    sortedcnts = sorted(contours, key=cv2.contourArea, reverse=True)
    
    #Roi contour (top 1 by area)
    cnt=sortedcnts[0]
    roiArea = cv2.contourArea(cnt)
    roiHighCountPerc=roiHighCount/roiArea
    log("roiHighCountperc %f"%roiHighCountPerc,1)
    roisDataFrame.loc[dfIndex,"roiArea"]=roiArea
    
    roisDataFrame.loc[dfIndex,"roiHighCountPerc"]=roiHighCountPerc
    roisDataFrame.loc[dfIndex,"roiHighCount"]=roiHighCount
    
    log("roiArea %d"%roiArea)
    x,y,w,h = cv2.boundingRect(cnt)
    roisDataFrame.loc[dfIndex,"roiRectX"]=x
    roisDataFrame.loc[dfIndex,"roiRectY"]=y
    roisDataFrame.loc[dfIndex,"roiRectW"]=w
    roisDataFrame.loc[dfIndex,"roiRectH"]=h
    
    #Draw findings/fits for ROI
    cv2.rectangle(mask,(x,y),(x+w,y+h),(0,255,0),1)
    (circlex,circley),radius = cv2.minEnclosingCircle(cnt)
    
    roisDataFrame.loc[dfIndex,"roiCircleX"]=circlex
    roisDataFrame.loc[dfIndex,"roiCircleY"]=circley
    roisDataFrame.loc[dfIndex,"roiCircleRadius"]=radius
    
    
    center = (int(circlex),int(circley))
    radius = int(radius)
    cv2.circle(mask,center,radius,30,1)
    
    ellipse = cv2.fitEllipse(cnt)
    (elx,ely),(elMA,elma),elangle=ellipse
    roisDataFrame.loc[dfIndex,"roiElx"]=elx
    roisDataFrame.loc[dfIndex,"roiEly"]=ely
    roisDataFrame.loc[dfIndex,"roiElMA"]=elMA
    roisDataFrame.loc[dfIndex,"roiElma_"]=elma
    roisDataFrame.loc[dfIndex,"roiElAngle"]=elangle
        
    cv2.ellipse(mask,ellipse,60,1)
    
    show(mask,0,"ROI fits")
    
    #Try at roughness
    perimeter = cv2.arcLength(cnt,True)
    roisDataFrame.loc[dfIndex,"perimeter"]=perimeter
    hull = cv2.convexHull(cnt)
    
    hullperimeter = cv2.arcLength(hull,True)
    roisDataFrame.loc[dfIndex,"hullperimeter"]=hullperimeter
    roughness = perimeter/hullperimeter
    roisDataFrame.loc[dfIndex,"roughness"]=roughness
    hull2 = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull2)
    
    log((area,perimeter,hullperimeter,roughness,len(defects)),1)
    
    #try to get roughness estimate using area from defect triangles
    totalDefectArea=0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        #cv2.line(mask,start,end,100,1)
        defectArea=cv2.contourArea(np.array([start,end,far]))
        log("defectArea #%d: %f"%(i,defectArea))
        totalDefectArea=totalDefectArea+defectArea
        cv2.circle(mask,far,5,50,1)
    
    roisDataFrame.loc[dfIndex,"hullDefects"]=len(defects)
    roisDataFrame.loc[dfIndex,"totalDefectArea"]=totalDefectArea
    roisDataFrame.loc[dfIndex,"totalDefectAreaPerc"]=totalDefectArea/roiArea
    
    show(mask,1,"defects %d"% (len(defects)))
    
    log("wall dists %s" % wallDists,1)
    
    show(lungsMask,3,"mask/dists")
    return True


# In[ ]:


#Ok, let's test it
LOG_LEVEL=3
SHOW_PLOTS=True
SEGMENT_ONLY=False

#Search ROIs on case between 15% and 20% z Index (not usable out of 15% to ~55%)

#Cancer
dfRois=segmentdAndSearchCase("0c60f4b87afcb3e2dfa65abbbf3ef2f9",0.15,.55)
dfRois


# In[ ]:


#Search specific slice
dfRois=segmentdAndSearchSlice("0c60f4b87afcb3e2dfa65abbbf3ef2f9/03c3db59b8b029a11803f28f81e4870e.dcm")
dfRois


# In[ ]:


#Search specific slice by index
dfRois=segmentdAndSearchInstanceNumber("0c60f4b87afcb3e2dfa65abbbf3ef2f9",24)
dfRois


# In[ ]:


#Cancer
segmentdAndSearchCase("0c0de3749d4fe175b7a5098b060982a1")


# In[ ]:


#No cancer but ROI
segmentdAndSearchCase("00cba091fa4ad62cc3200a657aeb957e")


# In[ ]:


#Cancer
segmentdAndSearchCase("0acbebb8d463b4b9ca88cf38431aac69")


# In[ ]:


#Cancer, Some False Negatives
segmentdAndSearchCase("0d06d764d3c07572074d468b4cff954f")
segmentdAndSearchCase("0c37613214faddf8701ca41e6d43f56e")

