import numpy as numpy
import numpy as np
import pandas as pd
import cv2
from numpy import pi, arctan2, exp, log
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SZ = 28

def deskew(img):
	#print img, img.dtype
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		# no deskewing needed. 
		return img.copy()
	# Calculate skew based on central momemts. 
	skew = m['mu11']/m['mu02']
	# Calculate affine transform to correct skewness. 
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	# Apply affine transform
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

def limits(mask):
	# Compute some basic cross-section quantities:
	# left edge, right edge, center, span, width, number of gaps/holes, and width of largest hole.
	if not mask.any():
		return [0, 99, 99, 99, 0, 99, 99]
	indices = numpy.where(mask)[0]
	lo, hi = numpy.min(indices), numpy.max(indices)
	width_filled = numpy.sum(mask)
	width_total = hi - lo
	mean = numpy.mean(indices)
	ngaps = numpy.sum(indices[1:] - indices[:-1] != 1)
	if len(indices) > 1:
		largestgapsize = numpy.max(indices[1:] - indices[:-1])
	else:
		largestgapsize = 1
	return [lo, hi, mean, width_filled, width_total, ngaps, largestgapsize]

def imagefeatures(img):
	# calls the above function with a few thresholds
	# lets measure a few things:
	mid = numpy.median(img)
	max = numpy.max(img)
	thresh_lo = mid + 3 * numpy.std(img[img <= mid])
	thresh_hi = max / 3.
	thresh_vhi = max * 3. / 4.
	features = []
	for thresh in thresh_lo, thresh_hi, thresh_vhi:
		mask = img > thresh
		x = numpy.any(mask, axis=1)
		xlo, xhi, _, _, _, _, _ = limits(x)
		y = numpy.any(mask, axis=0)
		ylo, yhi, _, _, _, _, _ = limits(y)
		# measure width at 1/6, 2/6, 3/6, 4/6, 5/6
		# measure also the the number of holes
		features += limits(mask[:,int(ylo*5/6.+yhi*1/6.)])
		features += limits(mask[:,int(ylo*4/6.+yhi*2/6.)])
		features += limits(mask[:,int(ylo*3/6.+yhi*3/6.)])
		features += limits(mask[:,int(ylo*2/6.+yhi*4/6.)])
		features += limits(mask[:,int(ylo*1/6.+yhi*5/6.)])
		features += limits(mask[int(xlo*5/6.+xhi*1/6.),:])
		features += limits(mask[int(xlo*4/6.+xhi*2/6.),:])
		features += limits(mask[int(xlo*3/6.+xhi*3/6.),:])
		features += limits(mask[int(xlo*2/6.+xhi*4/6.),:])
		features += limits(mask[int(xlo*1/6.+xhi*5/6.),:])
		# 0123456789
		
		# mark the quadrant where most ink is
		i, j = numpy.where(mask)
		#ictr = (numpy.mean(i) - (xhi+xlo)/2.) / (xhi - xlo+1)
		ictr = int(numpy.mean(i))
		#jctr = (numpy.mean(j) - (yhi+ylo)/2.) / (yhi - xlo+1)
		jctr = int(numpy.mean(j))
		features += [int(ictr), int(jctr)]
	return features

def print_img_mask(mask):
	# simple visualisation for masked images
	for row in mask:
		print(' '.join(['#' if col else ' ' for col in row]))
	print()

def caterpillar_path(mask, startpos):
	# the caterpillar eats ink and moves to the nearest ink
	# thereby tracing out the strokes needed to paint the digit
	istart, jstart = startpos
	img = (mask*255).astype(np.uint8)
	size = np.size(img)
	skel = np.zeros(img.shape, np.uint8)
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	while True:
		eroded = cv2.erode(img,element)
		skel = cv2.bitwise_or(skel, cv2.subtract(img, cv2.dilate(eroded,element)))
		img = eroded
		zeros = size - cv2.countNonZero(img)
		if zeros==size:
			break
	
	i0, j0 = istart, jstart
	i1, j1 = i0, j0
	mask = skel > 0
	#mask = mask.copy()
	i, j = numpy.where(mask)
	path = []
	while True:
		path.append((i0,j0))
		#print_img_mask(mask)
		# eat i,j and around there
		eaten = (i0-i)**2 + (j0-j)**2 < 4**2
		"""
		# eating should be wider sideways than forward
		R = 6
		weight = 0.3
		ibefore = i0 + (i0-i1) * -0.2
		jbefore = j0 + (j0-j1) * -0.2
		iahead = i0 + (i0-i1) * 0.2
		jahead = j0 + (j0-j1) * 0.2
		eaten1 = (i0-ibefore)**2 + (j0-jbefore)**2 < R**2
		eaten2 = (i0-iahead)**2 + (j0-jahead)**2 < R**2
		eaten = numpy.logical_and(eaten1, eaten2)
		"""
		#print 'eating %d' % eaten.sum()
		for k in numpy.where(eaten)[0]:
			mask[i[k], j[k]] = False
		# find nearest remaining one
		if not mask.any():
			return path
		i, j = numpy.where(mask)
		dist = (i0-i)**2 + (j0-j)**2
		forward_weight = 0.3
		inext = i0 + (i0-i1) * forward_weight
		jnext = j0 + (j0-j1) * forward_weight
		dist = (inext-i)**2 + (jnext-j)**2
		# objects in our current moving direction appear closer
		k = numpy.argmin(dist)
		i1, j1 = i0, j0
		i0, j0 = i[k], j[k]

def step_angles(path):
	# takes a caterpillar path and computes some statistics:
	# number of jumps (indicating separate strokes)
	# angles between steps (to find turn-arounds, e.g. for 3)
	angles = []
	angles2 = []
	jumps = []

	for (i1,j1),(i0,j0) in zip(path[1:], path[:-1]):
		step_length2 = ((i0-i1)**2 + (j0-j1)**2)
		angles.append(numpy.arctan2(j1-j0,i1-i0))
		jumps.append(step_length2>5**2)

	for (i2,j2),(i1,j1),(i0,j0) in zip(path[2:], path[1:-1], path[:-2]):
		angles2.append(numpy.arctan2(j2-j0,i2-i0))

	angles = numpy.asarray(angles)
	angles2 = numpy.asarray(angles2)
	jumps = numpy.asarray(jumps)

	deltaangles = numpy.min([
		numpy.abs(angles[1:] - angles[:-1]), 
		numpy.abs(angles[1:] - angles[:-1] + 2*pi),
		numpy.abs(angles[1:] - angles[:-1] - 2*pi),
	], axis=0)
	
	# are there big jumps?
	njumps = jumps.sum()
	# now we can add features about this shape
	# is it all continuous or are there 90 degree angles?
	nonjumpangles = deltaangles[~jumps[:-1]]
	pathangle = numpy.median(nonjumpangles)
	ncorners = (nonjumpangles * 180 / pi > 80).sum()
	# are there any back and forth steps?
	if jumps.any():
		jumpangles = deltaangles[~jumps[:-1]]
		jump_angle = (jumpangles * 180 / pi).max()
	else:
		jump_angle = 0
	
	features = [njumps, ncorners, pathangle, jump_angle]
	return features, jumps

def caterpillar(img):
	# runs several caterpillar runs so that it starts
	# from a likely end point
	# notes the quadrant of the end points and jumps (stroke starts/ends)
	
	mid = numpy.median(img)
	thresh_lo = mid + 3 * numpy.std(img[img <= mid])
	mask = img > thresh_lo
	i, j = numpy.where(mask)
	path = caterpillar_path(mask, (i[0], j[0]))
	i0, j0 = path[-1]
	path = caterpillar_path(mask, (i0, j0))
	i0, j0 = path[-1]
	path = caterpillar_path(mask, (i0, j0))
	# now characterise curve
	features, jumps = step_angles(path)
	
	# we can also make notes about the end points of the curve
	x = numpy.any(mask, axis=1)
	xlo, xhi, _, _, _, _, _ = limits(x)
	y = numpy.any(mask, axis=0)
	ylo, yhi, _, _, _, _, _ = limits(y)
	
	quadrant_has_edges = numpy.zeros((3,3), dtype=bool)
	endpoints = [path[0], path[-1]]
	# coords before and after jump
	endpoints += [p for p, j in zip(path, jumps[:-1]) if j]
	endpoints += [p for p, j in zip(path, jumps[1:]) if j]
	
	for i, j in endpoints:
		# assign to a quadrant
		ih = (i - (xhi + xlo)/2.) / (xhi - xlo)
		jh = (j - (yhi + ylo)/2.) / (yhi - ylo)
		if abs(ih) < 0.2:
			iq = 1
		elif ih < 0.2:
			iq = 2
		else:
			iq = 0
		if abs(jh) < 0.2:
			jq = 1
		elif jh < 0.2:
			jq = 2
		else:
			jq = 0
		quadrant_has_edges[iq,jq] = True
	features += (quadrant_has_edges*1).flatten().tolist()
	return features

print("reading training data...")
data = pd.read_csv('../input/train.csv')
X = data.loc[:,"pixel0":"pixel783"]
y = data.label
train = numpy.array(X).reshape((-1,SZ,SZ)).astype(numpy.uint8)
trainpart = train
#trainpart = train[:1000]

img = trainpart[0]
features = imagefeatures(deskew(img))
features += caterpillar(img)
nfeatures = len(features)

print("convert training data into feature space...")
fout = open('train_cv.csv', 'w')
fout.write('label,' + ','.join(['feat%d' % i for i in range(nfeatures)]) + '\n')

for i, img in enumerate((trainpart)):
	dimg = deskew(img)
	features = imagefeatures(dimg)
	features += caterpillar(dimg)
	assert len(features) == nfeatures, (i, len(features), nfeatures)
	fout.write('%d' % y[i])
	for f in features:
		fout.write(',%d' % f)
	fout.write('\n')
fout.close()

print("reading training data...")
data = pd.read_csv('../input/test.csv')
test = numpy.array(data).reshape((-1,SZ,SZ)).astype(numpy.uint8)

print("convert test data into feature space ...")
fout = open('test_cv.csv', 'w')
fout.write(','.join(['feat%d' % i for i in range(nfeatures)]) + '\n')

for i, img in enumerate((test)):
	dimg = deskew(img)
	features = imagefeatures(dimg)
	features += caterpillar(dimg)
	assert len(features) == nfeatures, (i, len(features), nfeatures)
	for f in features:
		fout.write(',%d' % f)
	fout.write('\n')
fout.close()







# now we train the classifier on the features
# in this case, we use a Random Forest

n_estimators = 500


print('loading...')
data = pd.read_csv('train_cv.csv')
trainX = data.loc[:,"feat0":"feat228"]
trainY = data.label

i = int(len(trainX)*0.8)
trainX1 = trainX[:i]
trainY1 = trainY[:i]
train1 = train[:i]
testX1 = trainX[i:]
testY1 = trainY[i:]
test1 = train[i:]
clf = RandomForestClassifier(n_estimators = n_estimators, n_jobs=3)
clf.fit(trainX1, trainY1)
pred = clf.predict(testX1)
for i, (img, predi, labeli) in enumerate(zip(test1, pred, testY1)):
	if predi != labeli:
		print('Image %d mispredicted as %d should be %d' % (i, predi, labeli))
		cv2.imwrite('problematic-%d-%d-as-%d.png' % (i, labeli, predi), img)


data = pd.read_csv('test_cv.csv')
testX = data.loc[:,"feat0":"feat228"]

clf = RandomForestClassifier(n_estimators = n_estimators, n_jobs=3)
print('training...')
clf.fit(trainX, trainY)
print('predicting...')
testy = clf.predict(testX)

print('saving...')
fout = open('test_cv_randomforest_predict.csv', 'w')
fout.write("ImageId,Label\n")
for i, label in enumerate(testy):
	fout.write("%d,%d\n" % (i+1, label))

