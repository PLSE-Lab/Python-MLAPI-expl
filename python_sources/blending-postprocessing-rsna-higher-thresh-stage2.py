#!/usr/bin/env python
# coding: utf-8

# 

# ## Phases of post-processing for RSNA predictions

# In[ ]:


on_kaggle = True


# In[ ]:


# Non-max suppression parameters for phase 0
OTHRESH=0.05    # Maximum acceptable overlap
PTHRESH=0.94    # Confidence threshold
AVPTHRESH=0.65  # Minimum average confidence
# Class probability threshold for phase 1
THRESH = .175
# Confidence threshold for phase 2
MIN_MAX_MRCNN_CONF = .975
# Parameters for phase 3
MINPROB = 0.69  # Value to set for below threshold confidence (<0.7)
C1 = 1e6  # Logistic regularization parameters
C2 = 0.06
# Parameters for phase 5
UNET_MINPROB = 0.28  # Minimum probability to add Unet cases
UNET_MINCONF = 0.35  # Minimum confidence to add Unet cases
# Parameters for phase 6
YOLO_MINCONF = 0.15  # Minimum confidence for a yolo box to be considered
MINMIN = 0.20  # Minimum minimum probability in an iteration of yolo additions before stopping


# In[ ]:


import numpy as np 
import pandas as pd 
import os
from sklearn import metrics
from scipy.special import logit,expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


# In[ ]:


os.listdir('../input')


# In[ ]:


get_ipython().system('ls ../input/gs-dense-chexnet-predict-test-from-all-models')


# In[ ]:


RESULTS_LOC = '../input'  # Outputs from base models
RAW_DATA_LOC = '../input/rsna-pneumonia-detection-challenge'          # Raw input data
YOLO_SUB_STEM = 'yolov3-inference-from-multiple-saved-wts-stage-2/submission_yolo'    # YoloV3 predictions
YOLO_SUB_STEM2 = 'yolov3-inference-from-more-saved-weights-stage-2/submission_yolo'    # YoloV3 predictions
MORE_YOLO_EPOCHS = [5300, 1500, 6700, 2000] if on_kaggle else []
UNET_RESULTS_PATH = '../input/resnet-unet-from-saved-weights-stage-2/submission_resnet34unet_thr0.35.csv'  # Resnet-Unet predictions
CHEXNET_OOF_STEM = 'rsna-oof-predictions/predictions_valid_fold_'        # Giulia's classification model
CHEXNET_TEST_STEM = 'gs-dense-chexnet-predict-stage-2-from-all-models/test_preds_pth_fold'
# ^ but fold0 may be different
SPECIAL_FILE0 = 'gs-dense-chexnet-predict-stage-2-from-all-models/test_preds_pth_fold0_for_combined_folds'
CHEXNET_TEST_FILE0 = SPECIAL_FILE0 if on_kaggle else CHEXNET_TEST_STEM + '0'
ADENSE_TEST_STEM = 'andy-densenet-from-multiple-saved-weights-stage-2/submission_adense_f'                    # Andy's classification model
ADEMSE_OOF_STEM = 'rsna-oof-predictions/val_dense_v5_a1_f'
MRCNN_TEST_STEM = 'mrcnn-inference-from-saved-wts-4dig-70-stage-2/submission_mrcnn_v'  # MRCNN predicted boxes
# ^ need to fix v8 vs v9 (same weights, different output, but here both used v8 name, so 'v8'+'',
#   instead of 'v'+'9')
MRCNN95_TEST_STEM = 'mrcnn-inference-from-saved-wts-2dig-95-stage-2/submission_mrcnn_v'    if on_kaggle else MRCNN_TEST_STEM  # MRCNN predicted boxes
MRCNN_OUT_STEM = 'temp_mrcnn_v' if on_kaggle else MRCNN_TEST_STEM
MRCNN_OOF_STEM = 'rsna-oof-predictions/val_v'
MRCNN_OOF_OUT_STEM = 'val_v' if on_kaggle else MRCNN_OOF_STEM
MRCNN_TEST_SEP = ''
MRCNN_OOF_SEP = '_'
IN_OOF_SUFFIX = ''
IN_SUFFIX = '.csv'
OUT_SUFFIX = '.csv'
CHEX_CLASS_PRED_PATH = '../input/combine-fold-class-predictions-stage-2/test_probs.csv'  # Logisitc average of Giulia's test class preds
KERNEL_OUTPUT_PATH = '../input/filter-199-final-with-higher-thresh-stage-2/filter199.csv'      # "Pneumonia - Segm. filtered through Class." kernel


# In[ ]:


# For phase 3 only
N_FOLDS = 5
OUTPUT_LOC = '.'

OOF_INFILE_STEM = MRCNN_OOF_STEM + '9' + MRCNN_OOF_SEP + 'a1' + MRCNN_OOF_SEP + 'f'
vers = '8' if on_kaggle else '9'
TEST_INFILE_STEM = MRCNN_TEST_STEM + vers + MRCNN_TEST_SEP + 'a1' + MRCNN_TEST_SEP + 'f'
OOF_OUTFILE_NAME = MRCNN_OOF_OUT_STEM + '10' + MRCNN_OOF_SEP + 'a1' +                    MRCNN_OOF_SEP + 'allfolds_out.csv'
#TEST_OUTFILE_STEM = MRCNN_TEST_STEM + '10' + MRCNN_TEST_SEP + 'a1' + MRCNN_TEST_SEP + 'f_out'
# above won't work on kaggle
TEST_OUTFILE_STEM = MRCNN_OUT_STEM + '10' + MRCNN_TEST_SEP + 'a1' + MRCNN_TEST_SEP + 'f_out'
OOF_CLASS_PROBS_OUTFILE = 'phase3_oof_out.csv'
TEST_CLASS_PROBS_OUTFILE_STEM = 'phase3_test_out'
FULL_TEST_CLASS_PROBS_OUTFILE = 'phase3_test_avg_out.csv'


# ### Phase 0:<br> Apply non-max suppression to fold predictions for 2 different 5-fold assignments

# In[ ]:


dfs = []

for a in range(1,3):
    for f in range(5):
        vers = '8' if a==2 and f==0 else '1'
        fn = RESULTS_LOC +'/'+ MRCNN95_TEST_STEM + vers + MRCNN_TEST_SEP +              'a' + str(a) + MRCNN_TEST_SEP + 'f' + str(f) + IN_SUFFIX
        dfs.append( pd.read_csv(fn).set_index('patientId') )

for i,f in enumerate(dfs):
    if i:
        df = df.join(f.rename(columns={'PredictionString':'pred'+str(i)}))
    else:
        df = f.rename(columns={'PredictionString':'pred'+str(i)})


# In[ ]:


# Implementation of non-max suppression from
#   https://github.com/jrosebr1/imutils/blob/master/imutils/object_detection.py
def non_max_suppression(boxes, probs=None, overlapThresh=OTHRESH):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int"), list(np.array(probs)[pick])


# In[ ]:


box_dict = {}
conf_dict = {}
for pat,row in df.iterrows():
    boxes = []
    confs = []
    maxconfs = []
    for pred in row:
        maxconf = 0.
        if isinstance(pred, str):
            s = pred.split(' ')
            if s[-1]=='':
                s.pop()  # remove terminating null
            if s[0]=='':
                s.pop(0)  # remove initial null
            if ( len(s)%5 ):
                print( 'Bad prediction string.')
            while len(s):
                conf = float(s.pop(0))
                x = int(round(float(s.pop(0))))
                y = int(round(float(s.pop(0))))
                w = int(round(float(s.pop(0))))
                h = int(round(float(s.pop(0))))
                if conf>maxconf:
                    maxconf = conf
                if conf>PTHRESH:
                    boxes.append( [x,y,x+w,y+h] )
                    confs.append( conf )
        maxconfs.append(maxconf)
    avgconf = sum(maxconfs)/len(maxconfs)
    if len(boxes) and avgconf>AVPTHRESH:
        box_dict[pat] = boxes
        conf_dict[pat] = confs
len(box_dict), len(conf_dict)


# In[ ]:


box_dict_nms = {}
conf_dict_nms = {}
for p in box_dict:
    boxes, confs = non_max_suppression(np.array(box_dict[p]), np.array(conf_dict[p]))
    box_dict_nms[p] = boxes
    conf_dict_nms[p] = confs


# In[ ]:


sub_dict = {}
for p in df.index:
    predictionString = ''
    if p in box_dict_nms:
        for box, conf in zip(box_dict_nms[p], conf_dict_nms[p]):
            # retrieve x, y, height and width
            x, y, x2, y2 = box
            height = y2 - y
            width = x2 - x
            # add to predictionString
            predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
    sub_dict[p] = predictionString

# save submission file
sub = pd.DataFrame.from_dict(sub_dict,orient='index')
sub.index.names = ['patientId']
sub.columns = ['PredictionString']
phase0_output = sub


# ### Phase 1<br>Apply classification probability threshold to non-max suppression output

# In[ ]:


probs = pd.read_csv(CHEX_CLASS_PRED_PATH)


# In[ ]:


df = phase0_output.join(probs.set_index('patientId'))
out = df.copy()
out.loc[df.prob<THRESH,'PredictionString'] = np.nan
out.loc[df.PredictionString=='','PredictionString'] = np.nan
phase1_output = out.drop(['prob'],axis=1)


# ### Phase 2<br>Add high-confidnence results from phase1 to results form Kaggle kernel

# In[ ]:


df1 = pd.read_csv(KERNEL_OUTPUT_PATH).set_index('patientId')
df2 = phase1_output
df1_cases = df1[~df1.PredictionString.isnull()]
df2_cases = df2[~df2.PredictionString.isnull()]
df1_pos_ids = df1_cases.index.values
df2_pos_ids = df2_cases.index.values


# In[ ]:


df1_only_dict = {}
for p in df1_pos_ids:
    if not p in df2_pos_ids:
        df1_only_dict[p] = float(df1_cases.loc[p,'PredictionString'].split(' ')[0])


# In[ ]:


df2_only_dict = {}
for p in df2_pos_ids:
    if not p in df1_pos_ids:
        df2_only_dict[p] = float(df2_cases.loc[p,'PredictionString'].split(' ')[0])


# In[ ]:


df_out = df1.copy()
for p,r in df1.iterrows():
    if p in df2_only_dict:
        if df2_only_dict[p] > MIN_MAX_MRCNN_CONF:
            df_out.loc[p,'PredictionString'] = df2.loc[p,'PredictionString']
phase2_output = df_out


# In[ ]:


phase2_output.head()


# ### Phase 3<br>Convert MRCNN confidence to fitted class proability

# In[ ]:


# Read in OOF predictions from MRCNN model

input_dir = RESULTS_LOC
infile_stem = OOF_INFILE_STEM
oof_preds_input = []
for ifold in range( N_FOLDS ):
    fp_in = os.path.join(input_dir, infile_stem + str(ifold) + IN_OOF_SUFFIX)
    oof_preds_input.append( pd.read_csv(fp_in) )
    oof_preds_input[-1]['fold'] = ifold
oof_preds = pd.concat(oof_preds_input).set_index('patientId')


# In[ ]:


# Convert to class probabilities by taking maximum conf for each patient
probs_dict = {}
for p, r in oof_preds.iterrows():
    probs_dict[p] = MINPROB
oof_positive = oof_preds[~oof_preds.PredictionString.isnull()]
for p, r in oof_positive.iterrows():
    s = r.PredictionString.split(' ')
    if s[-1]=='':
        s.pop()  # remove terminating null
    if s[0]=='':
        s.pop(0)  # remove initial null
    if ( len(s)%5 ):
        print( 'Bad prediction string.')
    while len(s):
        conf = float(s.pop(0))
        x = int(round(float(s.pop(0))))
        y = int(round(float(s.pop(0))))
        w = int(round(float(s.pop(0))))
        h = int(round(float(s.pop(0))))
        if conf>probs_dict[p]:
            probs_dict[p] = conf
prob_df = pd.DataFrame(probs_dict,index=['prob']).transpose()


# In[ ]:


# Read in actual classes and join with class predictions
input_dir = RAW_DATA_LOC
fp = os.path.join(input_dir, 'stage_2_train_labels.csv')
act = pd.read_csv(fp).set_index('patientId').rename(columns={'Target':'actual'})[['actual']]
act = act[~act.index.duplicated()]


# In[ ]:


df = act.join(prob_df,how='right')  # Have to do right join because using stage 1 OOF data
df.head()


# In[ ]:


# Fit logistic-on-logits model
rawprobs = df.prob.values.reshape(-1,1)
lr = LogisticRegression(C=C1)
lr.fit(logit(rawprobs),df.actual)
b1 = lr.coef_[0,0]
b0 = lr.intercept_[0]
b0, b1


# In[ ]:


# Patient IDs for folds
folds = []
for ifold in range(N_FOLDS):
    folds.append( oof_preds[oof_preds.fold==ifold].index.values )


# In[ ]:


def run_logistic_by_fold():
    df = act.join(prob_df,how='right')   # Have to do right join because using stage 1 OOF data
    logits = df.prob.apply(logit)
    df['oofprob'] = np.nan
    b1 = []
    b0 = []
    for i in range(len(folds)):

        fs = folds.copy()
        te = fs.pop(i)  # pop off current validation fold
        tr = np.concatenate(fs)  # the rest is for training

        # Divide the data
        Xtr = logits[tr].values.reshape(-1,1)
        Xte = logits[te].values.reshape(-1,1)
        ytr = df.actual[tr].copy()
        yte = df.actual[te].copy()

        # Fit and predict for this fold
        lr.fit(Xtr, ytr)
        df.loc[te,'oofprob'] = lr.predict_proba(Xte)[:,1]
        b1.append(lr.coef_[0,0])
        b0.append(lr.intercept_[0])

    coefs = pd.DataFrame( {'b0':b0, 'b1':b1}, index=range(len(folds)) )
    coefs.index.name = 'fold'

    return( df.sort_values('oofprob'), coefs )   


# In[ ]:


lr = LogisticRegression(C=C2)


# In[ ]:


df, coefs = run_logistic_by_fold()


# In[ ]:


# Transform confidences in OOF data to show realistic probabilities
pat_dict = {}
fold_dict = {}
for pat,row in oof_preds.iterrows():
    fold = row.fold
    pred = row.PredictionString
    boxes = []
    confs = []
    if isinstance(pred, str):
        s = pred.split(' ')
        if s[-1]=='':
            s.pop()  # remove terminating null
        if s[0]=='':
            s.pop(0)  # remove initial null
        if ( len(s)%5 ):
            print( 'Bad prediction string.')
        b0 = coefs.b0[fold]
        b1 = coefs.b1[fold]
        while len(s):
            conf = float(s.pop(0))
            x = int(round(float(s.pop(0))))
            y = int(round(float(s.pop(0))))
            w = int(round(float(s.pop(0))))
            h = int(round(float(s.pop(0))))
            boxes.append( [x,y,w,h] )
            confs.append( expit( b0+b1*logit(conf) ) )
    predictionString = ''
    if len(boxes):
        for box, conf in zip(boxes, confs):
            x, y, w, h = box
            # add to predictionString
            predictionString += '{:6.4f} '.format(conf)
            predictionString += str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' '
    pat_dict[pat] = predictionString
    fold_dict[pat] = fold
xform_oof_preds = pd.DataFrame(pat_dict, 
                               index=['PredictionString']).transpose()
xform_oof_preds.index.name = 'patientId'
xform_oof_preds['fold'] = [fold_dict[x] for x in xform_oof_preds.index.values]
xform_oof_preds.head()


# In[ ]:


output_dir = OUTPUT_LOC
xform_oof_preds.to_csv(os.path.join(output_dir, OOF_OUTFILE_NAME))


# In[ ]:


input_dir = RESULTS_LOC
output_dir = OUTPUT_LOC
infile_stem = TEST_INFILE_STEM
outfile_stem = TEST_OUTFILE_STEM


# In[ ]:


for ifold in range(N_FOLDS):
    
    fp_in = os.path.join(input_dir, infile_stem + str(ifold) + IN_SUFFIX)
    fp_out = os.path.join(output_dir, outfile_stem + str(ifold) + OUT_SUFFIX)
    inpreds = pd.read_csv(fp_in).set_index('patientId')
    pat_dict = {}

    for pat,row in inpreds.iterrows():
        pred = row.PredictionString
        boxes = []
        confs = []
        if isinstance(pred, str):
            s = pred.split(' ')
            if s[-1]=='':
                s.pop()  # remove terminating null
            if s[0]=='':
                s.pop(0)  # remove initial null
            if ( len(s)%5 ):
                print( 'Bad prediction string.')
            while len(s):
                conf = float(s.pop(0))
                x = int(round(float(s.pop(0))))
                y = int(round(float(s.pop(0))))
                w = int(round(float(s.pop(0))))
                h = int(round(float(s.pop(0))))
                boxes.append( [x,y,w,h] )
                confs.append( expit( b0+b1*logit(conf) ) )
        predictionString = ''
        if len(boxes):
            for box, conf in zip(boxes, confs):
                x, y, w, h = box
                # add to predictionString
                predictionString += '{:6.4f} '.format(conf)
                predictionString += str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' '
        pat_dict[pat] = predictionString
    outpreds = pd.DataFrame(pat_dict, 
                            index=['PredictionString']).transpose()
    outpreds.index.name = 'patientId'
    outpreds.to_csv(fp_out)
fp_out


# In[ ]:


# Generate class probability file for OOF data
outprobs = prob_df.join(oof_preds[['fold']])
b0_ = coefs.b0[outprobs.fold].values
b1_ = coefs.b1[outprobs.fold].values
outprobs['prob'] =  expit( b0_ + logit(outprobs.prob)*b1_ )
outprobs.index.name = 'patientId'
output_dir = OUTPUT_LOC
outprobs.to_csv(os.path.join(output_dir, OOF_CLASS_PROBS_OUTFILE))
outprobs.head()


# In[ ]:


# Generate class probability files for test data (by training fold)
output_dir = OUTPUT_LOC
box_outfile_stem = TEST_OUTFILE_STEM
probs_outfile_stem = TEST_CLASS_PROBS_OUTFILE_STEM
minprob_out = expit( b0 + b1*logit(MINPROB))

for ifold in range(N_FOLDS):
    
    fp_in = os.path.join(output_dir, box_outfile_stem + str(ifold) + OUT_SUFFIX)
    inpreds = pd.read_csv(fp_in).set_index('patientId')
    fp_out = os.path.join(output_dir, probs_outfile_stem + str(ifold) + OUT_SUFFIX)

    probs_dict = {}

    for p, r in inpreds.iterrows():
        probs_dict[p] = minprob_out
    inpreds_positive = inpreds[~inpreds.PredictionString.isnull()]
    for p, r in inpreds_positive.iterrows():
        s = r.PredictionString.split(' ')
        if s[-1]=='':
            s.pop()  # remove terminating null
        if s[0]=='':
            s.pop(0)  # remove initial null
        if ( len(s)%5 ):
            print( 'Bad prediction string.')
        while len(s):
            conf = float(s.pop(0))
            s.pop(0)
            s.pop(0)
            s.pop(0)
            s.pop(0)
            if conf>probs_dict[p]:
                probs_dict[p] = conf
    outprob_df = pd.DataFrame(probs_dict,index=['prob']).transpose()
    outprob_df.index.name = 'patientId'
    outprob_df.to_csv(fp_out)
    if not ifold:
        allprobs_df = outprob_df.rename(columns={'prob':ifold})
    else:
        allprobs_df = allprobs_df.join(outprob_df.rename(columns={'prob':ifold}))


# ### Phase 4<br>Stack class probability estimates

# In[ ]:


fp = OOF_CLASS_PROBS_OUTFILE
oof_probs_mrcnn = pd.read_csv(fp).set_index('patientId').rename(columns={'prob':'ph'})


# In[ ]:


for i in range(N_FOLDS):
    fp = TEST_CLASS_PROBS_OUTFILE_STEM + str(i) + '.csv'
    indf = pd.read_csv(fp).set_index('patientId').rename(columns={'prob':'ph'+str(i)})
    if i:
        test_prob_df = test_prob_df.join(indf)
    else:
        test_prob_df = indf


# In[ ]:


test_prob_df.head()


# In[ ]:


coefs


# In[ ]:


df.tail()


# In[ ]:


f0 = pd.read_csv(RESULTS_LOC + '/' + ADEMSE_OOF_STEM + '0.csv').set_index('patientId')
f1 = pd.read_csv(RESULTS_LOC + '/' + ADEMSE_OOF_STEM + '1.csv').set_index('patientId')
f2 = pd.read_csv(RESULTS_LOC + '/' + ADEMSE_OOF_STEM + '2.csv').set_index('patientId')
f3 = pd.read_csv(RESULTS_LOC + '/' + ADEMSE_OOF_STEM + '3.csv').set_index('patientId')
f4 = pd.read_csv(RESULTS_LOC + '/' + ADEMSE_OOF_STEM + '4.csv').set_index('patientId')


# In[ ]:


den = pd.concat([f0,f1,f2,f3,f4],axis=0)
print( den.shape )
den.head()


# In[ ]:


tf0 = pd.read_csv(RESULTS_LOC + '/' + ADENSE_TEST_STEM + '0.csv').set_index('patientId')
tf1 = pd.read_csv(RESULTS_LOC + '/' + ADENSE_TEST_STEM + '1.csv').set_index('patientId')
tf2 = pd.read_csv(RESULTS_LOC + '/' + ADENSE_TEST_STEM + '2.csv').set_index('patientId')
tf3 = pd.read_csv(RESULTS_LOC + '/' + ADENSE_TEST_STEM + '3.csv').set_index('patientId')
tf4 = pd.read_csv(RESULTS_LOC + '/' + ADENSE_TEST_STEM + '4.csv').set_index('patientId')


# In[ ]:


test_den = tf0.rename(columns={'predicted':'pa0'})
test_den = test_den.join(tf1.rename(columns={'predicted':'pa1'}))
test_den = test_den.join(tf2.rename(columns={'predicted':'pa2'}))
test_den = test_den.join(tf3.rename(columns={'predicted':'pa3'}))
test_den = test_den.join(tf4.rename(columns={'predicted':'pa4'}))
print( test_den.shape )
test_den.head()


# In[ ]:


# Patient IDs for folds
p0 = f0.index.values
p1 = f1.index.values
p2 = f2.index.values
p3 = f3.index.values
p4 = f4.index.values
pt = test_den.index.values


# In[ ]:


c0 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_OOF_STEM + '0.csv').set_index('patientId')
c1 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_OOF_STEM + '1.csv').set_index('patientId')
c2 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_OOF_STEM + '2.csv').set_index('patientId')
c3 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_OOF_STEM + '3.csv').set_index('patientId')
c4 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_OOF_STEM + '4.csv').set_index('patientId')
chex = pd.concat([c0,c1,c2,c3,c4],axis=0)
print( chex.shape )
chex.head()


# In[ ]:


df = den.join(chex[['validPredProba']]).rename(columns={'predicted':'pa','validPredProba':'pg'})
df = df.join(oof_probs_mrcnn).drop(['fold'],axis=1)
print( df.shape )
df.head()


# In[ ]:


tc0 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_TEST_FILE0 + '.csv').set_index('patientId')
tc1 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_TEST_STEM + '1.csv').set_index('patientId')
tc2 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_TEST_STEM + '2.csv').set_index('patientId')
tc3 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_TEST_STEM + '3.csv').set_index('patientId')
tc4 = pd.read_csv(RESULTS_LOC + '/' + CHEXNET_TEST_STEM + '4.csv').set_index('patientId')
tc0.head()


# In[ ]:


INFINITY = 100  # No regularization
lr = LogisticRegression(C=INFINITY)


# In[ ]:


test_chex = tc0.rename(columns={'targetPredProba':'pg0'}).drop(['targetPred'],axis=1)
test_chex = test_chex.join(tc1.rename(columns={'targetPredProba':'pg1'}).drop(['targetPred'],axis=1))
test_chex = test_chex.join(tc2.rename(columns={'targetPredProba':'pg2'}).drop(['targetPred'],axis=1))
test_chex = test_chex.join(tc3.rename(columns={'targetPredProba':'pg3'}).drop(['targetPred'],axis=1))
test_chex = test_chex.join(tc4.rename(columns={'targetPredProba':'pg4'}).drop(['targetPred'],axis=1))
print( test_chex.shape )
test_chex.head()


# In[ ]:


all_test_probs = test_prob_df.join(test_den.join(test_chex))
all_test_probs.head()


# In[ ]:


test_sets = []
test_set = ['pa','pg','ph']
for i in range(5):
    this_set = [n + str(i) for n in test_set]
    test_sets.append(this_set)
test_sets


# In[ ]:


yps = []
MAXCONF = .996
Xtr = df.drop('actual',axis=1).copy()
ytr = df.actual.copy()
X_train_adjust = Xtr
X_train_adjust.loc[Xtr.ph>MAXCONF,'ph'] = MAXCONF
X_train_logit = X_train_adjust.apply(logit)
lr.fit(X_train_logit, ytr)
for tset in test_sets:
    Xte = all_test_probs[tset].copy()
    Xte.columns = test_set
    X_test_adjust = Xte
    X_test_adjust.loc[Xte.ph>MAXCONF,'ph'] = MAXCONF
    X_test_logit = X_test_adjust.apply(logit)
    yp = lr.predict_proba(X_test_logit)[:,1]
    yps.append(yp)
len(yps), [len(y) for y in yps]


# In[ ]:


p = pd.DataFrame(yps,columns=Xte.index).transpose().apply(logit).mean(axis=1).apply(expit)
class_preds = pd.DataFrame(p,columns=['prob'])
class_preds.index.name = 'patientId'
print(class_preds.shape)
class_preds.head()


# In[ ]:


phase4_test_output = class_preds


# In[ ]:


yps = []
ids = []
MAXCONF = .995
folds = [p0,p1,p2,p3,p4]
for i in range(len(folds)):
    fs = folds.copy()
    te = fs.pop(i)
    tr = np.concatenate(fs)
    Xtr = df.loc[tr,:].drop('actual',axis=1).copy()
    Xte = df.loc[te,:].drop('actual',axis=1).copy()
    ytr = df.actual[tr].copy()
    yte = df.actual[te].copy()
    X_train_adjust = Xtr
    X_test_adjust = Xte
    X_train_adjust.loc[Xtr.ph>MAXCONF,'ph'] = MAXCONF
    X_test_adjust.loc[Xte.ph>MAXCONF,'ph'] = MAXCONF
    X_train_logit = X_train_adjust.apply(logit)
    X_test_logit = X_test_adjust.apply(logit)
    lr.fit(X_train_logit, ytr)
    yp = lr.predict_proba(X_test_logit)[:,1]
    yps = yps + list(yp)
    ids = ids + list(Xte.index.values)


# In[ ]:


oof_class_preds = pd.DataFrame({'prob':yps},index=ids)
oof_class_preds.index.name = 'patientId'
print(oof_class_preds.shape)
oof_class_preds.head()


# In[ ]:


phase4_oof_output = oof_class_preds


# ### Phase 5<br>Add high-confidence Resnet-uunet results to results from phase 2

# In[ ]:


df1 = phase2_output
df2 = pd.read_csv(UNET_RESULTS_PATH).rename(columns={
    'predictionString':'PredictionString'}).set_index('patientId')

class_preds = phase4_test_output
class_probs = class_preds.prob.to_dict()

df1_cases = df1[~df1.PredictionString.isnull()]
df2_cases = df2[~df2.PredictionString.isnull()]

df1_pos_ids = df1_cases.index.values
df2_pos_ids = df2_cases.index.values


# In[ ]:


df1_only_dict = {}
for p in df1_pos_ids:
    if not p in df2_pos_ids:
        df1_only_dict[p] = float(df1_cases.loc[p,'PredictionString'].split(' ')[0])

df2_only_dict = {}
for p in df2_pos_ids:
    if not p in df1_pos_ids:
        df2_only_dict[p] = float(df2_cases.loc[p,'PredictionString'].split(' ')[0])
        
candidates = pd.DataFrame(pd.Series(df2_only_dict,name='conf')).join(class_preds,how='left')
accepted = candidates[ (candidates.prob>UNET_MINPROB) & (candidates.conf>UNET_MINCONF)
                     ].index.values


# In[ ]:


accepted


# In[ ]:


df_out = df1.copy()
for p,r in df1.iterrows():
    if p in accepted:
        df_out.loc[p,'PredictionString'] = df2.loc[p,'PredictionString']
phase5_output = df_out
phase5_output.head()


# ### Phase 6<br>Add high-confidnece yolo results to results from phase 5

# In[ ]:


yolo_epochs = [1500, 2000, 3200, 4100, 5300, 6700, 10000, 13700]


# In[ ]:


df1 = phase5_output

dfs = []
for eps in yolo_epochs:
    yolo_stem = YOLO_SUB_STEM2 if (eps in MORE_YOLO_EPOCHS) else YOLO_SUB_STEM
    dfs.append( 
        pd.read_csv(RESULTS_LOC+'/'+yolo_stem+str(eps)+'.csv').set_index('patientId'))

class_preds = phase4_test_output
class_probs = class_preds.prob.to_dict()

df1_cases = df1[~df1.PredictionString.isnull()]
df1_pos_ids = df1_cases.index.values
included = df1_pos_ids.tolist()
df_out = df1.copy()


# In[ ]:


while True:
    for df2 in dfs:
        df2_cases = df2[~df2.PredictionString.isnull()]
        df2_pos_ids = df2_cases.index.values
        df2_only_dict = {}
        outstring_dict = {}
        for p in df2_pos_ids:
            if not p in included:
                maxconf = 0
                boxes = []
                confs = []
                s = df2.loc[p,'PredictionString'].split(' ')
                if s[-1]=='':
                    s.pop()  # remove terminating null
                if s[0]=='':
                    s.pop(0)  # remove initial null
                if ( len(s)%5 ):
                    print( 'Bad prediction string.')
                while len(s):
                    conf = float(s.pop(0))
                    x = int(round(float(s.pop(0))))
                    y = int(round(float(s.pop(0))))
                    w = int(round(float(s.pop(0))))
                    h = int(round(float(s.pop(0))))
                    if (conf>YOLO_MINCONF):
                        boxes.append( [x,y,w,h] )
                        confs.append( conf )
                        if conf>maxconf:
                            maxconf = conf
                predictionString = ''
                if len(boxes):
                    for box, conf in zip(boxes, confs):
                        x, y, w, h = box
                        # add to predictionString
                        predictionString += '{:6.4f} '.format(conf)
                        predictionString += str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' '
                        df2_only_dict[p] = maxconf*class_probs[p]
                outstring_dict[p] = predictionString
        candidates = pd.DataFrame(pd.Series(df2_only_dict,name='prod'))
        best = candidates.sort_values('prod').tail(1).index.values[0]
        print(best,candidates.loc[best,'prod'])
        df_out.loc[best,'PredictionString'] = outstring_dict[best]
        included.append( best )
    minprob = class_preds.loc[included[-8:],:].prob.min()
    if minprob<MINMIN:
        break


# In[ ]:


class_preds.loc[included[-32:],:]   # This was 16 in stage 1. Changed to 32 for curiosity.


# In[ ]:


df_out.to_csv('phase6_output.csv')


# In[ ]:


df_out.head()


# In[ ]:




