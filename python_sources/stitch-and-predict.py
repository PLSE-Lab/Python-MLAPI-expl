#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
from PIL import ImageFilter
import multiprocessing
import random; random.seed(2016);
import cv2
import re
import os, glob

sample_sub = pd.read_csv('../input/sample_submission.csv')
train_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/train_sm/*.jpeg")])
train_files.columns = ['path', 'group', 'pic_no']
test_files = pd.DataFrame([[f,f.split("/")[3].split(".")[0].split("_")[0],f.split("/")[3].split(".")[0].split("_")[1]] for f in glob.glob("../input/test_sm/*.jpeg")])
test_files.columns = ['path', 'group', 'pic_no']
print(len(train_files),len(test_files),len(sample_sub))
train_images = train_files[train_files["group"]=='set107']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 0
a = []
for l in train_images.path:
    im = cv2.imread(l)
    plt.subplot(5, 2, i_+1).set_title(l)
    plt.hist(im.ravel(),256,[0,256]); plt.axis('off')
    a.append([im.mean(),im.max(),im.min()])
    plt.subplot(5, 2, i_+2).set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 2
print(a)


# In[ ]:


kaze = cv2.KAZE_create()
akaze = cv2.AKAZE_create()
brisk = cv2.BRISK_create()

plt.rcParams['figure.figsize'] = (7.0, 18.0)
plt.subplots_adjust(wspace=0, hspace=0)
i = 0
for detector in [kaze, akaze, brisk]:
    start_time = time.time()
    im = cv2.imread(train_images.path[0])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    (kps, descs) = detector.detectAndCompute(gray, None)       
    cv2.drawKeypoints(im, kps, im, (0, 255, 0))
    plt.subplot(3, 1, i+1).set_title(list(['kaze','akaze','brisk'])[i] + " " + str(round(((time.time() - start_time)/60),5)))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i+=1


# In[ ]:


print(cv2.__version__)

img1 = cv2.imread(train_images.path[0], 0)
img2 = cv2.imread(train_images.path[1], 0)
brisk = cv2.BRISK_create()
kp1, des1 = brisk.detectAndCompute(img1,None)
kp2, des2 = brisk.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img1 = cv2.imread(train_images.path[0])
img2 = cv2.imread(train_images.path[1])
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100], flags=2, outImg=img2, matchColor = (0,255,0))
plt.rcParams['figure.figsize'] = (14.0, 8.0)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)); plt.axis('off')


# In[ ]:


brisk = cv2.BRISK_create()
dm = cv2.DescriptorMatcher_create("BruteForce")

def c_resize(img, ratio):
    wh = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
    img = cv2.resize(img, wh, interpolation = cv2.INTER_AREA)
    return img
    
def im_stitcher(imp1, imp2, imsr = 1.0, withTransparency=False):
    img1 = cv2.imread(imp1, 0)
    img2 = cv2.imread(imp2, 0)
    if imsr < 1.0:
        img1 = c_resize(img1,imsr); img2 = c_resize(img2,imsr)
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    kp1, des1 = brisk.detectAndCompute(img1,None)
    kp2, des2 = brisk.detectAndCompute(img2,None)
    matches = dm.knnMatch(des1,des2, 2)
    matches_ = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches_.append((m[0].trainIdx, m[0].queryIdx))
    kp1_ = np.float32([kp1[m[1]].pt for m in matches_]).reshape(-1,1,2)
    kp2_ = np.float32([kp2[m[0]].pt for m in matches_]).reshape(-1,1,2)
    H, mask = cv2.findHomography(kp2_,kp1_, cv2.RANSAC, 4.0)
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    img1 = cv2.imread(imp1)
    img2 = cv2.imread(imp2)
    if imsr < 1.0:
        img1 = c_resize(img1,imsr); img2 = c_resize(img2,imsr)
    im = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    if withTransparency == True:
        h3,w3 = im.shape[:2]
        bim = np.zeros((h3,w3,3), np.uint8)
        bim[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
        im = cv2.addWeighted(im,1.0,bim,0.9,0)
    else:
        im[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return im


# In[ ]:


img = im_stitcher(train_images.path[0], train_images.path[4], 0.5, True)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(img); plt.axis('off')


# In[ ]:


img = cv2.imread(train_images.path[0])
cv2.imwrite('panoramic.jpeg',img)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
for i in range(1,5):
    img = im_stitcher(train_images.path[i], 'panoramic.jpeg', 0.5, False)
    cv2.imwrite('panoramic.jpeg',img)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')


# In[ ]:


train_images = train_files[train_files["group"]=='set4']
train_images = train_images.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)
img = cv2.imread(train_images.path[0])
cv2.imwrite('panoramic2.jpeg',img)
plt.rcParams['figure.figsize'] = (12.0, 12.0)
for i in range(1,5):
    img = im_stitcher(train_images.path[i], 'panoramic2.jpeg', 0.5, False)
    cv2.imwrite('panoramic2.jpeg',img)
img[np.where((img < [20,20,20]).all(axis = 2))] = [255,255,255]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')


# In[ ]:


import time; start_time = time.time()
import warnings; warnings.filterwarnings('ignore');
import multiprocessing
from sklearn import ensemble
from sklearn import pipeline, grid_search
from sklearn.metrics import label_ranking_average_precision_score as lraps

def image_features(path, tt, group, pic_no):
    im = cv2.imread(path)
    me_ = cv2.mean(im)
    s=[path, tt, group, pic_no, im.mean(), me_[2], me_[1], me_[0]]
    f = open("data.csv","a")
    f.write((',').join(map(str, s)) + '\n')
    f.close()
    return

f = open("data.csv","w");
col = ['path','tt', 'group', 'pic_no', 'individual_im_mean','rm','bm','gm']
f.write((',').join(map(str,col)) + '\n')
f.close()

if __name__ == '__main__':
    cpu = multiprocessing.cpu_count(); print (cpu);
    
    j = []
    for s_ in range(0,len(train_files),cpu):     #train
        for i in range(cpu):
            i_=s_+i
            if (i_)<len(train_files):
                if i_ % 100 == 0:
                    print("train ", i_)
                filename = train_files.path[i_]
                p = multiprocessing.Process(target=image_features, args=(filename,'train', train_files["group"][i_], train_files["pic_no"][i_],))
                j.append(p)
                p.start()
    j = []
    for s_ in range(0,len(test_files),cpu):     #test
        for i in range(cpu):
            i_=s_+i
            if (i_)<len(test_files):
                if i_ % 100 == 0:
                    print("test ", i_)
                filename = test_files.path[i_]
                p = multiprocessing.Process(target=image_features, args=(filename,'test', test_files["group"][i_], test_files["pic_no"][i_],))
                j.append(p)
                p.start()
    
    while len(j) > 0: #end all jobs
        j = [x for x in j if x.is_alive()]
        time.sleep(1)
    df_all = pd.read_csv('data.csv', index_col=None)
    df_all = df_all.reset_index(drop=True)
    df_all['group_min_im_mean'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['individual_im_mean'].min())
    df_all['group_max_im_mean'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['individual_im_mean'].max())
    df_all['group_mean'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['individual_im_mean'].mean())
    df_all['a'] = df_all['individual_im_mean'] - df_all['group_min_im_mean']
    df_all['b'] = df_all['group_max_im_mean'] - df_all['individual_im_mean']
    df_all['c'] = df_all['group_mean'] - df_all['individual_im_mean']
    #red
    df_all['group_min_im_mean_r'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['rm'].min())
    df_all['group_max_im_mean_r'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['rm'].max())
    df_all['group_mean_r'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['rm'].mean())
    df_all['a_r'] = df_all['rm'] - df_all['group_min_im_mean_r']
    df_all['b_r'] = df_all['group_max_im_mean_r'] - df_all['rm']
    #df_all['c_r'] = df_all['group_mean_r'] - df_all['rm']
    #green
    df_all['group_min_im_mean_g'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['gm'].min())
    df_all['group_max_im_mean_g'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['gm'].max())
    df_all['group_mean_g'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['gm'].mean())
    df_all['a_g'] = df_all['gm'] - df_all['group_min_im_mean_g']
    df_all['b_g'] = df_all['group_max_im_mean_g'] - df_all['gm']
    #df_all['c_g'] = df_all['group_mean_g'] - df_all['gm']
    #blue
    df_all['group_min_im_mean_b'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['bm'].min())
    df_all['group_max_im_mean_b'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['bm'].max())
    df_all['group_mean_b'] = df_all["group"].map(lambda x: df_all[df_all['group']==x]['bm'].mean())
    df_all['a_b'] = df_all['bm'] - df_all['group_min_im_mean_b']
    df_all['b_b'] = df_all['group_max_im_mean_b'] - df_all['bm']
    #df_all['c_b'] = df_all['group_mean_b'] - df_all['bm']
    
    df_all['setId'] = df_all["group"].map(lambda x: x.replace('set',''))
    df_all.to_csv('data.csv')
    print("Features Ready: ", round(((time.time() - start_time)/60),2))
    
    X_train = df_all[df_all['tt'] == 'train']
    X_train = X_train.sort_values(by=['setId','pic_no'], ascending=[1, 1])
    X_train = X_train.reset_index(drop=True)
    y_train = X_train["pic_no"].values
    X_train = X_train.drop(['path','tt','group','pic_no','setId','individual_im_mean','group_min_im_mean','group_max_im_mean','group_mean', 'rm','group_min_im_mean_r','group_max_im_mean_r','group_mean_r', 'gm','group_min_im_mean_g','group_max_im_mean_g','group_mean_g', 'bm','group_min_im_mean_b','group_max_im_mean_b','group_mean_b'], axis=1)
    X_test = df_all[df_all['tt'] == 'test']
    X_test = X_test.sort_values(by=['setId','pic_no'], ascending=[1, 1])
    #X_test.fillna(0, inplace=True)
    X_test = X_test.reset_index(drop=True)
    id_test = X_test[["setId","pic_no"]] #.values
    X_test = X_test.drop(['path','tt','group','pic_no','setId','individual_im_mean','group_min_im_mean','group_max_im_mean','group_mean', 'rm','group_min_im_mean_r','group_max_im_mean_r','group_mean_r', 'gm','group_min_im_mean_g','group_max_im_mean_g','group_mean_g', 'bm','group_min_im_mean_b','group_max_im_mean_b','group_mean_b'], axis=1)
    rfr = ensemble.RandomForestClassifier(n_estimators = 50, n_jobs = -1, random_state = 2016, verbose = 0)
    param_grid = {'max_depth': [6], 'max_features': [1.0]}
    model = grid_search.GridSearchCV(estimator = rfr, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 0)
    model.fit(X_train, y_train)
    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:", model.best_score_)
    y_pred = model.predict_proba(X_test)
    #y_pred = model.predict(X_test)
    df = pd.concat((pd.DataFrame(id_test), pd.DataFrame(y_pred)), axis=1)
    df.columns = ['setId','pic_no','day1','day2','day3','day4','day5']
    #df.to_csv('submission2.csv',index=False)
    f = open('submission.csv', 'w')
    f.write('setId,day\n')
    setID = df.setId.unique()
    for i in setID:
        a = []
        df1 = df[df['setId'] == str(i)].reset_index(drop=True)
        for j in range(1,6):
            df1 = df1.sort_values(by=['day'+str(j)], ascending=[0]).reset_index(drop=True)
            #print(df1)
            a.append(df1.pic_no[0])
            df1 = df1[1:]
        f.write(str(i)+","+" ".join(map(str,a))+"\n")
        #break
    f.close()
    print("Ready to submit: ", round(((time.time() - start_time)/60),2))


# In[ ]:




