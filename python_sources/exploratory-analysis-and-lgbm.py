#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#this kernel is based on personal work and other kernels found on kaggle 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import random
from datetime import datetime
import json
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#print(os.listdir("../input/petfinder-adoption-prediction/"))
print(os.listdir("../input/train"))
print(os.listdir("../input/test"))

# Any results you write to the current directory are saved as output.

def plog(msg):
    print(datetime.now(), msg)

train_df = pd.read_csv("../input/train/train.csv")
test_df = pd.read_csv("../input/test/test.csv")
breed_labels = pd.read_csv("../input/breed_labels.csv")
state_labels = pd.read_csv("../input/state_labels.csv")
color_labels = pd.read_csv("../input/color_labels.csv")
#with open('../input/breed_attributes.json', 'r') as f:
    #breed_attributes = json.load(f)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[ ]:


from enum import Enum

class Columns(Enum):
    rescuer_id = ["RescuerID"]
    ind_cont_columns = ["Age", "Fee", "VideoAmt", "PhotoAmt","Quantity",
                        "DescScore", "DescMagnitude", "DescLength", "SentMagnitude", "SentMagnitute_Mean", 
                                   "SentScore", "SentScore_Mean", "EntSalience", "EntSalience_Mean",
                        "NameLength","Pet_Maturity", "Color_Type","Fee_Per_Pet"]
    ind_num_cat_columns = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                           "Vaccinated", "Dewormed", "Sterilized", "Health", "State", "MaturitySize",
                           "FurLength", "Pet_Breed", "Breed_Merge", "Pet_Purity", "Overall_Status"]
    ind_text_columns = ["Name", "Description", "Entities"]
    iden_columns = ["PetID"]
    dep_columns = ["AdoptionSpeed"]
    n_desc_svd_comp = 120
    desc_svd_cols = ["desc_svd_"+str(i) for i in range(n_desc_svd_comp)]
    n_desc_nmf_comp = 5
    desc_nmf_cols = ["desc_nmf_"+str(i) for i in range(n_desc_nmf_comp)]
    img_num_cols_all = ["P_RGB", "P_Dom_Px_Frac", "P_Dom_Px_Frac_Mean", "P_Dom_Score", "P_Dom_Score_Mean", "P_Vertex_X", "P_Vertex_Y",
                 "P_Bound_Conf", "P_Bound_Conf_Mean", "P_Bound_Imp_Frac", "P_Bound_Imp_Frac_Mean", "P_Label_Score", "P_Label_Score_Mean"]
    img_num_cols_1 = ["Vertex_X_1", "Vertex_Y_1", "Bound_Conf_1", "Bound_Imp_Frac_1",
                      "Dom_Blue_1", "Dom_Green_1", "Dom_Red_1",
                "RGBint_1", "Dom_Px_Fr_1", "Dom_Scr_1", "Lbl_Scr_1",]
    img_num_cols_2 = ["Vertex_X_2", "Vertex_Y_2", "Bound_Conf_2", "Bound_Imp_Frac_2",
                      "Dom_Blue_2", "Dom_Green_2", "Dom_Red_2",
                "RGBint_2", "Dom_Px_Fr_2", "Dom_Scr_2", "Lbl_Scr_2"]
    img_num_cols_3 = ["Vertex_X_3", "Vertex_Y_3", "Bound_Conf_3", "Bound_Imp_Frac_3",
                      "Dom_Blue_3", "Dom_Green_3", "Dom_Red_3",
                "RGBint_3", "Dom_Px_Fr_3", "Dom_Scr_3", "Lbl_Scr_3"]
    img_lbl_cols_1 = ["Lbl_Img_1"]
    img_lbl_cols_2 = ["Lbl_Img_2"]
    img_lbl_cols_3 = ["Lbl_Img_3"]
    img_lbl_col = ["Lbl_Dsc"]
    n_iann_svd_comp = 5
    iann_svd_cols = ["iann_svd_" + str(i) for i in range(n_iann_svd_comp)]
    n_iann_nmf_comp = 5
    iann_nmf_cols = ["iann_nmf_" + str(i) for i in range(n_iann_nmf_comp)]
    n_entities_svd_comp = 5
    entities_svd_cols = ["entities_svd_" + str(i) for i in range(n_entities_svd_comp)]
    n_entities_nmf_comp = 5
    entities_nmf_cols = ["entities_nmf_" + str(i) for i in range(n_entities_nmf_comp)]
    item_cnt_incols = ["RescuerID", "Breed1", "Breed2", "Breed_Merge", "Age"]#, "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    item_cnt_cols =  [c + "_Cnt" for c in item_cnt_incols]
    item_cnt_mtype_cols =  [c + "_Cnt_MType" for c in item_cnt_incols]
    item_type_incols = ["RescuerID_Cnt", "Breed1_Cnt", "Breed2_Cnt", "Breed_Merge_Cnt", "Age_Cnt"]#, "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    item_type_cols =  [c + "_StdType" for c in item_type_incols]
    kbin_incols = ["Age", "Fee", "Fee_Per_Pet"] + item_cnt_cols  #, "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    kbin_cols =  [c + "_Kbin" for c in kbin_incols]
    item_adp_incols = item_type_cols + item_cnt_cols + kbin_cols + item_cnt_mtype_cols#["RescuerID_Cnt_StdType", "Breed_Merge_Cnt_StdType"]#, "Breed1","Breed2", "Color1", "Color2", "Color3", "State"]
    item_adp_cols =  [c + "_Adp" for c in item_adp_incols]
    fee_mean_incols = ["Breed1", "Breed2", "Age", "Breed_Merge", "State"] + item_cnt_cols + item_cnt_mtype_cols + item_type_cols
    fee_mean_cols = ["Fee_Per_Pet_" + c for c in fee_mean_incols]
    loo_incols = ["Pet_Breed", "State"]
    loo_cols = [c + "_Loo" for c in loo_incols]
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    scaling_cols = ["Age", "Fee", "RescuerID_Cnt", "Breed1_Cnt", "Breed2_Cnt", "Breed_Merge_Cnt", "Age_Cnt", "Fee_Per_Pet"]  + fee_mean_cols
    
    ft_cat_cols = ["Breed1", "Breed2", "Breed_Merge", "Overall_Status"] + item_cnt_cols + item_type_cols + kbin_cols
    ft_new_cols = ["Age", "Fee", "Quantity", "VideoAmt", "PhotoAmt"]
    agg_calc = ["std", "skew", "mean", "max", "min"]
    
    def feature_cols(fcc, fnc, agg_calc):
        tmp_ft_cols = []
        for cc in fcc:
            # print(cc)
            for a in agg_calc:
                # print(a)
                if a != "COUNT":
                    for x in fnc:
                        # print(x)
                        tmp_ft_cols.append(cc + "_" + a + "(Pets." + x + ")")
                else:
                    tmp_ft_cols.append(cc + "_" + a + "(Pets)")
        return tmp_ft_cols

    ft_cols = feature_cols(ft_cat_cols, ft_new_cols, agg_calc)
    #"Type", "Gender", "Color1", "Color2", "Color3","Vaccinated", "Dewormed", "Sterilized", "FurLength",
    
    
    barplot_cols = ind_num_cat_columns + item_type_cols + kbin_cols + item_cnt_mtype_cols 
    boxplot_cols = ind_cont_columns + desc_svd_cols + desc_nmf_cols + img_num_cols_all + img_num_cols_1 + img_num_cols_2 + img_num_cols_3 + img_lbl_cols_1 + img_lbl_cols_2 + img_lbl_cols_3                     + iann_svd_cols + iann_nmf_cols + entities_svd_cols + entities_nmf_cols + ft_cols + item_cnt_cols  + item_adp_cols + fee_mean_cols 
    
plog("Done")


# In[ ]:


def leaveoneout(X, y):
    
    ret = pd.DataFrame(columns=Columns.loo_cols.value, index=None)
    R = 10
    sums = np.zeros([len(X), 1])
    lens = np.zeros([len(X), 1])
    randomness =1+np.random.normal(scale = 0.03, size=len(sums)).reshape(len(sums), 1)
    for c in X.columns.values:
        for uval in X[c].unique():
            indexes = X[X[c] == uval].index
            s_tot = y.loc[indexes].sum()
            l = len(indexes)-1+R
            lens[indexes, 0] = 1/l
            #print(uval, indexes, s_tot)
            for i in indexes:
                s_ind = s_tot - int(y.loc[i])
                sums[i,0] = s_ind
        ret[c+"_Loo"] = (sums*lens*randomness).ravel().tolist()
                
    return ret
if 1 == 0:
    for c in Columns.loo_cols.value:
        if c in train_df.columns.values.tolist():
            plog("Dropping "+c +" in train dataset")
            train_df.drop(c, axis=1, inplace=True)
            plog("Dropped "+c +" in train dataset")

    df_loo = leaveoneout(train_df[Columns.loo_incols.value], train_df["AdoptionSpeed"])
    train_df = pd.concat([train_df,df_loo], axis=1)
    #print(train_df.head())


    for c in Columns.loo_cols.value:
        if c in test_df.columns.values.tolist():
            plog("Dropping "+c +" in test dataset")
            test_df.drop(c, axis=1, inplace=True)
            plog("Dropped "+c +" in test dataset")

    for c in Columns.loo_incols.value:
        df_loo = train_df[[c,"AdoptionSpeed"]].groupby(c).agg({'AdoptionSpeed':['sum', 'count']})
        df_loo.columns = df_loo.columns.droplevel(0)
        df_loo[c+"_Loo"] = df_loo["sum"] / (df_loo["count"]+10)
        df_loo.drop(["sum","count"], axis=1, inplace=True)
        test_df = test_df.set_index(c).join(df_loo).reset_index()

    for c in Columns.loo_incols.value:
        if train_df[c].isna().any():
            print("null", c)
            #train_df[c].fillna(train_df[c].mean(), inplace=True)
plog("Done")



# In[ ]:


from sklearn.preprocessing import OneHotEncoder
import numpy.core.defchararray as np_f
from category_encoders.leave_one_out import LeaveOneOutEncoder

def onehotencoding(df):
    enc = OneHotEncoder(categories="auto")
    encodedcols = []
    for c in ["Type", "Breed1", "Breed2", "Gender"]: #Columns.ind_num_cat_columns.value:
        y = enc.fit_transform(df[[c]])
        col_name = enc.get_feature_names()
        col_name = np_f.replace(col_name.astype(str), 'x0', c)
        dfo = pd.DataFrame(columns=col_name, data=y.toarray())       
        df = pd.concat([df, dfo], axis=1)
        for ec in col_name:
            encodedcols.append(ec)
    return df, encodedcols

if 1 == 0:
    plog("One hot encoding started for train set")
    train_df, train_encodedcols = onehotencoding(train_df)
    plog("One hot encoding ended for train set")
    plog("One hot encoding started for test set")
    test_df, test_encodedcols = onehotencoding(test_df)
    plog("One hot encoding ended for test set")
    if train_encodedcols == test_encodedcols:
        plog("Columns are same on test and train sets for one hot encoded cols")
        ohc_cols = train_encodedcols
plog("Done")


# In[ ]:


if 1 == 0:
    breed_id = {}
    for id,name in zip(breed_labels.BreedID,breed_labels.BreedName):
        breed_id[id] = name

    cat_atts = breed_attributes['cat_breeds']
    dog_atts = breed_attributes['dog_breeds']

    cat_names = [i for i in cat_atts.keys()]
    #print(cat_names)
    dog_names = [i for i in dog_atts.keys()]
    #print(dog_names)

    def get_breed_atts(df, breed):
        i=1
        df_breed_ids = df[breed].unique()#(breed)["PetID"].count().reset_index()
        print(i, len(df_breed_ids))
        for id in df_breed_ids:
            #print(i, datetime.now())
            if id in breed_id.keys(): 
                name = breed_id[id] 
                if name in cat_names:
                    #print(cat_ratings[name])
                    for key in cat_atts[name].keys():
                        k = breed+"_"+key.strip().replace(" ", "_")
                        #if k in df.columns.values.tolist():
                        #    df.drop(k, axis=1, inplace=True)
                        if k not in df.columns.values.tolist():
                            df[k] = np.NAN
                        indexes = df.loc[:,breed] == id
                        df.loc[indexes, k] = cat_atts[name][key]
                        #print(df[df[breed]==id].head())
                        #df[k].head()
                if name in dog_names:
                    #print(dog_ratings[name])
                    for key in dog_atts[name].keys():
                        k = breed+"_"+key.strip().replace(" ", "_")
                        #if k in df.columns.values.tolist():
                        #    df.drop(k, axis=1, inplace=True)
                        if k not in df.columns.values.tolist():
                            df[k] = np.NAN
                        #print(len(df[df[breed]==id]), name, key, dog_atts[name][key])
                        indexes = df.loc[:,breed] == id
                        df.loc[indexes, k] = dog_atts[name][key]
                        #print(df[df[breed]==id].head())
                        #print(df[k].head())
            i += 1
        return df

    plog("Getting Breed1 attributes for train")
    train_df = get_breed_atts(train_df, "Breed1")
    plog("Getted Breed1 attributes for train")
    plog("Getting Breed2 attributes for train")
    train_df = get_breed_atts(train_df, "Breed2")
    plog("Getted Breed2 attributes for train")
    print(train_df.info())
    import sys
    sys.exit()
    plog("Getting Breed1 attributes for test")
    test_df = get_breed_atts(test_df, "Breed1")
    plog("Getted Breed1 attributes for test")
    plog("Getting Breed2 attributes for test")
    test_df = get_breed_atts(test_df, "Breed2")
    plog("Getted Breed2 attributes for test")


# In[ ]:


import json
#getting description sentiment analyses
    
def get_desc_anly(type, recalc):
    if recalc == 1:
        if type == "train":
            path = "../input/train_sentiment/"#../input/train_sentiment/
        elif type == "test":
            path = "../input/test_sentiment/"#../input/test_sentiment/
        plog("Getting description sentiment analysis for "+type+"_sentiment.csv")
        files = [f for f in os.listdir(path) if (f.endswith('.json') & os.path.isfile(path+f))]

        df = pd.DataFrame(columns=["PetID", "DescScore", "DescMagnitude"])
        i = 0
        for f in files:
            #print(path + f)
            with open(path+f, encoding="utf8") as json_data:
                data = json.load(json_data)
            #print(data)
            #pf = pd.DataFrame.from_dict(data, orient='index').T.set_index('index')

            #print(data["documentSentiment"]["score"],data["documentSentiment"]["magnitude"])
            df.loc[i]= [f[:-5],data["documentSentiment"]["score"],data["documentSentiment"]["magnitude"]]
            i = i+1
        #df.to_csv(type+"_sentiment.csv", index=False)
    elif recalc == 0:
        df = pd.read_csv(type+"_sentiment.csv")
    plog("Got description sentiment analysis for "+type+"_sentiment.csv")
    return df

rc = 1
if 1 == 0:
    train_snt = get_desc_anly("train",rc)
    test_snt = get_desc_anly("test", rc)

    train_df = train_df.set_index("PetID").join(train_snt.set_index("PetID")).reset_index()
    test_df = test_df.set_index("PetID").join(test_snt.set_index("PetID")).reset_index()
    #train_df.drop(["Unnamed: 0"], axis=1, inplace=True)
    #test_df.drop(["Unnamed: 0"], axis=1, inplace=True)


    #train_df["DescScore"].fillna(0, inplace=True)
    #train_df["DescMagnitude"].fillna(0, inplace=True)
    train_df["Description"].fillna("", inplace=True)
    train_df["Name"].fillna("", inplace=True)
    #test_df["DescScore"].fillna(0, inplace=True)
    #test_df["DescMagnitude"].fillna(0, inplace=True)
    test_df["Description"].fillna("", inplace=True)
    test_df["Name"].fillna("", inplace=True)
    train_df["DescLength"] = train_df["Description"].str.len()
    test_df["DescLength"] = test_df["Description"].str.len()
    train_df["NameLength"] = train_df["Name"].str.len()
    test_df["NameLength"] = test_df["Name"].str.len()

    #print(train_df.describe(include="all"))
    #print(test_df.describe(include="all"))
plog("Done")


# In[ ]:


import json
#getting description sentiment analyses
    
def get_desc_anly_v2(type, recalc):
    if recalc == 1:
        if type == "train":
            path = "../input/train_sentiment/"#../input/train_sentiment/
        elif type == "test":
            path = "../input/test_sentiment/"#../input/test_sentiment/
        plog("Getting description sentiment analysis for "+type+"_sentiment.csv")
        files = [f for f in os.listdir(path) if (f.endswith('.json') & os.path.isfile(path+f))]

        df = pd.DataFrame(columns=["PetID", "DescScore", "DescMagnitude", "SentMagnitude", "SentMagnitute_Mean", 
                                   "SentScore", "SentScore_Mean", "EntSalience", "EntSalience_Mean", "Entities"])
        i = 0
        for f in files:
            #print(path + f)
            with open(path+f, encoding="utf8") as json_data:
                data = json.load(json_data)
            #print(data)
            #pf = pd.DataFrame.from_dict(data, orient='index').T.set_index('index')
            s_magnitude = 0
            m_magnitude = 0
            s_score = 0
            m_score = 0
            num_sentences = 0
            for sentence in data.get("sentences"):
                s_magnitude += sentence.get("sentiment").get("magnitude",0)
                s_score += sentence.get("sentiment").get("score",0)
                num_sentences += 1
            if num_sentences > 0:
                m_magnitude = s_magnitude / num_sentences
                m_score = s_score / num_sentences
                
            s_salience = 0
            m_salience = 0
            s_entities = ""
            num_entities = 0
            for entity in data.get("entities"):
                s_salience += entity.get("salience",0)
                s_entities = s_entities + " " + entity.get("name", "")
                num_entities += 1
            if num_entities > 0:
                m_salience = s_salience / num_entities
            
            d_score = data["documentSentiment"]["score"]
            d_magnitude = data["documentSentiment"]["magnitude"]
            #print(data["documentSentiment"]["score"],data["documentSentiment"]["magnitude"])
            df.loc[i]= [f[:-5],d_score,  d_magnitude, s_magnitude, m_magnitude, s_score, m_score, s_salience, m_salience, s_entities]
            i = i+1
        #df.to_csv(type+"_sentiment.csv", index=False)
    elif recalc == 0:
        df = pd.read_csv(type+"_sentiment.csv")
    plog("Got description sentiment analysis for "+type+"_sentiment.csv")
    return df

rc = 1
if 1 == 1:
    
    for c in [ "DescScore", "DescMagnitude" , "SentMagnitude", "SentMagnitute_Mean", 
                                   "SentScore", "SentScore_Mean", "EntSalience", "EntSalience_Mean", "Entities"]:
    
        if c in train_df.columns.values.tolist():
            train_df.drop([c], axis=1, inplace=True)
        if c in test_df.columns.values.tolist():
            test_df.drop([c], axis=1, inplace=True)
        
    
    train_snt = get_desc_anly_v2("train",rc)
    test_snt = get_desc_anly_v2("test", rc)

    train_df = train_df.set_index("PetID").join(train_snt.set_index("PetID")).reset_index()
    test_df = test_df.set_index("PetID").join(test_snt.set_index("PetID")).reset_index()
    #train_df.drop(["Unnamed: 0"], axis=1, inplace=True)
    #test_df.drop(["Unnamed: 0"], axis=1, inplace=True)


    for c in [ "DescScore", "DescMagnitude" , "SentMagnitude", "SentMagnitute_Mean", 
                                   "SentScore", "SentScore_Mean", "EntSalience", "EntSalience_Mean"]:
        #train_df[c].fillna(0, inplace=True)
        #test_df[c].fillna(0, inplace=True)
        pass
    for c in ["Description", "Name", "Entities"]:
        train_df[c].fillna("", inplace=True)
        test_df[c].fillna("", inplace=True)
        
    train_df["DescLength"] = train_df["Description"].str.len()
    test_df["DescLength"] = test_df["Description"].str.len()
    train_df["NameLength"] = train_df["Name"].str.len()
    test_df["NameLength"] = test_df["Name"].str.len()

    #print(train_df.describe(include="all"))
    #print(test_df.describe(include="all"))
plog("Done")


# In[ ]:


def get_img_meta(type, img_num, recalc):
    #getting image analyse metadata
    if recalc == 1:
        if type == "train":
            path = "../input/train_metadata/"  
        else:
            path = "../input/test_metadata/" 
            
        if img_num == "1":
            cols = Columns.iden_columns.value + Columns.img_num_cols_1.value + Columns.img_lbl_cols_1.value
            df_imgs = pd.DataFrame(columns=cols)
        elif img_num == "2":
            cols = Columns.iden_columns.value + Columns.img_num_cols_2.value + Columns.img_lbl_cols_2.value
            df_imgs = pd.DataFrame(columns=cols)
        elif img_num == "3":
            cols = Columns.iden_columns.value + Columns.img_num_cols_3.value + Columns.img_lbl_cols_3.value
            df_imgs = pd.DataFrame(columns=cols)
        else:
            plog("This function supports images until 3rd, so img_num should be <= 3")
            return False
            
        plog("Getting image analyse metadata for " +type + str(img_num)+" files")

        images = [f for f in sorted(os.listdir(path)) if (f.endswith("-"+img_num+".json") & os.path.isfile(path + f))]

        i = 0
        for img in images:
            PetID = img[:-7]
            #print(i, PetID,k, img, (img[-6:-5]), l_petid)

            with open(path + img, encoding="utf8") as json_data:
                data = json.load(json_data)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('x',-1)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('y',-1)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0].get('confidence',-1)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('blue', 255)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('green', 255)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color'].get('red', 255)
            RGBint = (dominant_red << 16) + (dominant_green << 8) + dominant_blue
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0].get('pixelFraction', -1)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0].get('score', -1)

            if data.get('labelAnnotations'):
                label_description = ""
                label_score = 0
                j = 1
                for ann in data.get('labelAnnotations'):
                    if ann.get('score', 0) >= 0.00:
                        label_score = (ann.get('score', 0) + label_score) / j
                        label_description = label_description + " " + ann.get("description", "nothing")
                        j += 1
            else:
                label_description = 'nothing'
                label_score = -1

            df_imgs.loc[i, cols] = [PetID,vertex_x, vertex_y, bounding_confidence,bounding_importance_frac, 
                                    RGBint, dominant_blue, dominant_green, dominant_red, 
                            dominant_pixel_frac, dominant_score, label_score, label_description]

            i += 1

        #print(df_imgs.head())
        #df_imgs.to_csv(type+"_metadata-"+img_num+".csv", index=False)
    elif recalc == 0:
        df_imgs = pd.read_csv(type+"_metadata-"+img_num+".csv")
    plog("Got image analyse metadata for " +type + str(img_num)+" files")
    return df_imgs

rc = 1
if 1 == 0:
    train_metadata_1 = get_img_meta("train", "1", rc)
    train_metadata_2 = get_img_meta("train", "2", rc)
    train_metadata_3 = get_img_meta("train", "3", rc)
    test_metadata_1 = get_img_meta("test", "1", rc)
    test_metadata_2 = get_img_meta("test", "2", rc)
    test_metadata_3 = get_img_meta("test", "3", rc)

    train_metadata = train_metadata_1.set_index("PetID").join(train_metadata_2.set_index("PetID")).join(train_metadata_3.set_index("PetID")).reset_index()
    test_metadata = test_metadata_1.set_index("PetID").join(test_metadata_2.set_index("PetID")).join(test_metadata_3.set_index("PetID")).reset_index()

    #train_metadata["Lbl_Dsc"] = train_metadata["Lbl_Img_1"] + train_metadata["Lbl_Img_2"] + train_metadata["Lbl_Img_3"]
    #test_metadata["Lbl_Dsc"] = test_metadata["Lbl_Img_1"] + test_metadata["Lbl_Img_2"] + test_metadata["Lbl_Img_3"]

    #train_metadata["Lbl_Dsc"].fillna("", inplace=True)
    #test_metadata["Lbl_Dsc"].fillna("", inplace=True)

    train_metadata.drop(["Lbl_Img_1", "Lbl_Img_2", "Lbl_Img_3"], axis=1, inplace = True)
    test_metadata.drop(["Lbl_Img_1", "Lbl_Img_2", "Lbl_Img_3"], axis=1, inplace = True)

    #train_metadata.fillna(0, inplace=True)
    #test_metadata.fillna(0, inplace=True)

    train_df = train_df.set_index("PetID").join(train_metadata.set_index("PetID")).reset_index()
    test_df = test_df.set_index("PetID").join(test_metadata.set_index("PetID")).reset_index()

    #train_df["Lbl_Dsc"].fillna("", inplace=True)
    #test_df["Lbl_Dsc"].fillna("", inplace=True)

    for col in Columns.img_num_cols_1.value + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value:
        pass
        #train_df[col].fillna(train_df[col].mean(), inplace=True)
        #test_df[col].fillna(test_df[col].mean(), inplace=True)    

    print(train_df.columns.values)
    print(test_df.columns.values)
    plog("Done")


# In[ ]:


def get_all_img_meta(type, recalc):
    # getting image analyse metadata
    if recalc == 1:
        if type == "train":
            path = "../input/train_metadata/"
        else:
            path = "../input/test_metadata/"

        all_images = [f for f in sorted(os.listdir(path)) if
                  (f.endswith(".json") & os.path.isfile(path + f))]
        len_images = len(all_images)

        pets = list(set([f[:f.find("-")] for f in sorted(os.listdir(path)) if
                  (f.endswith(".json") & os.path.isfile(path + f))]))

        np_pets = np.asarray(pets).reshape(len(pets), 1)
        df_cols = Columns.img_num_cols_all.value + ["Lbl_Dsc"]
        #print(df_cols)
        np_data = np.zeros((len(pets),len(df_cols)))
        data = np.concatenate((np_pets,np_data), axis=1 )
        df_pet_img_meta = pd.DataFrame(columns=["PetID"]+df_cols, data=data)
        df_pet_img_meta["Lbl_Dsc"] = ""
        df_pet_img_meta.set_index("PetID", inplace=True)
        #print(df_pet_img_meta.head())

        h = 1
        for pet in pets:
            images = [k for k in all_images if pet in k]

            p_rgb = 0
            p_dominant_pixel_frac = 0
            p_dominant_score = 0
            p_vertex_x = 0
            p_vertex_y = 0
            p_bounding_confidence = 0
            p_bounding_importance_frac = 0
            p_label_score = 0
            p_label_description = ""
            num_of_images = len(images)

            for img in images:
                imgnum = img.split("-", 1)[1].strip(".json")
                with open(path + img, encoding="utf8") as json_data:
                    image = json.load(json_data)

                #print(img, h, len_images)
                h = h+1

                i_rgb = 0
                i2_rgb = 0
                i_dominant_pixel_frac = 0
                i_dominant_score = 0
                i_label_score = 0
                i_label_description = ""
                i_num_colors = 0
                i_num_label_annotations = 0

                i_vertex_x = image['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('x', 0)
                i_vertex_y = image['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2].get('y', 0)
                i_bounding_confidence = image['cropHintsAnnotation']['cropHints'][0].get('confidence', 0)
                i_bounding_importance_frac = image['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', 0)

                if image.get('imagePropertiesAnnotation').get("dominantColors").get("colors"):
                    i_num_colors = 0 #len(image.get('imagePropertiesAnnotation').get("dominantColors").get("colors"))
                    t_rgb = 0
                    for color in image.get('imagePropertiesAnnotation').get("dominantColors").get("colors"):
                        #print(color, color.get("color").get("red"))
                        if color.get('score', 0) >= 0.00:
                            r = color.get("color").get('red', 255)
                            g = color.get("color").get('green', 255)
                            b = color.get("color").get('blue', 255)
                            #rgbint = ((r << 16) + (g << 8) + b)
                            m_rgb = (r**2 + g**2 + b**2)/3
                            t_rgb = t_rgb + m_rgb
                            i_dominant_pixel_frac = i_dominant_pixel_frac + color.get('pixelFraction', 0)
                            i_dominant_score = i_dominant_score + color.get('score', 0)
                            i_num_colors = i_num_colors + 1

                    if i_num_colors>0:
                        if t_rgb > 0:
                            i_rgb = np.sqrt(t_rgb/i_num_colors)
                        if i_dominant_score > 0:
                            i_dominant_score = i_dominant_score/i_num_colors
                    #print(i_rgb, i_dominant_pixel_frac, i_dominant_score)
                if image.get('labelAnnotations'):
                    i_num_label_annotations = 0 #len(image.get('labelAnnotations'))
                    for ann in image.get('labelAnnotations'):
                        if ann.get('score', 0) >= 0.00:
                            i_label_score = ann.get('score', 0) + i_label_score
                            i_num_label_annotations = i_num_label_annotations + 1
                            if ann.get("description"):
                                i_label_description = i_label_description + " " + ann.get("description", "")

                    if i_num_label_annotations>0:
                        if i_label_score > 0:
                            i_label_score = i_label_score/i_num_label_annotations

                    #print(i_label_score, i_label_description)

                p_rgb = p_rgb + i_rgb**2
                p_dominant_pixel_frac = p_dominant_pixel_frac + i_dominant_pixel_frac
                p_dominant_score = p_dominant_score + i_dominant_score
                p_vertex_x = p_vertex_x + i_vertex_x
                p_vertex_y = p_vertex_y + i_vertex_y
                p_bounding_confidence = p_bounding_confidence + i_bounding_confidence
                p_bounding_importance_frac = p_bounding_importance_frac + i_bounding_importance_frac
                p_label_score = p_label_score + i_label_score
                p_label_description = p_label_description + " " + i_label_description

            p_rgb = np.sqrt(p_rgb/num_of_images)
            pm_dominant_pixel_frac = p_dominant_pixel_frac/num_of_images
            pm_dominant_score = p_dominant_score/num_of_images
            p_vertex_x = p_vertex_x/num_of_images
            p_vertex_y = p_vertex_y/num_of_images
            pm_bounding_confidence = p_bounding_confidence/num_of_images
            pm_bounding_importance_frac = p_bounding_importance_frac/num_of_images
            pm_label_score = p_label_score/num_of_images

            df_pet_img_meta.loc[pet, df_cols] = [p_rgb, p_dominant_pixel_frac, pm_dominant_pixel_frac, p_dominant_score, pm_dominant_score, 
                                                 p_vertex_x, p_vertex_y, p_bounding_confidence, pm_bounding_confidence, 
                                                 p_bounding_importance_frac, pm_bounding_importance_frac, p_label_score, pm_label_score,
                                                 p_label_description]
            #print(df_pet_img_meta.head())
        #print(df_pet_img_meta.head())
        #df_pet_img_meta.reset_index().to_csv(type + "_metadata_all.csv", index=False)
    elif recalc == 0:
        df_pet_img_meta = pd.read_csv(type + "_metadata_all.csv")

    return df_pet_img_meta

rc = 1
if 1 == 1:
    plog("Starting all images metadata collection on train")
    train_metadata = get_all_img_meta("train", rc)
    plog("Ended all images metadata collection on train")
    plog("Starting all images metadata collection on test")
    test_metadata = get_all_img_meta("test", rc)
    plog("Ended all images metadata collection on test")


    train_df = train_df.set_index("PetID").join(train_metadata).reset_index()
    test_df = test_df.set_index("PetID").join(test_metadata).reset_index()
    
    train_df["Lbl_Dsc"].fillna("", inplace=True)
    test_df["Lbl_Dsc"].fillna("", inplace=True)

    for col in Columns.img_num_cols_all.value:
        pass
        #train_df[col].fillna(train_df[col].mean(), inplace=True)
        #test_df[col].fillna(test_df[col].mean(), inplace=True)    

    #print(train_df.columns.values)
    #print(test_df.columns.values)
plog("Done")


# In[ ]:


#tfidf and tsvd implemendation on text columns description and first three image labels

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

def tfidf_2(train, n_comp, out_cols):
    stop_words = set(stopwords.words('english'))

    tfv = TfidfVectorizer(min_df=3, max_features=10000,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = stop_words)

    # Fit TFIDF
    tfv.fit(list(train))
    X = tfv.transform(train)

    svd = TruncatedSVD(n_components=n_comp, random_state=1337)
    svd.fit(X)
    plog("explained_variance_ratio_.sum() "+str(svd.explained_variance_ratio_.sum()))
    #print(svd.explained_variance_ratio_)
    X = svd.transform(X)
    df_svd = pd.DataFrame(X, columns=out_cols)
    
    return df_svd

plog("TSVD for train description started")
for c in Columns.desc_svd_cols.value:
    if c in train_df.columns.values:
        train_df.drop([c], axis=1, inplace=True)
svd_train_desc = tfidf_2(train_df.Description, Columns.n_desc_svd_comp.value, Columns.desc_svd_cols.value)
plog("TSVD for train description ended")
plog("TSVD for test description started")
for c in Columns.desc_svd_cols.value:
    if c in test_df.columns.values:
        test_df.drop([c], axis=1, inplace=True)
svd_test_desc = tfidf_2(test_df.Description, Columns.n_desc_svd_comp.value, Columns.desc_svd_cols.value)
plog("TFIDF for test description ended")

plog("TSVD for train image label started")
for c in Columns.iann_svd_cols.value:
    if c in train_df.columns.values:
        train_df.drop([c], axis=1, inplace=True)
svd_train_lbldsc = tfidf_2(train_df.Lbl_Dsc, Columns.n_iann_svd_comp.value, Columns.iann_svd_cols.value)
plog("TSVD for train image label ended")
plog("TSVD for test image label started")
for c in Columns.iann_svd_cols.value:
    if c in test_df.columns.values:
        test_df.drop([c], axis=1, inplace=True)
svd_test_lbldsc = tfidf_2(test_df.Lbl_Dsc, Columns.n_iann_svd_comp.value, Columns.iann_svd_cols.value)
plog("TSVD for test image label ended")

plog("TSVD for train desc entities started")
for c in Columns.entities_svd_cols.value:
    if c in train_df.columns.values:
        train_df.drop([c], axis=1, inplace=True)
svd_train_entities = tfidf_2(train_df.Entities, Columns.n_entities_svd_comp.value, Columns.entities_svd_cols.value)
plog("TSVD for train desc entities ended")
plog("TSVD for test desc entities started")
for c in Columns.entities_svd_cols.value:
    if c in test_df.columns.values:
        test_df.drop([c], axis=1, inplace=True)
svd_test_entities = tfidf_2(test_df.Entities, Columns.n_entities_svd_comp.value, Columns.entities_svd_cols.value)
plog("TSVD for test desc entities ended")

train_df = pd.concat([train_df,svd_train_desc, svd_train_lbldsc, svd_train_entities], axis=1)
#train_df = pd.concat([train_df,svd_train_desc, svd_train_lbldsc], axis=1)
test_df = pd.concat([test_df,svd_test_desc, svd_test_lbldsc, svd_test_entities], axis=1)
#test_df = pd.concat([test_df,svd_test_desc, svd_test_lbldsc], axis=1)

print(train_df.shape)
#print(train_df.columns.values)
print(test_df.shape)
#print(test_df.columns.values)
print("Done")


# In[ ]:


from sklearn.decomposition import NMF
from nltk.corpus import stopwords

def nmf_d(train, n_comp, out_cols):
    stop_words = set(stopwords.words('english'))
    tfv = TfidfVectorizer(min_df=3, max_features=10000,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = stop_words)

    # Fit TFIDF
    tfv.fit(list(train))
    X = tfv.transform(train)

    nmf = NMF(n_components=n_comp, random_state=1337)
    nmf.fit(X)
    #plog("nmf components/dictionary")
    #print(nmf.components_ )
    #print(svd.explained_variance_ratio_)
    X = nmf.transform(X)
    df_nmf = pd.DataFrame(X, columns=out_cols)
    
    return df_nmf

plog("NMF for train description started")
for c in Columns.desc_nmf_cols.value:
    if c in train_df.columns.values:
        train_df.drop([c], axis=1, inplace=True)
nmf_train_desc = nmf_d(train_df.Description, Columns.n_desc_nmf_comp.value, Columns.desc_nmf_cols.value)
plog("NMF for train description ended")
plog("NMF for test description started")
for c in Columns.desc_nmf_cols.value:
    if c in test_df.columns.values:
        test_df.drop([c], axis=1, inplace=True)
nmf_test_desc = nmf_d(test_df.Description, Columns.n_desc_nmf_comp.value, Columns.desc_nmf_cols.value)
plog("NMF for test description ended")

plog("NMF for train image label started")
for c in Columns.iann_nmf_cols.value:
    if c in train_df.columns.values:
        train_df.drop([c], axis=1, inplace=True)
nmf_train_lbldsc = nmf_d(train_df.Lbl_Dsc, Columns.n_iann_nmf_comp.value, Columns.iann_nmf_cols.value)
plog("NMF for train image label ended")
plog("NMF for test image label started")
for c in Columns.iann_nmf_cols.value:
    if c in test_df.columns.values:
        test_df.drop([c], axis=1, inplace=True)
nmf_test_lbldsc = tfidf_2(test_df.Lbl_Dsc, Columns.n_iann_nmf_comp.value, Columns.iann_nmf_cols.value)
plog("NMF for test image label ended")

plog("NMF for train desc entities started")
for c in Columns.entities_nmf_cols.value:
    if c in train_df.columns.values:
        train_df.drop([c], axis=1, inplace=True)
nmf_train_entities = nmf_d(train_df.Entities, Columns.n_entities_nmf_comp.value, Columns.entities_nmf_cols.value)
plog("NMF for train desc entities ended")
plog("NMF for test desc entities started")
for c in Columns.entities_nmf_cols.value:
    if c in test_df.columns.values:
        test_df.drop([c], axis=1, inplace=True)
nmf_test_entities = nmf_d(test_df.Entities, Columns.n_entities_nmf_comp.value, Columns.entities_nmf_cols.value)
plog("NMF for test desc entities ended")

train_df = pd.concat([train_df,nmf_train_desc, nmf_train_lbldsc, nmf_train_entities], axis=1)
#train_df = pd.concat([train_df,svd_train_desc, svd_train_lbldsc], axis=1)
test_df = pd.concat([test_df,nmf_test_desc, nmf_test_lbldsc, nmf_test_entities], axis=1)
#test_df = pd.concat([test_df,svd_test_desc, svd_test_lbldsc], axis=1)

train_df.drop(["Description", "Lbl_Dsc", "Entities", "Name"], axis=1, inplace=True)
test_df.drop(["Description", "Lbl_Dsc", "Entities", "Name"], axis=1, inplace=True)


for c in Columns.entities_nmf_cols.value + Columns.iann_nmf_cols.value + Columns.desc_nmf_cols.value:
    if c in train_df.columns.values:
        #train_df[c].fillna(0, inplace=True)
        pass
    if c in test_df.columns.values:
        #test_df[c].fillna(0, inplace=True)
        pass

print(train_df.shape)
#print(train_df.columns.values)
print(test_df.shape)
#print(test_df.columns.values)
print("Done")


# In[ ]:


#data augmentation
#this section needs to be refactored because of return samples of smoteenc, don't know if original dataset is included in returned samples


if 1 == 0:
    for c in train_df.drop(["AdoptionSpeed", "PetID", "RescuerID"], axis=1).columns.values.tolist():
        if train_df[c].isna().any():
            train_df[c].fillna(0, inplace=True)

    c_features = [train_df.columns.get_loc(c) for c in train_df.columns if c in Columns.ind_num_cat_columns.value +                       ["Age", "Fee", "VideoAmt", "PhotoAmt","Quantity"]]
    
    import random
    import string
    from imblearn.over_sampling import RandomOverSampler, SMOTENC
    print(len(train_df))
    plog("Generating mock data with smote")
    ros = SMOTENC(random_state=0, sampling_strategy='minority', categorical_features=c_features)
    train_df_c = train_df.copy()
    x_train_df_c = train_df_c.drop(["AdoptionSpeed", "PetID", "RescuerID"], axis=1)
    y_train_df_c = train_df_c["AdoptionSpeed"]
    x_res, y_res = ros.fit_resample(x_train_df_c, y_train_df_c)
    
    #filter on newly generated columns
    x_res = pd.DataFrame(data=x_res[len(x_train_df_c)+1:], columns=x_train_df_c.columns.values.tolist())
    y_res = pd.DataFrame(data=y_res[len(y_train_df_c)+1:], columns=["AdoptionSpeed"])
    
    plog(str(len(x_res)) + " lines of mock data has been generated")
    rnd_resc_arr = []

    for i in range (int(len(x_res)/4)):
        rnd_RescuerID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
        rnd_resc_arr.append(rnd_RescuerID)
    x_res["RescuerID"] = x_res.apply(lambda x : random.choice(rnd_resc_arr), axis=1)

    rnd_petID_arr = []
    for i in range (len(x_res)):
        rnd_petID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(9)])
        rnd_petID_arr.append(rnd_petID)
    x_res["PetID"] = np.asarray(rnd_petID_arr).reshape(-1,1)

    train_df_smote = pd.concat([x_res, y_res], axis=1, sort=False)
    print(len(train_df_smote))

    #print(x_res.shape, y_res.reshape(-1,1).shape)
    train_df = pd.concat([train_df, train_df_smote], axis=0 ,sort=False)
    print(len(train_df))
plog("Done")


# In[ ]:



plog("Sorting breed1 and breed2 cols before concatenation")
bm_df_tr = pd.DataFrame(np.sort(train_df[["Breed1", "Breed2"]], axis=1), train_df[["Breed1", "Breed2"]].index, train_df[["Breed1", "Breed2"]].columns)
bm_df_te = pd.DataFrame(np.sort(test_df[["Breed1", "Breed2"]], axis=1), test_df[["Breed1", "Breed2"]].index, test_df[["Breed1", "Breed2"]].columns)
plog("Sorted breed1 and breed2 cols before concatenation")


plog("Merging Breed1 and Breed2 on train")
train_df["Breed_Merge"] = bm_df_tr.apply(lambda x: "_".join([str(x['Breed1']), str(x['Breed2'])]), axis=1 )
plog("Merged Breed1 and Breed2 on train")
plog("Merging Breed1 and Breed2 on test")
test_df["Breed_Merge"] = bm_df_te.apply(lambda x: "_".join([str(x['Breed1']), str(x['Breed2'])]), axis=1 )
plog("Merged Breed1 and Breed2 on test")

#new feature listing purity of breed type
def set_pet_breed(b1, b2):
    #print(b1, b2)
    if (b1 in  (0, 307)) & (b2 in  (0, 307)):
        return 4
    elif (b1 ==  307) & (b2 not in  (0, 307)):
        return 3
    elif (b2 ==  307) & (b1 not in  (0, 307)):
        return 3
    elif (b1 not in  (0, 307)) & (b2 not in  (0, 307)) & (b1 != b2):
        return 2
    elif (b1 == 0) & (b2 not in  (0, 307)):
        return 1
    elif (b2 == 0) & (b1 not in  (0, 307)):
        return 1
    elif (b1 not in  (0, 307)) & (b2 not in  (0, 307)) & (b1 == b2):
        return 0
    else: 
        return 3
plog("Setting Pet Breed for test dataset")
test_df["Pet_Breed"] = test_df.apply(lambda x: set_pet_breed(x['Breed1'], x['Breed2']), axis=1)                
plog("Setted Pet Breed for test dataset")
plog("Setting Pet Breed for train dataset")
train_df["Pet_Breed"] = train_df.apply(lambda x: set_pet_breed(x['Breed1'], x['Breed2']), axis=1)
plog("Setted Pet Breed for train dataset")

def chVal(val, col):
    if (col == "Health") & (val == 0):
        return 4
    else:
        return val


plog("Setting Pet Breed for test dataset")
test_df["Pet_Purity"] = test_df.apply(lambda x: 1 if x["Pet_Breed"] in (0, 1) else  0, axis=1)                
plog("Setted Pet Breed for test dataset")
plog("Setting Pet Purity for train dataset")
train_df["Pet_Purity"] = train_df.apply(lambda x: 1 if x["Pet_Breed"] in (0, 1) else  0, axis=1)  
plog("Setted Pet Purity for train dataset")

plog("Setting Pet Breed for test dataset")
test_df["Overall_Status"] = test_df.apply(lambda x: math.sqrt(x["Sterilized"]) +                                                             math.sqrt(x["Dewormed"]) +                                                             math.sqrt(x["Vaccinated"]) +                                                             math.sqrt(chVal(x["Health"], "Health")), axis=1)
plog("Setted Pet Breed for test dataset")
plog("Setting Pet Purity for train dataset")
train_df["Overall_Status"] = train_df.apply(lambda x: math.sqrt(x["Sterilized"]) +                                                             math.sqrt(x["Dewormed"]) +                                                             math.sqrt(x["Vaccinated"]) +                                                             math.sqrt(chVal(x["Health"], "Health")), axis=1)  
plog("Setted Pet Purity for train dataset")



#print(train_df.columns.values)
#print(test_df.columns.values)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
def label_encoder(train, test, col):
    le = LabelEncoder()
    #print(pd.unique(train[col]))
    #print(pd.unique(test[col]))
    val_train = pd.unique(train[col])
    val_test = pd.unique(test[col])
    #print(val_train)
    #print(val_test)
    if np.array_equal(val_train, val_test):
        #print(1)
        vals = val_train
    else :
        #print(2)
        vals = np.unique(np.concatenate((val_train, val_test), axis=None))
    #print(vals)
    le.fit(vals)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    return train, test

if 1 == 1 :
    
    for c in ["State", "Breed_Merge"]:
        plog("Started " + c + " encoding")
        train_df, test_df = label_encoder(train_df, test_df, c)
        plog("Ended " + c + " encoding")


# In[ ]:


for c in Columns.barplot_cols.value:
    if c in train_df.columns.values.tolist():
        train_df[c].astype("int32")
    if c in test_df.columns.values.tolist():
        test_df[c].astype("int32")

for c in Columns.boxplot_cols.value:
    if c in train_df.columns.values.tolist():
        train_df[c].astype("float32")
    if c in test_df.columns.values.tolist():
        test_df[c].astype("float32")

plog("Done")


# In[ ]:


plog("Creating Fee_Per_Pet on train dataset")
train_df["Fee_Per_Pet"] = train_df["Fee"]/train_df["Quantity"]
plog("Created Fee_Per_Pet on train dataset")
plog("Creating Fee_Per_Pet on test dataset")
test_df["Fee_Per_Pet"] = test_df["Fee"]/test_df["Quantity"]
plog("Created Fee_Per_Pet on test dataset")


# In[ ]:


#new feature listing purity of breed type
def set_pet_maturity(typ, age):
    if typ == 1:
        if age <= 6:
            return 0
        elif age <= 12:
            return 1
        elif age <= 3*12:
            return 2
        elif age <= 6*12:
            return 3
        elif age <=10*12:
            return 4
        else:
            return 5
    if typ == 2 :
        if age <= 6:
            return 0
        elif age <= 12:
            return 1
        elif age <= 3*12:
            return 2
        elif age <= 6*12:
            return 3
        elif age <=10*12:
            return 4
        else:
            return 5

plog("Setting Pet_Maturity for test dataset")
test_df["Pet_Maturity"] = test_df.apply(lambda x: set_pet_maturity(x['Type'], x['Age']), axis=1)    
plog("Setted Pet_Maturitye for test dataset")
plog("Setting Pet_Maturity for train dataset")
train_df["Pet_Maturity"] = train_df.apply(lambda x: set_pet_maturity(x['Type'], x['Age']), axis=1)
plog("Setted Pet_Maturity for train dataset")


# In[ ]:


#new feature listing purity of breed type
def set_color_type(c1, c2, c3):
    if c1 != 0:
        c1 = 1
    if c2 != 0:
        c2 = 1
    if c3 != 0:
        c3 = 1
    return c1 + c2 + c3

plog("Setting Color Type for test dataset")
test_df["Color_Type"] = test_df.apply(lambda x: set_color_type(x['Color1'], x['Color2'], x['Color3']), axis=1) 
plog("Setted Color Type for test dataset")
plog("Setting Color Type for train dataset")
train_df["Color_Type"] = train_df.apply(lambda x: set_color_type(x['Color1'], x['Color2'], x['Color3']), axis=1)
plog("Setted Color Type for train dataset")


# In[ ]:


#feature engineering for each categorical variable based on number of items in each category 
def setItemCnt(val, c, agg):
    if agg == 1:
        if c in ["Color1", "Color2", "Color3"]:
            v1, v2, v3, v4, v5, v6, v7 = 100, 200, 350, 500, 1000, 1000, 1000
        elif c in ["RescuerID"]:
            v1, v2, v3, v4, v5, v6, v7 = 1, 5, 10, 20, 50, 100, 200
        elif c in ["Breed1", "Breed2", "Breed_Merge"]:
            v1, v2, v3, v4, v5, v6, v7 = 1, 10, 50, 100, 200, 500, 1000
        elif c in ["Pet_Breed"]:
            v1, v2, v3, v4, v5, v6, v7  = 1, 10, 50, 250, 1000, 1000, 1000
        else:
            v1, v2, v3, v4, v5, v6, v7  = 1, 5, 10, 20, 50, 100, 200
        if val > v7:
            return 7
        elif val > v6:
            return 6
        elif val > v5:
            return 5
        elif val > v4:
            return 4
        elif val > v3:
            return 3
        elif val > v2:
            return 2
        elif val > v1:
            return 1
        else:
            return 0
    else:
        return val

def itemCnt(df, col):
    train_r = df.groupby(col)["PetID"].count().reset_index()
    if col == "RescuerID":
        pass#print(train_r)
    train_r[col+"_Cnt"] = train_r.apply(lambda x: setItemCnt(x['PetID'], col, 0), axis=1)
    train_r[col+"_Cnt_MType"] = train_r.apply(lambda x: setItemCnt(x['PetID'], col, 1), axis=1)
    
    #train_r[col+"_Type"] = train_r['PetID']
    return train_r[[col, col+"_Cnt", col+"_Cnt_MType"]]

for c in Columns.item_cnt_cols.value:
    if c in train_df.columns.values:
        train_df.drop(c, axis=1, inplace=True)
    if c in test_df.columns.values:
        test_df.drop(c, axis=1, inplace=True)

for c in Columns.item_cnt_mtype_cols.value:
    if c in train_df.columns.values:
        train_df.drop(c, axis=1, inplace=True)
    if c in test_df.columns.values:
        test_df.drop(c, axis=1, inplace=True)

for c in Columns.item_cnt_incols.value:
    #if c in train_df.columns.values:
    #    plog("Deleting "+c)
    #    train_df.drop(c, axis=1, inplace=True)
    plog("Creating "+c+"_Cnt for train on "+ c)
    df_itr = itemCnt(train_df, c)
    train_df = train_df.set_index(c).join(df_itr.set_index(c)).reset_index()
    plog("Created "+c+"_Cnt for train on " + c)
    plog("Creating "+c+"_Cnt for test on "+ c)
    df_its = itemCnt(test_df, c)
    test_df = test_df.set_index(c).join(df_its.set_index(c)).reset_index()
    plog("Created "+c+"_Cnt for test on " + c)

def stdType(min, max, mean, std, value):
    if min <= value < mean - 5 * std:
        return 0
    elif  mean -5*std <= value < mean - 4*std:
        return 1
    elif mean -4*std <= value < mean - 3*std:
        return 2
    elif mean - 3*std <= value < mean - 2*std:
        return 3
    elif mean - 2*std <= value < mean - std:
        return 4
    elif mean - std <= value < mean:
        return 5
    elif mean <= value < mean + std:
        return 6
    elif mean + std <= value < mean + 2*std:
        return 7
    elif mean + 2*std <= value < mean + 3*std:
        return 8
    elif mean + 3*std <= value < mean + 4*std:
        return 9
    elif mean + 4*std <= value < mean + 5*std:
        return 10
    elif mean + 5*std <= value <= max:
        return 11
    
def setItemByStdType(df, col):
    mean = df[col].mean()
    std = df[col].std()
    min = df[col].min()
    max = df[col].max()
    df[col + "_StdType"] = df.apply(lambda x: stdType(min, max, mean, std, x[col]), axis=1)
    return df


for c in Columns.item_type_cols.value:
    if c in train_df.columns.values:
        train_df.drop(c, axis=1, inplace=True)
    if c in test_df.columns.values:
        test_df.drop(c, axis=1, inplace=True)

for c in Columns.item_type_incols.value:
    #if c in train_df.columns.values:
    #    plog("Deleting "+c)
    #    train_df.drop(c, axis=1, inplace=True)
    plog("Creating "+c+"_StdType for train on "+ c)
    train_df = setItemByStdType(train_df, c)
    plog("Created "+c+"_StdType for train on " + c)
    plog("Creating "+c+"_StdType for test on "+ c)
    test_df = setItemByStdType(test_df, c)
    plog("Created "+c+"_StdType for test on " + c)
    
#print(train_df.columns.values)
#print(test_df.columns.values)

plog("Done")


# In[ ]:



def set_mean_fee(df, col):
    dfm = df[[col, "Fee", "Quantity"]].groupby(col).agg({"Fee":"sum", "Quantity":"sum"}).reset_index()
    dfm["Fee_" + c] = dfm["Fee"] / dfm["Quantity"]
    dfm.drop(["Quantity","Fee"], axis = 1, inplace=True)
    df = df.set_index(col).join(dfm.set_index(col)).reset_index()
    return df

for c in Columns.fee_mean_incols.value:
    if "Fee_" + c in train_df.columns.values.tolist():
        plog("Dropping "+"Fee_Per_Pet_" + c + " on train dataset")
        train_df.drop("Fee_Per_Pet_" + c, axis=1, inplace=True)
    plog("Creating Fee_Per_Pet_" + c + " on train dataset")
    train_df = set_mean_fee(train_df, c)
    plog("Created Fee_Per_Pet_" + c +  " on train dataset")
    if "Fee_" + c in test_df.columns.values.tolist():
        plog("Dropping "+"Fee_Per_Pet_" + c + " on test dataset")
        test_df.drop("Fee_Per_Pet_" + c, axis=1, inplace=True)
    plog("Creating Fee_Per_Pet_" + c + " on test dataset")
    test_df = set_mean_fee(test_df, c)
    plog("Created Fee_Per_Pet_" + c +  " on test dataset")

#print(train_df.columns.values)
#print(test_df.columns.values)


# In[ ]:


from sklearn.preprocessing import KBinsDiscretizer

for c in Columns.kbin_cols.value:
    if c in train_df.columns.values.tolist():
        train_df.drop(c, axis=1, inplace = True)
    if c in test_df.columns.values.tolist():
        test_df.drop(c, axis=1, inplace = True)

for c in Columns.kbin_incols.value:
    if c in train_df.columns.values.tolist():
        kbin_est = KBinsDiscretizer(n_bins=12, encode='ordinal', strategy='quantile')
        kbin_est.fit(train_df[c].values.reshape(-1, 1))  
        kbin_train = kbin_est.transform(train_df[c].values.reshape(-1, 1))
        pd_kbin_train = pd.DataFrame(data=kbin_train, columns=[c+"_Kbin"])
        train_df = pd.concat([train_df,pd_kbin_train], axis=1)
        kbin_test = kbin_est.transform(test_df[c].values.reshape(-1, 1))
        pd_kbin_test = pd.DataFrame(data=kbin_test, columns=[c+"_Kbin"])
        test_df = pd.concat([test_df,pd_kbin_test], axis=1)
plog("Done")


# In[ ]:


#feature engineering for each categorical variable based on number mean adoption speed
def setItemAdp(val):
    if val > 3.5:
        return 4
    elif val > 2.5:
        return 3
    elif val > 1.5:
        return 2
    elif val > 0.5:
        return 1
    else:
        return 0

def itemAdp(df, col):
    train_r = df.groupby(col)["AdoptionSpeed"].mean().reset_index()
    train_r[col+"_Adp"] = train_r.apply(lambda x: setItemAdp(x['AdoptionSpeed']), axis=1)
    #train_r[col+"_Adp"] = train_r['AdoptionSpeed']
    return train_r[[col, col+"_Adp"]]

if 1 == 1:
    for c in Columns.item_adp_incols.value:
        if c+"_Adp" in train_df.columns.values:
            plog("Deleting train "+c+"_Adp")
            train_df.drop([c+"_Adp"], axis=1, inplace=True)
        plog("Creating "+c+"_Adp for train on "+ c)
        df_itr = itemAdp(train_df, c)
        train_df = train_df.set_index(c).join(df_itr.set_index(c)).reset_index()
        plog("Created "+c+"_Adp for train on " + c)
        if c+"_Adp" in test_df.columns.values:
            plog("Deleting test "+c+"_Adp")
            test_df.drop([c+"_Adp"], axis=1, inplace=True)
        plog("Creating "+c+"_Adp for test on "+ c)
        #df_its = itemAdp(test_df, c)
        #print(df_itr)
        test_df = test_df.set_index(c).join(df_itr.set_index(c)).reset_index()
        import math
        #test_df[c+"_Adp"].fillna(test_df[c+"_Adp"].mean(), inplace=True)
        plog("Created "+c+"_Adp for test on " + c)

    #print(train_df.columns.values)
    #print(train_df.head())
    #print(test_df.columns.values)
    #print(test_df.head())

plog("Done")


# In[ ]:


#drop rescuer id as no more needed
train_df.drop(["RescuerID"], axis=1, inplace=True)
test_df.drop(["RescuerID"], axis=1, inplace=True)
plog("Done")


# In[ ]:


#featureengineering using featuretools

import featuretools as ft
print(train_df.columns.values)
print("------------------------------------")
print(test_df.columns.values)
print("------------------------------------")
def auto_features(df, cols, entities):
    df_c = df[cols]
    es = ft.EntitySet(id='petfinder')
    es.entity_from_dataframe(entity_id="Pets", dataframe=df_c, index="PetID")
    ignored_variable =  {}
    ignored_variable.update({'Pets': entities})
    for e in entities:
        plog(e)

        es.normalize_entity(base_entity_id='Pets', new_entity_id=e, index=e)
        feature_matrix, feature_names = ft.dfs(entityset=es,
                                               target_entity=e,
                                               max_depth=2,
                                               verbose=1,
                                               #n_jobs=3,
                                               ignore_variables=ignored_variable,
                                               agg_primitives =Columns.agg_calc.value)
        fm = feature_matrix.add_prefix(e+"_")
        #print(fm.head())
        #print("--------", e)
        #print(df[e].head())
        df = df.set_index(e).join(fm).reset_index()

    return df

if 1 == 0:
    plog("creating new features on train using featuretools")
    for c in Columns.ft_cols.value:
        if c in train_df.columns.values:
            train_df.drop([c], axis=1, inplace=True)
    train_df = auto_features(train_df, 
                             Columns.iden_columns.value + Columns.ft_cat_cols.value + Columns.ft_new_cols.value, 
                             Columns.ft_cat_cols.value)
    plog("created new features on train using featuretools")
    for c in Columns.ft_cols.value:
        if c in test_df.columns.values:
            test_df.drop([c], axis=1, inplace=True)
    plog("creating new features on test using featuretools")
    test_df = auto_features(test_df, 
                            Columns.iden_columns.value + Columns.ft_cat_cols.value + Columns.ft_new_cols.value, 
                            Columns.ft_cat_cols.value)
    plog("created new features on test using featuretools")

    print(train_df.shape)
    #print(train_df.columns.values)
    print(test_df.shape)
    #print(test_df.columns.values)
plog("Done")


# In[ ]:



def auto_adp_features(train, test, cols, entities):

    df_c = train[cols]
    es = ft.EntitySet(id='petfinder')
    es.entity_from_dataframe(entity_id="Pets", dataframe=df_c, index="PetID")
    ignored_variable =  {}
    ignored_variable.update({'Pets': entities})
    for e in entities:
        plog(e)

        es.normalize_entity(base_entity_id='Pets', new_entity_id=e, index=e)
        feature_matrix, feature_names = ft.dfs(entityset=es,
                                               target_entity=e,
                                               max_depth=2,
                                               verbose=1,
                                               #n_jobs=3,
                                               ignore_variables=ignored_variable)
        fm = feature_matrix.add_prefix(e+"_")
        #print(feature_names)
        fm.drop([e+"_COUNT(Pets)"], axis = 1, inplace=True)
        train = train.set_index(e).join(fm).reset_index()
        test = test.set_index(e).join(fm).reset_index()

    return train, test

if 1 == 0:
    plog("creating adoptionspeed based new features on train and test using featuretools")
    train_df, test_df = auto_adp_features(train_df, test_df, 
                                          Columns.iden_columns.value + Columns.ft_cat_cols.value + Columns.item_cnt_mtype_cols.value +
                                          Columns.item_cnt_cols.value + Columns.item_type_cols.value + ["AdoptionSpeed"], 
                                          Columns.item_cnt_mtype_cols.value + Columns.item_cnt_cols.value + Columns.item_type_cols.value + Columns.ft_cat_cols.value)
    plog("Created adoptionspeed based new features on train and test using featuretools")

    print(train_df.shape)
    #print(train_df.columns.values)
    print(test_df.shape)
    #print(test_df.columns.values)
plog("Done")


# In[ ]:


#rermoving features with low variances
if 1 == 0:
    from sklearn.feature_selection import VarianceThreshold
    if "PetID" in train_df.columns.values:
        train_pet_id = train_df["PetID"]
        train_adoption_speed = train_df["AdoptionSpeed"]
        train_df.drop(["PetID", "AdoptionSpeed"], axis=1, inplace=True)

    if "PetID" in test_df.columns.values:
        test_pet_id = test_df["PetID"]
        test_df.drop(["PetID"], axis=1, inplace=True)

    def filter_by_varth(train, test, threshold):
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(train)
        indices = vt.get_support(indices=True)
        return(train.iloc[:,indices], test.iloc[:,indices])

    for c in train_df.columns.values:
        if train_df[c].isna().any():
            print("null", c)
            #train_df[c].fillna(0, inplace=True)

    plog("lists similarity check before from removal by variance")
    print(train_df.columns.values == train_df.columns.values)

    plog("train - test diff")
    print(list(set(train_df.columns.values.tolist()) - set(test_df.columns.values.tolist())))
    #print(train_df.columns.values.tolist())
    plog("test - train diff")
    list(set(test_df.columns.values.tolist()) - set(train_df.columns.values.tolist()))
    #print(test_df.columns.values.tolist())

    train_t, test_t = filter_by_varth(train_df, test_df, 0.20)
    plog("lists similarity check after from removal by variance")
    print(train_t.columns.values == test_t.columns.values)
    print(train_t.shape)
    print(test_t.shape)

    train_df = pd.concat([train_t, train_pet_id, train_adoption_speed], axis=1)
    test_df = pd.concat([test_t, test_pet_id], axis=1)
    print(train_df.shape)
    #print(train_df.columns.values)
    print(test_df.shape)
    #print(test_df.columns.values)
plog("Done")


# In[ ]:


def group_x_by_y(df, x, y, type):
    df_g = df.groupby(y).agg({x:type})
    df_g.rename({x: x+"_"+type+"_by_"+y}, axis='columns', inplace=True)
    df = df.set_index(y).join(df_g).reset_index()
    return df

plog("Done")


# In[ ]:


#check skewness of variable and make a logtransform

from scipy.stats import skewtest, normaltest
from sklearn.preprocessing import QuantileTransformer

rng = np.random.RandomState(304)
qt = QuantileTransformer(output_distribution='normal', random_state=rng)
if 1 == 0:
    for c in Columns.scaling_cols.value:
        if c in train_df.columns.values.tolist():
            p_val = normaltest(train_df[c])[1]

            if p_val < 0.05:
                plog("Transforming "+ c + " on train and test datasets with pval of normaltest:" + str(p_val))
                qt.fit(x_train[c].reshape(-1, 1)) 
                qt.transform(x_train[c].reshape(-1, 1))
                qt.transform(x_test[c].reshape(-1, 1))
            else:
                plog("Not transforming "+ c + " on train and test datasets with pval of normaltest:" + str(p_val))

plog("Done")
        


# In[ ]:


if 1 == 0:
    for c in Columns.scaling_cols.value:
        if c in train_df.columns.values.tolist():
            p_val = normaltest(train_df[c])[1]

            if p_val < 0.05:
                plog("Transforming "+ c + " on train and test datasets with pval of normaltest:" + str(p_val))
                #qt.fit(x_train[c].reshape(-1, 1)) 
                train_df[c] = np.log(train_df[c]+1)#qt.transform(x_train[c].reshape(-1, 1))
                test_df[c] = np.log(test_df[c]+1)#qt.transform(x_test[c].reshape(-1, 1))
            else:
                plog("Not transforming "+ c + " on train and test datasets with pval of normaltest:" + str(p_val))

plog("Done")


# In[ ]:


def normalize(train, test, col):
    mean = train[col].mean(axis=0)
    std = train[col].std(axis=0)
    train[col] = (train[col]-mean)/std
    test[col] = (test[col]-mean)/std
    return train, test

if 1 == 0:
    for c in Columns.scaling_cols.value:
        if c in train_df.columns.values.tolist():
            plog("Starting normalizing for " + c)
            train_df, test_df = scale_num_var(train_df, test_df, c)
            #print(train_df[col].head())
            #print(est_df[col].head())
            plog("Normalizing Ended for " + c)


# In[ ]:


from sklearn.preprocessing import RobustScaler

if 1 == 0:
    scaler = RobustScaler()
    plog("Starting RobustScaler for train")
    scaler.fit(train_df[Columns.scaling_cols.value])
    train_df[Columns.scaling_cols.value] = scaler.transform(train_df[Columns.scaling_cols.value])
    plog("Ended RobustScaler for train")
    
    scaler = RobustScaler()
    plog("Starting RobustScaler for test")
    test_df[Columns.scaling_cols.value] = scaler.transform(test_df[Columns.scaling_cols.value])
    plog("Ended RobustScaler for test")


# In[ ]:


#outlier detection
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from collections import Counter
def detect_outliers(f_train):
    #id = train["PetID"]
    #f_train = train.drop(["PetID"], axis=1)
    n_samples = len(f_train)
    outliers_fraction = 0.10
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # define outlier/anomaly detection methods to be compared
    anomaly_algorithms = [
        # ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                          gamma=0.1)),
        ("Isolation Forest", IsolationForest(behaviour='new',
                                             contamination=outliers_fraction,
                                             random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(
            n_neighbors=35, contamination=outliers_fraction))]

    # Define datasets
    #print(f_train.head())
    df_pred = pd.DataFrame(columns=["One-Class SVM","Isolation Forest","Local Outlier Factor"])
    if 1 == 1 :
        for name, algorithm in anomaly_algorithms:
            #t0 = time.time()
            # algorithm.fit(f_train)
            plog(name)
            #t1 = time.time()

            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(f_train)
            else:
                y_pred = algorithm.fit(f_train).predict(f_train)

            df_pred[name] = y_pred
            #print(df_pred)

    df_pred["outlier"] = df_pred.apply(lambda x: Counter([x['One-Class SVM'], x['Isolation Forest'], x["Local Outlier Factor"]]).most_common(1)[0][0], axis=1)
    #print(df_pred.shape)
    #prediction_df = pd.concat([id,df_pred], axis=1)

    # create submission file print(prediction_df)
    #df_pred.to_csv("outliers.csv")
    return df_pred["outlier"]

#print(train_df.head())
o_cols = train_df.columns.values.tolist().copy()

low_imp_features = []

for c in ["PetID", "RescuerID"]+low_imp_features: #Columns.ind_num_cat_columns.value + ["AdoptionSpeed","PetID"] + low_imp_features :
     if c in o_cols:
        o_cols.remove(c)
#print(train_df[o_cols].shape)

for c in o_cols:
    if train_df[c].isna().any():
        plog("train null value exist for column " + c)
        train_df[c].fillna(0, inplace=True)#train_df[c].mean(), inplace=True)
    if c != "AdoptionSpeed":
        if test_df[c].isna().any():
            plog("test null value exist for column " + c)
            test_df[c].fillna(0, inplace=True)#test_df[c].mean(), inplace=True)

#train_df[o_cols] = train_df[o_cols].apply(lambda x: x.fillna(0),axis=0)
#print(train_df[o_cols].shape)
#test_df[o_cols] = test_df[o_cols].apply(lambda x: x.fillna(0),axis=0)

#train_df[col].fillna(train_df[col].mean(), inplace=True)
#test_df[col].fillna(test_df[col].mean(), inplace=True)
    
print(o_cols)

plog("Outlier detection started")

call=1
if call ==1:
    df_o = detect_outliers(train_df[o_cols])
    print("df_o", df_o.shape, "train_df", train_df.shape)
    train_df = pd.concat([train_df, df_o], axis=1)
else:
    plog("For Dogs")
    train_df_dogs=train_df[train_df["Type"]==1].reset_index().drop(["index"] ,axis=1)
    df_od = detect_outliers(train_df_dogs[o_cols+["AdoptionSpeed"]])
    print("df_od", df_od.shape, "train_df_dogs", train_df_dogs.shape)
    train_df_dogs = pd.concat([train_df_dogs,df_od], axis=1)
    print("train_df_dogs", train_df_dogs.shape)
    plog("For Cats")
    train_df_cats=train_df[train_df["Type"]==2].reset_index().drop(["index"] ,axis=1)
    df_oc = detect_outliers(train_df_cats[o_cols+["AdoptionSpeed"]])
    print("df_oc", df_oc.shape, "train_df_cats", train_df_cats.shape)
    train_df_cats = pd.concat([train_df_cats, df_oc], axis=1)
    print("train_df_cats", train_df_cats.shape)
    plog("Concatenating dogs and cats datasets")
    train_df=pd.concat([train_df_dogs,train_df_cats])
plog("Outlier detection ended")
#train_df = train_df[train_df["outlier"]==1]
#train_df["outlier"].fillna(1, inplace=True)
print("train_df", train_df.shape)
#print(train_df.columns.values)
print("test_df", test_df.shape)
#print(test_df.columns.values)
plog("Done")


# In[ ]:


#preparing final datasets by excluding outliers
def prepare_data(train_df, test_df):
    
    #train_x = train_df[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value 
    #                  + Columns.desc_svd_cols.value + Columns.img_num_cols_1.value 
    #                  + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_svd_cols.value 
    #                  + Columns.ft_cols.value +  Columns.item_cnt_cols.value]
    train_x = train_df.drop(["AdoptionSpeed", "outlier", "PetID"], axis=1)
    train_y = train_df[Columns.dep_columns.value]
    #test_x = test_df[Columns.ind_cont_columns.value + Columns.ind_num_cat_columns.value 
    #                  + Columns.desc_svd_cols.value + Columns.img_num_cols_1.value 
    #                  + Columns.img_num_cols_2.value + Columns.img_num_cols_3.value + Columns.iann_svd_cols.value
    #                  + Columns.ft_cols.value + Columns.item_cnt_cols.value]
    test_x = test_df.drop(["PetID"], axis=1)
    test_id = test_df[Columns.iden_columns.value]
    
    return train_x, train_y, test_x, test_id

# -1 means with outliers 1 means without outliers
outlier = 1
#test without columns having text values (name and description) and rescuer_id
x_train, y_train, x_test, id_test = prepare_data(train_df[train_df["outlier"]>=outlier], test_df)
x_train_dogs, y_train_dogs, x_test_dogs, id_test_dogs = prepare_data(train_df[(train_df["Type"]==1)&(train_df["outlier"]>=outlier)], 
                                                                     test_df[test_df["Type"]==1])

x_train_cats, y_train_cats, x_test_cats, id_test_cats = prepare_data(train_df[(train_df["Type"]==2)&(train_df["outlier"]>=outlier)], 
                                                                     test_df[test_df["Type"]==2])

#tr_idx_dogs = x_train.index[x_train["Type"] == 1].tolist()
#tr_idx_cats = x_train.index[x_train["Type"] == 2].tolist()
#x_train_dogs = x_train.iloc[tr_idx_dogs].reset_index()
#y_train_dogs = y_train.iloc[tr_idx_dogs].reset_index()
#x_train_cats = x_train.iloc[tr_idx_cats].reset_index()
#y_train_cats = y_train.iloc[tr_idx_cats].reset_index()

#te_idx_dogs = x_test.index[x_test["Type"] == 1].tolist()
#te_idx_cats = x_test.index[x_test["Type"] == 2].tolist()
#x_test_dogs = x_test.iloc[te_idx_dogs].reset_index()
#id_test_dogs = id_test.iloc[te_idx_dogs].reset_index()
#x_test_cats = x_test.iloc[te_idx_cats].reset_index()
#id_test_cats = id_test.iloc[te_idx_cats].reset_index()
plog("x_train information")
print(x_train.shape)
print(x_train.columns.values)
plog("y_train information")
print(y_train.shape)
print(y_train.columns.values)
plog("x_test information")
print(x_test.shape)
print(x_test.columns.values)
plog("id_test information")
print(id_test.shape)
print(id_test.columns.values)
pd.concat([x_train, y_train], axis=1).to_csv("train.csv", index=False)
pd.concat([id_test, x_test], axis=1).to_csv("test.csv", index=False)

plog("Done")


# In[ ]:


if 1 == 0:
    print(x_train.head())
    print("---------------------")
    print(x_test.head())


# In[ ]:


import matplotlib as mpl

def check_cat_perc(arr, cols):
    for col in cols:
        if col != "RescuerID":
            fig, axes = plt.subplots(2, 1, figsize=[16, 16])
            # axes[0,0].set_title("Perc. of " + col + " dist.")
            # axes[0,1].set_title("Perc. of " + col + " dist. by AdopSp.")
            # axes[1,0].set_title("Perc. of " + col + " dist. by AdopSp. for Dogs")
            # axes[1,1].set_title("Perc. of " + col + " dist. by AdopSp. for Cats")
            df1 = arr[col].value_counts(normalize=True).rename("percentage").mul(100).reset_index()  # .sort_values(col)
            df1.rename(columns={"index": col}, inplace=True)
            ax1 = sns.catplot(x=col, y="percentage", data=df1, kind="bar", ax=axes[0])
            ax1.set_axis_labels("All "+col, "Percentage")
            ax1.set_xticklabels(rotation=90)
            df2 = arr.groupby([col])["AdoptionSpeed"].value_counts(normalize=True).rename('percentage').mul(100).reset_index()
            ax2 = sns.catplot(x=col, y="percentage", data=df2, hue="AdoptionSpeed", kind="bar", ax=axes[1])
            ax2.set_axis_labels("All "+col, "Percentage")
            ax2.set_xticklabels(rotation=90)
            plt.close(2)
            plt.close(3)
            plt.show()
if 1 == 0:
    cols = []

    for c in Columns.barplot_cols.value:
        if c in x_train.columns.values.tolist():
            cols.append(c)

    check_cat_perc(pd.concat([x_train, y_train], axis=1, sort=False), cols)


# In[ ]:


def check_num_dist(arr, cols):
    for col in cols:
        print(col)
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        f, axes = plt.subplots(3, 2, figsize=[25, 25])
        f.suptitle(col + ' histogram')
        sns.distplot(arr[col], ax=axes[0, 0], axlabel="All")
        sns.distplot(arr[arr["AdoptionSpeed"] == 0][col], ax=axes[0, 1], axlabel="All for Adoption Speed 0")
        sns.distplot(arr[arr["AdoptionSpeed"] == 1][col], ax=axes[1, 0], axlabel="All for Adoption Speed 1")
        sns.distplot(arr[arr["AdoptionSpeed"] == 2][col], ax=axes[1, 1], axlabel="All for Adoption Speed 2")
        sns.distplot(arr[arr["AdoptionSpeed"] == 3][col], ax=axes[2, 0], axlabel="All for Adoption Speed 3")
        sns.distplot(arr[arr["AdoptionSpeed"] == 4][col], ax=axes[2, 1], axlabel="All for Adoption Speed 4")

        for i in range(2, 6):
            plt.close(i)

        plt.show()

if 1 == 0:
    #print(Columns.ind_cont_columns.value)
    cols = []

    for c in Columns.barplot_cols.value + Columns.boxplot_cols.value:
        if c in x_train.columns.values.tolist():
            cols.append(c)

    check_num_dist(pd.concat([x_train, y_train], axis = 1), cols)


# In[ ]:


from pylab import rcParams
def num_boxp(arr, cols):
    #print(arr.describe())
    for col in cols:
        print(col)
        #mpl.rcParams['axes.labelsize'] = 10
        #mpl.rcParams['xtick.labelsize'] = 10
        #mpl.rcParams['ytick.labelsize'] = 10
         #f, axes = plt.subplots(1, 3, figsize=[20, 9])
         #f.suptitle(col +' by AdopSp.')
         #ax1 = sns.boxplot(y=col, x="AdoptionSpeed", data=arr, ax=axes[0])
         #ax2 = sns.boxplot(y=col, x="AdoptionSpeed", data=arr[arr["Type"] == 1], ax=axes[1])
         #ax3 = sns.boxplot(y=col, x="AdoptionSpeed", data=arr[arr["Type"] == 2], ax=axes[2])
         #axes[0].set_title("For All")
         #axes[1].set_title("For Dogs")
         #axes[2].set_title("For Cats")
        a_0 = arr[arr["AdoptionSpeed"] == 0][col]
        a_1 = arr[arr["AdoptionSpeed"] == 1][col]
        a_2 = arr[arr["AdoptionSpeed"] == 2][col]
        a_3 = arr[arr["AdoptionSpeed"] == 3][col]
        a_4 = arr[arr["AdoptionSpeed"] == 4][col]
        d_0 = arr[(arr["Type"] == 1) & (arr["AdoptionSpeed"] == 0)][col]
        d_1 = arr[(arr["Type"] == 1) & (arr["AdoptionSpeed"] == 1)][col]
        d_2 = arr[(arr["Type"] == 1) & (arr["AdoptionSpeed"] == 2)][col]
        d_3 = arr[(arr["Type"] == 1) & (arr["AdoptionSpeed"] == 3)][col]
        d_4 = arr[(arr["Type"] == 1) & (arr["AdoptionSpeed"] == 4)][col]
        c_0 = arr[(arr["Type"] == 2) & (arr["AdoptionSpeed"] == 0)][col]
        c_1 = arr[(arr["Type"] == 2) & (arr["AdoptionSpeed"] == 1)][col]
        c_2 = arr[(arr["Type"] == 2) & (arr["AdoptionSpeed"] == 2)][col]
        c_3 = arr[(arr["Type"] == 2) & (arr["AdoptionSpeed"] == 3)][col]
        c_4 = arr[(arr["Type"] == 2) & (arr["AdoptionSpeed"] == 4)][col]
        #data = [arr[col], arr[arr["Type"] == 1][col], arr[arr["Type"] == 2][col]]
        data = [a_0, a_1, a_2, a_3, a_4, d_0, d_1, d_2, d_3, d_4, c_0, c_1, d_2, c_3, c_4]
        rcParams['figure.figsize'] = [20, 10]
        fig7, ax7 = plt.subplots()
        
        ax7.set_title('Boxplot of ' + col + ' of All and by Type')
        #sns.boxplot(hue="AdoptionSpeed", y=col, x="Type", data=arr )
        ax7.boxplot(data)
        
        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
                   ['All 0', 'All 1', 'All 2', 'All 3', 'All 4', 'Dogs 0', 'Dogs 1', 'Dogs 2', 'Dogs 3', 'Dogs 4', 
                    'Cats 0', 'Cats 1', 'Cats 2', 'Cats 3', 'Cats 4'])
        #for i in range(2,5):
        #    plt.close(i)
        plt.show()
if 1 == 0:
    cols = []

    for c in Columns.boxplot_cols.value:
        if c in x_train.columns.values.tolist():
            cols.append(c)

    num_boxp(pd.concat([x_train, y_train], axis = 1), cols)


# In[ ]:


# This function tests the dependence between two categorical variables
from collections import Counter
import math
from scipy.stats import entropy

def conditional_entropy(x,y):
    # for categorical correlation
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy

def theils_u(x, y):
    # for categorical correlation
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def by_theilsu(train,cols):
    # for categorical correlation
    theilu = pd.DataFrame(index=cols, columns=cols)
    i = 0
    for c1 in cols:
        for c2 in cols:
            #u = theils_u(train[c1].tolist(), train[c2].tolist())
            u = theils_u(train[c1], train[c2])
            theilu.loc[c1, c2] = u
        i = i+1
    theilu.fillna(value=np.nan, inplace=True)
    #plt.figure(figsize=(20, 1))
    return theilu

#print(Columns.ind_num_cat_columns.value)
cols = []
            
for c in Columns.ind_num_cat_columns.value:
    if c in x_train.columns.values.tolist():
        cols.append(c)
df_a = pd.concat([x_train[cols],y_train],axis=1)
#print(df_a.describe(include="all"))
print("Done")


# In[ ]:


if 1 == 0:
    print("Effect of x on y by theils, ex: Effect of Breed1 on Type is 0.99")
    theils_a = by_theilsu(df_a,cols+ Columns.dep_columns.value)
    theilu_d = by_theilsu(df_a[df_a["Type"]==1],cols+Columns.dep_columns.value)
    theilu_c = by_theilsu(df_a[df_a["Type"]==2],cols+Columns.dep_columns.value)
    theils_all = pd.concat([theils_a.loc["AdoptionSpeed",:], theilu_d.loc["AdoptionSpeed",:], theilu_c.loc["AdoptionSpeed",:]], axis=1)
    #print(theils_all)
    theils_all = pd.DataFrame(index=(["All", "Dogs", "Cats"]), data=(theils_all.values.T), columns=theils_a.columns.values)
    plt.rcParams["figure.figsize"] = [20,5]
    ax = plt.axes()
    sns.heatmap(theils_all, annot=True, fmt='.4f', ax=ax)
    ax.set_title("Entrophy based categorical to categorical dependent correlation for all types")
    plt.show()


# In[ ]:


if 1 == 0:
    print("Effect of x on y by theils, ex: Effect of Breed1 on Type is 0.99")
    theils_a = by_theilsu(df_a,cols+ Columns.dep_columns.value)
    plt.rcParams["figure.figsize"] = [14,14]
    ax = plt.axes()
    sns.heatmap(theils_a, annot=True, fmt='.2f', ax=ax)
    ax.set_title("Entrophy based categorical to categorical dependent correlation for all types")
    plt.show()


# In[ ]:


if 1 == 0:
    theilu_d = by_theilsu(df_a[df_a["Type"]==1],cols+Columns.dep_columns.value)
    plt.rcParams["figure.figsize"] = [14,14]
    ax = plt.axes()
    sns.heatmap(theilu_d, annot=True, fmt='.2f', ax=ax)
    ax.set_title("Entrophy based categorical independent to categorical dependent correlation for dogs")
    plt.show()


# In[ ]:


if 1 == 0:
    theilu_c = by_theilsu(df_a[df_a["Type"]==2],cols+Columns.dep_columns.value)
    plt.rcParams["figure.figsize"] = [14,14]
    ax = plt.axes()
    sns.heatmap(theilu_c, annot=True, fmt='.2f', ax=ax)
    ax.set_title("Entrophy based categorical to categorical dependent correlation for cats")
    plt.show()


# In[ ]:


# this function tests the dependency between  independent categorical variables  and dependent continous/ordinal variable
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    #print(cat_num)
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta


def by_correlation_ratio(df, dep_col, cols):
    # for numerical and categorical mix correlation
    cr = pd.DataFrame(columns=cols)
    i = 0
    for c in cols:
        eta = correlation_ratio(df[dep_col], df[c])
        #print(i,c)
        cr.loc[i, c] = eta
    cr.fillna(value=np.nan, inplace=True)
    # plt.figure(figsize=(20, 1))
    return cr

print("Done")


# In[ ]:


# tests the dependency between  independent categorical variables  and dependent continous/ordinal variable
if 1 == 0:
    cols = []

    for c in Columns.ind_num_cat_columns.value:
        if c in x_train.columns.values.tolist():
            cols.append(c)

    cr = by_correlation_ratio(pd.concat([x_train, y_train] ,axis=1).reset_index(), Columns.dep_columns.value[0], cols)
    cr_dogs = by_correlation_ratio(pd.concat([x_train_dogs, y_train_dogs] ,axis=1).reset_index(), Columns.dep_columns.value[0], cols)
    cr_cats = by_correlation_ratio(pd.concat([x_train_cats, y_train_cats] ,axis=1).reset_index(), Columns.dep_columns.value[0], cols)

    cr_all = pd.concat([cr, cr_dogs, cr_cats])
    df_cr_all = pd.DataFrame(index=(["All", "Dogs", "Cats"]), data=(cr_all.values), columns=cr_all.columns.values)
    plt.rcParams["figure.figsize"] = [20, 5]
    ax = plt.axes()

    sns.heatmap(df_cr_all, annot=True, fmt='.4f', ax=ax)
    ax.set_title('Correlations between independent categorical and dependant ordinal/continous variable')
    plt.show()
    


# In[ ]:


#pandas corr with spearman to test continous/ordinal independent and ordinal dependent value correlation
cols = []
            
    
for c in Columns.boxplot_cols.value + Columns.item_type_cols.value + Columns.kbin_cols.value + Columns.item_cnt_mtype_cols.value:
    if c in x_train.columns.values.tolist():
        cols.append(c)

corr = pd.concat([x_train[cols], y_train], axis=1)
#print(corr)
corr_dogs = pd.concat([x_train_dogs[cols], y_train_dogs], axis=1)
corr_cats = pd.concat([x_train_cats[cols], y_train_cats], axis=1)
corr_all = pd.concat([corr.corr('spearman').loc["AdoptionSpeed",:], corr_dogs.corr('spearman').loc["AdoptionSpeed",:], corr_cats.corr('spearman').loc["AdoptionSpeed",:]], axis=1)
df_corr_all = pd.DataFrame(columns=(["All", "Dogs", "Cats"]), data=(corr_all.values), index=corr_all.index.values)
plt.rcParams["figure.figsize"] = [10, 40]
ax = plt.axes()
sns.heatmap(df_corr_all.sort_values(by=['All']), annot=True, fmt='.4f', ax=ax)
ax.set_title('Dataset correlation between independent continous/ordinal and dependent ordinal variables')
plt.show()

df_corr_all.sort_values(by=["All"], ascending=False).to_csv("spearman_for_continous.csv")


# In[ ]:


from scipy.stats import kruskal
import sys
def by_kruskal(arr, dep_col):
#check adoption speed median for different categories, each category value should have at least 5 measurements
#The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal.
    cols = arr.columns.values.tolist()
    cols.remove(dep_col)
    df =  pd.DataFrame(columns=cols)
    
    for c in cols:
        #print("-------Kruskal Wallis H-test test for :--------", c, "on", dep_col)
        arg = []
        for v in arr[c].unique():
            #print("number of measurements of value", v, "on column", c, "is", len(arr[arr[c] == v]))
            if (len(arr[arr[c] == v]) >= 5):
                #print(arr[arr[dep_col] == v][c].head())
                arg.append(arr[arr[c] == v][dep_col])
        #print("number of categorical groups having more than 5 ameasurements found is ", len(arg), "/",len(arr[c].unique()) )
        if len(arg)>=2:
            #try:
            H, pval = kruskal(*arg)
            #print(c, "H-statistic:", H, "P-Value:", pval)

            if pval <= 0.01:
                #plog("Reject NULL hypothesis - Significant differences exist between groups.")
                df.loc[0, c] = pval
            if pval > 0.01:
                plog("Accept NULL hypothesis - No significant difference between groups for "+ c + " within 99% of significance level with p-value " + str(pval))
                df.loc[0, c] = pval

            #except:
                #plog("Test Error")
                #print(c, sys.exc_info())
                #df.loc[0, c] = 0

        else:
            plog("Not Tested Because of lack of at least 2 groupes having minimum 5 measurements")
            df.loc[0, c] = 0

    return df


# In[ ]:


#Exucuting kruskal test by constucting adoption speed samples over categorical variable values 
cols = []
for c in Columns.barplot_cols.value + Columns.dep_columns.value:
    if c in x_train.columns.values.tolist():
        cols.append(c)
        
df = by_kruskal(pd.concat([x_train, y_train], axis=1)[cols+Columns.dep_columns.value], Columns.dep_columns.value[0])
if 1 == 1:
    plt.rcParams["figure.figsize"] = [15, 2]
    sns.heatmap(df[df.columns].astype(float), annot=True)
    plt.show()


# In[ ]:


#Exucuting kruskal test by constucting numerical columns samples using adoption speed as categories. This time executing kruskal on inverse
cols = []
for c in Columns.boxplot_cols.value + Columns.item_type_cols.value + Columns.kbin_cols.value + Columns.item_cnt_mtype_cols.value:
    if c in x_train.columns.values.tolist():
        cols.append(c)

df_a = pd.DataFrame(columns=["AdoptionSpeed"])
for c in cols :
    print(c)
    df = by_kruskal(pd.concat([x_train, y_train], axis=1)[[c] + Columns.dep_columns.value], c)
    df["Variable"] = c
    df_a = pd.concat([df_a, df])
df_a = df_a.reset_index().drop("index", axis=1)
df_a.set_index("Variable", inplace=True)
print(df_a[df_a.AdoptionSpeed.isnull()])
#df_a["AdoptionSpeed"].fillna(2, inplace=True)
plt.rcParams["figure.figsize"] = [15, 100]
sns.heatmap(df_a.astype(float).sort_values(by=['AdoptionSpeed']), annot=True)
plt.show()


# In[ ]:


#dropping columns not having any statistical difference
if 1 == 0:
    df_a2 = df_a.reset_index()
    d_cols=""
    for c in df_a2[df_a2["AdoptionSpeed"]==0]["Variable"].values.tolist():
        x_train.drop(c, axis=1, inplace=True)
        x_test.drop(c, axis=1, inplace=True)
        d_cols = c + ", " + d_cols

    plog("Dropped " + d_cols)


# In[ ]:


if 1 == 0:
    #check correlation for continous independent variables and dependent ordinal variable
    cols = []

    for c in Columns.boxplot_cols.value + Columns.item_type_cols.value + Columns.kbin_cols.value + Columns.item_cnt_mtype_cols.value:
        if c in x_train.columns.values.tolist():
            cols.append(c)

    cr = by_correlation_ratio(pd.concat([x_train, y_train] ,axis=1).reset_index(), Columns.dep_columns.value[0], cols)
    cr = cr.T
    cr_dogs = by_correlation_ratio(pd.concat([x_train_dogs, y_train_dogs] ,axis=1).reset_index(), Columns.dep_columns.value[0], cols)
    cr_dogs = cr_dogs.T
    cr_cats = by_correlation_ratio(pd.concat([x_train_cats, y_train_cats] ,axis=1).reset_index(), Columns.dep_columns.value[0], cols)
    cr_cats = cr_cats.T

    cr_all = pd.concat([cr, cr_dogs, cr_cats], axis=1)
    df_cr_all = pd.DataFrame(data=(cr_all.values), index=cr_all.index)
    df_cr_all.columns = ["All", "Dogs", "Cats"]
    ax = plt.axes()
    plt.rcParams["figure.figsize"] = [5, 100]
    sns.heatmap(df_cr_all, annot=True, fmt='.4f', ax=ax)
    ax.set_title('Correlations between independent continous variables and dependant ordinal variable(supposed as continous)')
    plt.show()

    if 1 == 0:
        #listing cols where corr <= 0.0003
        cols = []
        for index, row in df_cr_all.iterrows():
            if row["All"] <= 0.0003:
                if index in Columns.boxplot_cols.value + Columns.item_type_cols.value + Columns.kbin_cols.value + Columns.item_cnt_mtype_cols.value:
                    if index in x_train.columns.values.tolist():
                        cols.append(index)
        print(cols)

        #dropping cols where corr <= 0.01

        x_train.drop(cols, axis=1, inplace=True)
        x_test.drop(cols, axis=1, inplace=True)


# In[ ]:



import traceback
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectFpr, SelectFwe


result = chi2(x_train[Columns.ind_num_cat_columns.value], y_train["AdoptionSpeed"].values.reshape(-1, 1))

chi2_pvals = []
for x in result[1]:
    chi2_pvals.append(x)
df_chi2_pvals = pd.DataFrame(index=Columns.ind_num_cat_columns.value, columns=["AdoptionSpeed"], data = chi2_pvals).sort_values(by=["AdoptionSpeed"], ascending=False)
df_chi2_pvals.to_csv("chitest_for_categorical.csv")
    
if 1 == 1:    
    x_train_fs = x_train.copy()
    
    mic = mutual_info_classif(x_train_fs.astype("float"), y_train, random_state=42)
    df_mic = pd.DataFrame(data=mic, index=x_train.columns.values.tolist(), columns=["importance"])
    print(df_mic)
    print(df_mic.sort_values(by=["importance"]))
    df_mic.sort_values(by=["importance"]).to_csv("df_mic.csv")
    
    try:
        x_fpr = SelectFpr(mutual_info_classif, alpha=0.03).fit_transform(x_train_fs.astype("float"), y_train)
        df_x_fpr = pd.DataFrame(data=x_fpr, index=x_train.columns.values.tolist(), columns=["importance"])
        print(df_x_fpr)
        print(df_x_fpr.sort_values(by=["importance"]))
        df_x_fpr.sort_values(by=["importance"]).to_csv("x_fpr.csv")
    except Exception as e:
        print(e)
        traceback.print_exc()
        
    try:
        x_fdr = SelectFdr(mutual_info_classif, alpha=0.03).fit_transform(x_train_fs.astype("float"), y_train)
        df_x_fdr = pd.DataFrame(data=x_fdr, index=x_train.columns.values.tolist(), columns=["importance"])
        print(df_x_fdr)
        print(df_x_fdr.sort_values(by=["importance"]))
        df_x_fdr.sort_values(by=["importance"]).to_csv("x_fdr.csv")
    except Exception as e:
        print(e)
        traceback.print_exc()
    
    try:
        x_fwe = SelectFwe(mutual_info_classif, alpha=0.03).fit_transform(x_train_fs.astype("float"), y_train)
        print(x_fwe)
        df_x_fwe = pd.DataFrame(data=x_fwe, index=x_train.columns.values.tolist(), columns=["importance"])
        print(df_x_fwe)
        print(df_x_fwe.sort_values(by=["importance"]))
        df_x_fwe.sort_values(by=["importance"]).to_csv("x_fwe.csv")
    except Exception as e:
        print(e)
        traceback.print_exc()


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from time import time
from sklearn.pipeline import Pipeline
from sklearn.metrics import cohen_kappa_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

def report(df, alg, best_est, perf, est, results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            #print("Model with rank: {0}".format(i))
            #print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            #      results['mean_test_score'][candidate],
            #      results['std_test_score'][candidate]))
            #print("Parameters: {0}".format(results['params'][candidate]))
            #print("")
            df.loc[len(df)] = [alg, best_est, perf, est, format(i), results['mean_test_score'][candidate],results['std_test_score'][candidate],results['params'][candidate]]
    return df

layer_sizes = []
for i in range(0,100):
    t = (np.random.random_integers(25,300), np.random.random_integers(25,300), np.random.random_integers(25,300))
    layer_sizes.append(t)

def iterate_by_randomsearch(train_x, train_y):
    classifiers = [
        (AdaBoostClassifier(), {"n_estimators": sp.stats.randint(25, 100),
                                 'learning_rate': sp.stats.uniform(0.0001, 1)}),
        # (BaggingClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        #                        "max_features": sp.stats.randint(1, 7),
        #                        "bootstrap": [True, False],
        #                        "bootstrap_features": [True, False],
        #                        }),
        # (ExtraTreesClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        #                           "max_depth": sp.stats.randint(3, 30),
        #                           "max_features": sp.stats.randint(1, 7),
        #                           "min_samples_split": sp.stats.randint(2, 11),
        #                           "bootstrap": [True, False],
        #                           "criterion": ["gini", "entropy"]}),
        # (RandomForestClassifier(), {"n_estimators": sp.stats.randint(25, 100),
        #                             "max_depth": sp.stats.randint(3, 30),
        #                             "max_features": sp.stats.randint(1, 7),
        #                             "min_samples_split": sp.stats.randint(2, 11),
        #                             "bootstrap": [True, False],
        #                             "criterion": ["gini", "entropy"]}),
        # (PassiveAggressiveClassifier(), {"max_iter": sp.stats.randint(0, 1230),
        #                                  "tol": sp.stats.uniform(0.0001, 0.05)}),
        # (RidgeClassifier(), {"max_iter": sp.stats.randint(0, 2000),
        #                      "tol": sp.stats.uniform(0.0001, 0.05),
        #                      "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}),
        # (SGDClassifier(), {"max_iter": sp.stats.randint(0, 2000),
        #                    "tol": sp.stats.uniform(0.0001, 0.05),
        #                    "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        #                    "penalty": ["none", "l2", "l1", "elasticnet"]}),
        # (KNeighborsClassifier(), {"n_neighbors": sp.stats.randint(1, 50),
        #                           "algorithm": ["ball_tree", "kd_tree", "brute"],
        #                           "leaf_size": sp.stats.randint(20, 100),
        #                           "p": [1, 2]}),
        # (DecisionTreeClassifier(), {"max_depth": sp.stats.randint(3, 10),
        #                             "max_features": sp.stats.randint(1, 7),
        #                             "min_samples_split": sp.stats.randint(2, 11),
        #                             "criterion": ["gini", "entropy"]}),
        # (QuadraticDiscriminantAnalysis(), {"tol": sp.stats.uniform(1e-5, 1e-2)}),
        # (LogisticRegression(), {"multi_class":["ovr", "multinomial", "auto"],
        #                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        #                        "penalty":["l2"],
        #                        "class_weight":["balanced"],
        #                        "max_iter": sp.stats.randint(100,300),
        #                        "n_jobs":[4]}),
        #(MLPClassifier(), {"hidden_layer_sizes":[(np.random.random_integers(25, 200), np.random.random_integers(25, 200), np.random.random_integers(25, 200))]}),
        # (xgb.XGBClassifier(), {"n_estimators": sp.stats.randint(25, 200),
        #                             "max_depth": sp.stats.randint(3, 30)}),
         (lgb.LGBMClassifier(), {'num_leaves': sp.stats.randint(25, 330),
                                 'n_estimators': sp.stats.randint(25, 150),
                              #'bagging_fraction': sp.stats.uniform(0.4, 0.9),
                               'learning_rate': sp.stats.uniform(0.001, 0.5),
                              #'min_data': sp.stats.randint(50,700),
                              #'is_unbalance': [True, False],
                              #'max_bin': sp.stats.randint(3,25),
                              'boosting_type' : ['gbdt', 'dart'],
                              #'bagging_freq': sp.stats.randint(3,35),
                               'max_depth': sp.stats.randint(3,30),
                                'min_split_gain': sp.stats.uniform(0.001, 0.5),
                               'objective': 'multiclass',
                                 "n_jobs":[4]} )
    ]
    print(len(train_x), len(train_y))
    df = pd.DataFrame(columns=['alg', 'best_estimator', 'perf', 'est','rank','mean','std', 'parameters'])
    
    for clf in classifiers:
        n_iter=10
        kappa_scorer = make_scorer(cohen_kappa_score, weights="quadratic")
        kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        cvs = kfold.split(train_x, train_y)
        print(cvs)
        random_search = RandomizedSearchCV(clf[0], param_distributions=clf[1], verbose=8,
                                            cv=cvs, n_iter=10, scoring=kappa_scorer, n_jobs = 4)
        
        print(type(clf[0]).__name__, "started at", datetime.now())
        start = time()
        random_search.fit(train_x, train_y)
        df = report(df, type(clf[0]).__name__, random_search.best_estimator_ ,  time() - start, n_iter, random_search.cv_results_)
        print(type(clf[0]).__name__, "ended at", datetime.now())
    best = df['mean'].idxmax()

    est = df.loc[best, ["best_estimator", "mean"]]
    i = 1
    clfs = []
    means = []
    clf_p=""
    dt=np.dtype('str,float')
    for idx, clf in df.sort_values(by="mean", ascending=False).iterrows():
        #if (clf_p != type(clf["best_estimator"]).__name__) & (i<6):
        if i<6:
            clfs.append((str(i)+type(clf["best_estimator"]).__name__, clf["best_estimator"]))
            means.append(clf["mean"])
            clf_p = type(clf["best_estimator"]).__name__ 
        i = i+1
        #clf_p = type(clf["best_estimator"]).__name__
    return clfs, means

print("Done")


# In[ ]:


def voting_predict(clfs, means, train_x, train_y, test_x, test_id):

    clf = VotingClassifier(estimators=clfs, voting = 'soft')
    clf.fit(train_x, train_y.values.ravel())
    pred = clf.predict(test_x)
    #print(test_id.shape, pred.shape)
    prediction_df = pd.DataFrame({'PetID': test_id.values.ravel(),
                                  'AdoptionSpeed': pred})

    # create submission file print(prediction_df)
    return prediction_df

def voting_predict_with_weights(clfs, means, train_x, train_y, test_x, test_id):

    clf = VotingClassifier(estimators=clfs, weights=means, voting = 'soft')
    clf.fit(train_x, train_y.values.ravel())
    
    pred = clf.predict(test_x)
    #print(test_id.shape, pred.shape)
    prediction_df = pd.DataFrame({'PetID': test_id.values.ravel(),
                                  'AdoptionSpeed': pred})

    # create submission file print(prediction_df)
    return prediction_df
print("Done")


# In[ ]:


if 1 == 0:
    
    clfs, means = iterate_by_randomsearch(x_train, y_train.values.ravel())
    print("--------clfs--------")
    print(clfs)
    print("--------means--------")
    print(means)
    pred = voting_predict(clfs, means, x_train, y_train,  x_test, id_test)
    #clfs = iterate_by_randomsearch(x_train, y_train.values.ravel())
    #voting_predict(clfs, x_train, y_train,  x_test, id_test)
    #pred.to_csv("submission.csv",index=False)
    class_sub = pred
print("Done")


# In[ ]:


if 1 == 0:
    
    x_train_a =  x_train
    x_test_a = x_test
    clfs, means = iterate_by_randomsearch(x_train_a, y_train.values.ravel())
    print(clfs)
    print(means)
    pred = voting_predict(clfs, means, x_train_a, y_train,  x_test_a, id_test)
    #clfs = iterate_by_randomsearch(x_train, y_train.values.ravel())
    #voting_predict(clfs, x_train, y_train,  x_test, id_test)
    pred.to_csv("submission.csv",index=False)
print("Done")


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from functools import partial
import lightgbm
from math import sqrt
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):
        x = a - min_rating
        y = b - min_rating
        if not np.isscalar(x):
            x=x[0]
        if not np.isscalar(y):
            y=y[0]
        #print(x,y)
        conf_mat[x][y] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        x = r - min_rating
        if not np.isscalar(x):
            x = x[0]
        hist_ratings[x] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p) #cohen_kappa_score(y, X_p, labels=[0,1,2,3,4], weights="quadratic")
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        #print(5)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        #print(initial_coef)
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        print(self.coef_)

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

def run_cv_model(train, test, target, model_fn, cat_features, params={}, eval_fn=None, label='model', n_splits=5, n_repeats=2):
    kf = RepeatedStratifiedKFold(n_splits=n_splits, random_state=42, n_repeats = n_repeats)
    fold_splits = kf.split(train, target)
    cv_scores = []
    qwk_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0], n_splits*n_repeats))
    all_coefficients = np.zeros((n_splits*n_repeats, 4))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        #print('Started ' + label + ' fold ' + str(i) + '/'+str(n_splits*n_repeats))
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
        if isinstance(target, pd.DataFrame):
            dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]
        else:
            dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, importances, coefficients, qwk = model_fn(dev_X, dev_y, val_X, val_y, test, params2, cat_features)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        all_coefficients[i-1, :] = coefficients
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            qwk_scores.append(qwk)
            #print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))
        fold_importance_df = pd.DataFrame()
        if isinstance(train, pd.DataFrame):
            fold_importance_df['feature'] = train.columns.values
        else:
            fold_importance_df['feature'] = ["pca_"+ str(i) for i in range(train.shape[1])]
        fold_importance_df['importance'] = importances
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        i += 1
    #print('{} cv RMSE scores : {}'.format(label, cv_scores))
    #print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))
    #print('{} cv std RMSE score : {}'.format(label, np.mean(cv_scores)))
    #print('{} cv QWK scores : {}'.format(label, qwk_scores))
    #print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
    #print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
    pred_full_test = pred_full_test / (n_splits*n_repeats)
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
                'cv': cv_scores, 'qwk': qwk_scores,
               'importance': feature_importance_df,
               'coefficients': all_coefficients}
    return results

def runLGB(train_X, train_y, test_X, test_y, test_X2, params, cat_features):
    #print('Prep LGB')
    
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    #print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    if len(cat_features) > 0:
        model = lgb.train(params,
                          categorical_feature=list(cat_features),
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)
    else:
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)
    #print('Predict 1/2')
    #plog("--------feature names--------")
    #print(model.feature_name())
    #plog("--------feature importances by split--------")
    #print(lightgbm.plot_importance(booster=model))
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients)
    #print("Valid Counts = ", Counter(test_y))
    #print("Predicted Counts = ", Counter(pred_test_y_k))
    #print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(test_y, pred_test_y_k) #cohen_kappa_score(test_y, pred_test_y_k, labels=[0,1,2,3,4], weights="quadratic")
    #print("QWK = ", qwk)
    #print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importance(importance_type="gain"), coefficients, qwk

plog("Done")


# In[ ]:


def by_regressor_rs(train, test, y_train, runALG, metric, name, cv, i, id_test):
    mqwk = 0
    j = 0
    for j in range(10):
        params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': random.randint(50, 300),
              'max_depth': random.randint(5, 25),
              'learning_rate': random.uniform(0.001, 0.05),
              'bagging_fraction': random.uniform(0.5, 1),
              'feature_fraction': random.uniform(0.5, 1),
              'min_split_gain': random.uniform(0.001, 0.05),
              'min_child_samples': random.randint(75, 200),
              'min_child_weight': random.uniform(0.001, 0.05),
              'verbosity': -1,
              'data_random_seed': 3,
              'early_stop': 100,
              'verbose_eval': False,
              'n_jobs':4,
              'lambda_l2': random.uniform(0.001, 0.1),
              'num_rounds': 10000}
        print("RS prms", params)
        results_t = run_cv_model(train, test, y_train, runALG, params, metric, name, cv, i)
        print("RS QWK Scores", results_t["qwk"], "rs mean qwk scores", np.mean(results_t["qwk"]), "rs mean rmse scores" , np.mean(results_t["cv"]))
        if np.mean(results_t["qwk"]) > mqwk:
            results = results_t
            mqwk = np.mean(results_t["qwk"])
    optR = OptimizedRounder()
    coefficients_ = np.mean(results['coefficients'], axis=0)
    print(coefficients_)
    train_predictions = [r[0] for r in results['train']]
    train_predictions = optR.predict(train_predictions, coefficients_).astype(int)
    Counter(train_predictions)

    optR = OptimizedRounder()
    test_predictions = [r[0] for r in results['test']]
    test_predictions = optR.predict(test_predictions, coefficients_).astype(int)
    Counter(test_predictions)

    pd.DataFrame(sk_cmatrix(y_train, train_predictions), index=list(range(5)), columns=list(range(5)))
    submission = pd.DataFrame({'PetID': id_test.PetID.values, 'AdoptionSpeed': test_predictions})
    return submission


# In[ ]:



def by_regressor(train, test, y_train, runALG, prms, metric, name, cv, i, id_test, pca, c_features):
    
    results = run_cv_model(train, test, y_train, runALG, c_features, prms, metric, name, cv, i)
    print("RS QWK Scores", results["qwk"], "rs mean qwk scores", np.mean(results["qwk"]), "rs mean rmse scores" , np.mean(results["cv"]))
    grp = results["importance"].groupby('feature',as_index=False)['importance'].mean()
    print(grp.head())
    grp2 = grp.sort_values(by=["importance"])
    grp2.plot(kind="barh", y="importance", x="feature")
    grp2.to_csv("feature_importances.csv")
    optR = OptimizedRounder()
    coefficients_ = np.mean(results['coefficients'], axis=0)
    print(coefficients_)
    train_predictions = [r[0] for r in results['train']]
    train_predictions = optR.predict(train_predictions, coefficients_).astype(int)
    Counter(train_predictions)

    optR = OptimizedRounder()
    test_predictions = [r[0] for r in results['test']]
    test_predictions = optR.predict(test_predictions, coefficients_).astype(int)
    Counter(test_predictions)

    pd.DataFrame(sk_cmatrix(y_train, train_predictions), index=list(range(5)), columns=list(range(5)))
    submission = pd.DataFrame({'PetID': id_test.PetID.values, 'AdoptionSpeed': test_predictions})
    return submission
plog("Done")


# In[ ]:


if 1 == 1:
    import warnings
    warnings.filterwarnings("ignore")
    params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'max_depth': 11, 
          'num_leaves': 350,
          'learning_rate': 0.01,
          'bagging_fraction': 0.85,
          'feature_fraction': 0.8,
          'min_split_gain': 0.01,          
          'min_child_samples': 75,
          'min_child_weight': 0.1,
          'verbosity': -1,
          'data_random_seed': 3,
          'early_stop': 100,
          'verbose_eval': False,
          'n_jobs':4,
          #'lambda_l2': 0.05,
          'num_rounds': 10000, }
    
    pca = 0
    if pca == 1:
        from sklearn.decomposition import KernelPCA
        kPCA = KernelPCA(n_components=150, kernel='rbf')
        kPCA.fit(x_train)
        x_train = kPCA.transform(x_train)
        x_test = kPCA.transform(x_test)
    
    x_train_a = x_train.drop(low_imp_features, axis=1)
    x_test_a = x_test.drop(low_imp_features, axis=1)
    
    if pca == 0:
        cat_features = [x_train_a.columns.get_loc(c) for c in x_train_a.columns if c in Columns.ind_num_cat_columns.value]
    else:
        cat_features = []
    
    plog("listing categorical features")
    print(cat_features)
    #print(df.iloc[cat_features].columns[(df.iloc[cat_features] < 0).any()])
    
    if call==1:
        submission = by_regressor(x_train_a, x_test_a, y_train, runLGB, params, rmse, 'lgb', 5, 2, id_test, pca ,cat_features)
        reg_sub = submission
        submission.to_csv('submission.csv', index=False)
    else:
        x_train_dogs.drop(["Type"], axis=1, inplace=True)
        x_test_dogs.drop(["Type"], axis=1, inplace=True)
        submission_d = by_regressor(x_train_dogs, x_test_dogs, y_train_dogs, runLGB, params, rmse, 'lgb', 5, 2, id_test_dogs)
        x_train_cats.drop(["Type"], axis=1, inplace=True)
        x_test_cats.drop(["Type"], axis=1, inplace=True)
        submission_c = by_regressor(x_train_cats, x_test_cats, y_train_cats, runLGB, params, rmse, 'lgb', 5, 2, id_test_cats)
        submission = pd.concat([submission_d, submission_c])
        submission.to_csv('submission.csv', index=False)


# In[ ]:


if 1 == 0:
    x_train_a =  x_train.drop(low_imp_features, axis=1) 
    x_test_a = x_test.drop(low_imp_features, axis=1)
    
    submission = by_regressor_rs(x_train_a, x_test_a, y_train, runLGB, rmse, 'lgb', 5, 2, id_test)
    #submission.to_csv('submission.csv', index=False)


# In[ ]:


from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
import numpy as np
import joblib
import pickle
from sklearn.metrics import cohen_kappa_score
import pathlib

class RatioOrdinalClassfier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, estimator=lgb.LGBMClassifier()):
        """
        Called when initializing the classifier
        """
        # Parameters should have same name as attributes
        self.estimator = estimator
        self.estimators_ = []

    def encode_classes(self, y, yi):
        if y[0] > yi:
            return 1
        else:
            return 0

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self.sorted_classes_ = np.unique(y)
        #print(self.sorted_classes_)
        self.num_classes_ = len(self.sorted_classes_)
        self.num_instances_ = len(X)
        self.probas_ = np.zeros((self.num_instances_, 1))

        for yi in self.sorted_classes_[:-1]:
            #print(yi)
            yt = y.copy()
            yt[yt <= yi] = 0
            yt[yt > yi] = 1
            est = self.estimator
            est.fit(X, yt.ravel())
            filename = "ownestimatormodel_"+str(yi)+".sav"
            with open(filename, 'wb') as file:
                pickle.dump(est, file)
            with open(filename ,'rb') as f:
                est = pickle.load(f)
            self.estimators_.append(est)
        return self

    def predict_proba(self, X):
        i = 0
        for yi in self.sorted_classes_[:-1]:
            filename = "ownestimatormodel_"+str(yi)+".sav"
            with open(filename ,'rb') as f:
                est = pickle.load(f)
            #est = self.estimators_[yi]
            yt_proba = est.predict_proba(X)[:, 1:2]
            if i == 0:
                # print(self.probas_.shape)
                self.probas_ = yt_proba
                # print(self.probas_.shape)
            else:
                # print(yt_proba.shape, self.probas_.shape)
                self.probas_ = np.concatenate((self.probas_, yt_proba), axis=1)
                # print(yt_proba.shape, self.probas_.shape)
            i += 1

    def predict(self, X):
        self.predict_proba(X)
        ypf = np.zeros((self.num_instances_,1))
        try:
            getattr(self, "probas_")
            for i in range(self.num_classes_):
                if i == 0:
                    ypi = 1 - self.probas_[:, 0:1]
                elif 0 < i < self.num_classes_-1:
                    ypi = self.probas_[:, i-1:i] - self.probas_[:, i:i+1]
                elif i == self.num_classes_-1:
                    ypi = self.probas_[:, i-1:i]
                if i == 0:
                    ypf = ypi
                else:
                    ypf = np.concatenate((ypf,ypi), axis=1)
            #print(ypf)
            return np.argmax(ypf, axis=1).reshape(len(ypf), 1)

        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
if 1 == 0:    
    params = {
              'boosting_type': 'gbdt',
              'metric': 'cohen_kappa_score',
              'max_depth': 9,
              'num_leaves': 70,
              'learning_rate': 0.01,
              'bagging_fraction': 0.85,
              'feature_fraction': 0.8,
              'min_split_gain': 0.01,
              'min_child_samples': 50,
              'min_child_weight': 0.1,
              'verbosity': -1,
              'data_random_seed': 3,
              'n_jobs': 4,
              'objective':'binary',
              # 'lambda_l2': 0.05,
              }
    est = lgb.LGBMClassifier(**params)
    clf = RatioOrdinalClassfier(estimator=est)
    clf.fit(x_train.values,y_train.values)
    pred_y = clf.predict(x_train.values)
    print(cohen_kappa_score(y_train, pred_y))
    submission_p = clf.predict(x_test.values)
    print(type(id_test), type(id_test.values), type(submission_p))
    submission_df = pd.DataFrame(data=submission_p, columns=["AdoptionSpeed"])
    submission = pd.concat([id_test, submission_df], axis=1)
    submission.to_csv('submission.csv', index=False)


# In[ ]:


if 1==0:
    import math
    merge_sub = (class_sub["AdoptionSpeed"].astype('int32') + reg_sub["AdoptionSpeed"].astype('int32'))/2
    submission_df = pd.DataFrame(data=merge_sub, columns=["AdoptionSpeed"])
    
    submission_df["AdoptionSpeed"] = submission_df.apply(lambda x: math.floor(x['AdoptionSpeed']), axis=1 )
    submission = pd.concat([id_test, submission_df], axis=1)
    submission.to_csv('submission.csv', index=False)


# In[ ]:


if 1 == 0 :
    from keras import models, layers, regularizers
    from keras.utils.np_utils import to_categorical

    import numpy as np
    import pandas as pd
    from datetime import datetime
    from sklearn.preprocessing import OneHotEncoder

    x_train_k = x_train.copy()
    x_test_k = x_test.copy()
    id_test_k = id_test.copy()

    #y_train_k = to_categorical(y_train_k.values)
    
    y_train_k = np.zeros((len(y_train), 5))
    i = 0
    for x in y_train["AdoptionSpeed"].values:
        if x == 1:
            y_train_k[i,2] = 1
        elif x == 2:
            y_train_k[i, 1:3] = 1
        elif x == 3:
            y_train_k[i, 1:4] = 1
        elif x == 4:
            y_train_k[i, 1:5] = 1
        i += 1
    print(y_train_k )



    train_cat = np.array([])
    test_cat = np.array([])
    for c in Columns.ind_num_cat_columns.value:
        val_cats = np.unique(x_train_k[c].values.tolist() + x_test_k[c].values.tolist())

        ohe = OneHotEncoder(categories=[val_cats], sparse=False)

        print(datetime.now(), c)
        print(x_train_k[c].values.shape)
        x_train_k[c] = x_train_k[c].astype('int32')
        t_cat = ohe.fit_transform(x_train_k[c].values.reshape(-1, 1))
        if train_cat.size == 0:
            train_cat = t_cat
        else:
            train_cat = np.concatenate((train_cat, t_cat), axis=1)
        x_train_k.drop(c, inplace=True, axis=1)

        x_test_k[c] = x_test_k[c].astype('int32')
        te_cat = ohe.fit_transform(x_test_k[c].values.reshape(-1, 1))
        if test_cat.size == 0:
            test_cat = te_cat
        else:
            test_cat = np.concatenate((test_cat, te_cat), axis=1)
        x_test_k.drop(c, inplace=True, axis=1)

        print(train_cat.shape, test_cat.shape)

    print(datetime.now(), "x_train and x_test to ndarray")
    x_train_k = np.array(x_train_k.values)
    x_test_k = np.array(x_test_k.values)

    print(datetime.now(), "concatenate x_train/x_test and train_cat/test_cat")
    x_train_k = np.concatenate((x_train_k, train_cat), axis=1)
    x_test_k = np.concatenate((x_test_k, test_cat), axis=1)

    print(datetime.now(), x_train_k.shape, y_train_k.shape)

    from sklearn.model_selection import train_test_split

    partial_x_train, x_val, partial_y_train, y_val = train_test_split(x_train_k, y_train_k, test_size=0.20, random_state=42, stratify=y_train_k)

    model = models.Sequential()
    model.add(layers.Dense(int(partial_x_train), kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(partial_x_train.shape[1],)))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(int(85), kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    history = model.fit(partial_x_train, partial_y_train, epochs=100,
                                batch_size=256, validation_data=(x_val, y_val))

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss)+1)
    plt.rcParams["figure.figsize"] = [20, 10]
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()

    pred_labels = model.predict_classes(x_val)
    true_labels = np.argmax(y_val, axis=1)

    print(cohen_kappa_score(pred_labels,true_labels ))


    pred = model.predict_classes(x_test_k)
    id_test_k = np.array(id_test_k.values)
    print(id_test_k.shape, pred.shape)
    submission = np.concatenate((id_test_k, pred.reshape(-1,1)), axis=1)
    print(submission.shape)
    print(submission)
    submission_df = pd.DataFrame(columns=["PetID, AdoptionSpeed"], data=submission)
    print(submission_df.groupby(['AdoptionSpeed']).size())
    #submission_df.to_csv('submission.csv', index=False)


