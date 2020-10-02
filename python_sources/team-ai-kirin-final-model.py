#*************************************************************************************************
#                 National Data Science Challenge 2019- Beginner Category                        *
#                                   Team AI Kirin                                                *
#           Team Members: Cheng Zhe, Derecho Keandre Caballes, Leo Sai Mun, Wang Wei             *
#                                    23 March 2019                                               *
#*************************************************************************************************

num_folds=20    #set StratifiedKFold to 20 folds
Submission_Version= 'Team_AI_Kirin_Submission.csv'

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB #for debugging purpose
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import *
from sklearn import metrics, linear_model
from sklearn.multiclass import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import  StratifiedKFold
import time
 
t0 = time.time()
t = time.time()

#****************Reading Raw Data******************************************************************

file  = '../input/train.csv'
file2 = '../input/test.csv'

train_df=pd.read_csv(file)
test_df=pd.read_csv(file2)
df_combined= pd.concat([train_df,test_df])
print ("Done reading csv, Time elapse: ", round((time.time()-t),3), "secs")
t = time.time()

#***********Seperate data by Product Group (beauty, fashion, mobile) for further processing*******

beauty_df=df_combined[df_combined['image_path'].str.contains("beauty")]
beauty_distinct_words=' '.join(beauty_df['title']).split()
print ("Total Distinct Words in beauty_df Before Feature Engineering: ", len(set(beauty_distinct_words)))

fashion_df=df_combined[df_combined['image_path'].str.contains("fashion")]
fashion_distinct_words=' '.join(fashion_df['title']).split()
print ("Total Distinct Words in fashion_df Before Feature Engineering: ", len(set(fashion_distinct_words)))
    
mobile_df=df_combined[df_combined['image_path'].str.contains("mobile")]
mobile_distinct_words=' '.join(mobile_df['title']).split()
print ("Total Distinct Words in mobile_df Before Feature Engineering: ", len(set(mobile_distinct_words)))

b_len=len(beauty_df)
f_len=len(fashion_df)
m_len=len(mobile_df)
print("beauty_df rows: ",b_len, "fashion_df rows: ",f_len, "mobile_df rows: ",m_len)   

#***************Defining Words to Delete********************************************************
# All words will be deleted when calling basic_data_cleaning_function (df, words_to_del)

mobile_words_to_del  = 'dan dengan untuk bayar di tempat'  
beauty_words_to_del  = 'dan dengan untuk bayar di tempat'
fashion_words_to_del = 'dan dengan untuk bayar di tempat'

#************** Defining Words to distill *****************************************************
# All words in each product group will be stemmed to words inside the listv[product group]_words_to_distill 
#when calling additional_data_cleaning_function (df, distinct_words, words_to_distill )

mobile_words_to_distill= ['iphone', 'samsung', 'sony', 'xiaomi', 'blackberry', 'lenovo', 'nokia', 'brandcode',
                          'infinix','oppo', 'vivo', 'evercoss', 'advan', 'mito', 'huawei', 'sharp', 'motorola', 
                          'strawberry', 'realme', 'icherry', 'smartfren', 'honor', 'alcatel', 'maxtron', 
                          'spc', 'oneplus'  ]
fashion_words_to_distill=[]
beauty_words_to_distill=[]

#****************Defining Words to Engineer***************************************************
# When calling feature_engineering_function (df, words_to_engineer), the first word of each sublist
# will be replaced by the second word of that sublist inside [product group]_words_to_engineer

mobile_words_to_engineer=[]

beauty_words_to_engineer=[
 ['highlighting', 'highlighter'], ['blusher', 'blush'], ['blush on', 'blushon'], ['stik', 'stick'],
 ['sticks', 'stick'], ['lip stick', 'lipstick'], ['lipstik', 'lipstick'], ['lipstikq', 'lipstick'],
 ['lipsticks', 'lipstick'], ['eye liner', 'eyeliner'], ['lip liner', 'lipliner']    
                                ]
fashion_words_to_engineer=[  
 ['3 4','3_4'] ,  ['aksen pita', 'aksen_pita'], ['aksen ruffle', 'aksen_ruffle'], ['atasan blus', 'atasan_blus'],
 ['bra crop', 'bra_crop'], ['baju atasan', 'baju_atasan'], ['atasan kemeja', 'atasan_kemeja'],
 ['baju pesta', 'baju_pesta'], ['bahan denim', 'bahan_denim'], ['bahan katun', 'bahan_katun'], 
 ['bahan lace', 'bahan_lace'], ['bahan linen', 'bahan_linen'],  ['bahan polyester', 'bahan_polyester'], 
 ['bahan rajut', 'bahan_rajut'], ['bahan sifon', 'bahan_sifon'], ['bahan velvet', 'bahan_velvet'],    
 ['bandage bodycon slim', 'bandage_bodycon_slim'] , ['bandage bodycon', 'bandage_bodycon'], 
 ['bodycon bandage', 'bandage_bodycon'], ['blouse atasan', 'blouse_atasan'], ['blus crop top', 'blus_crop_top'], 
 ['bralette crop', 'bralette_crop'], ['brokat midi', 'brokat_midi'], ['brokat terbatas', 'brokat_terbatas'],
 ['brokat terlaris', 'brokat_terlaris'],  ['brukat brokat lace', 'brukat_brokat_lace'], 
 ['brokat lace', 'brokat_lace'],['brukat brokat', 'brukat_brokat'],  
 ['crop top tanpa lengan', 'crop_top_tanpa_lengan'], ['crop top', 'crop_top'], ['cotton combed', 'cotton_combed'],
 ['chiffon dress', 'chiffon_dress'],  ['cold shoulder', 'cold_shoulder'], ['cover up', 'cover_up'], 
 ['cover ups', 'cover_up'],   ['kerah crew neck', 'kerah_crew_neck'], ['crew neck', 'crew_neck'], 
 ['dress bodycon pensil' ,'dress_bodycon_pensil'], ['dress bodycon', 'dress_bodycon'], 
 ['bodycon dress', 'dress_bodycon'],  ['dress bodycon mini', 'bodycon_mini_dress'], 
 ['bodycon mini dress', 'bodycon_mini_dress'], ['bodycon casual', 'bodycon_casual'], 
 ['dress casual', 'dress_casual'], ['dress cotton skirt', 'dress_cotton_skirt'], ['dress cotton', 'dress_cotton'],
 ['dress kaos', 'dress_kaos'], ['dress model', 'dress_model'], ['dress mini', 'mini_dress'], 
 ['mini dress', 'mini_dress'],  ['dress midi', 'dress_midi'], ['midi dress', 'dress_midi'],  ['dress wanita', 'dress_wanita'],
 ['dress pesta', 'dress_pesta'],['dress sexy', 'dress_sexy'],  ['dress skater', 'dress_skater'],
 ['dress maxi', 'dress_maxi'],  ['long dress', 'long_dress'],  ['dress casual', 'dress_casual'],
 ['beach dress', 'beach_dress'],   ['dress denim wanita', 'dress_denim_wanita'], ['denim dress', 'denim_dress'],
 ['long dress lace', 'long_dress_lace'], ['dress lace', 'dress_lace'],  ['dress off shoulder', 'dress_off_shoulder'],
 ['dress office', 'dress_office'],  ['dress swing', 'dress_swing'], ['dress fashion', 'dress_fashion'], 
 ['dress kemeja', 'dress_kemeja'], ['dress fit and flare', 'dress_fit_and_flare'], 
 ['dress fit flare', 'dress_fit_flare'], ['dress linen', 'dress_linen'],  ['dress longgar', 'dress_longgar'],    
 ['dress sifon longgar', 'dress_sifon_longgar'], ['dress sifon', 'dress_sifon'], 
 ['dress selutut', 'dress_selutut'],  ['dress slim', 'dress_slim'],  ['dress sweater', 'dress_sweater'],
 ['dress pantai', 'dress_pantai'],  ['dress pendek', 'dress_pendek'],  ['dress pensil', 'dress_pensil'],
 ['dress tunik', 'dress_tunik'],  ['dress vintage', 'dress_vintage'],    
 ['deep neck', 'deep_neck'], ['v neck' , 'v_neck'], ['neck v' , 'v_neck'],
 ['dress round neck', 'dress_round_neck'],['round neck', 'round_neck'],
 ['o neck', 'o_neck'], ['boat neck', 'boat_neck'], ['desain patchwork', 'desain_patchwork'],     
 ['gaya eropa amerika', 'gaya_eropa_amerika'],  ['eropa amerika', 'eropa_amerika'], 
 ['evening cocktail', 'evening_cocktail'], ['evening party', 'evening_party'],    
 ['fashion wanita', 'fashion_women'], ['fashion women', 'fashion_women'],  ['flower print', 'flower_print'],
 ['fit and flare', 'fit_and_flare'],  ['fg dress', 'fg_dress'],  ['flare sleeve', 'flare_sleeve'],
 ['floral dress', 'floral_dress'], ['floral lace', 'floral_lace'], ['floral printed', 'floral_printed'],
 ['bergaya boho', 'bergaya_boho'], ['bergaya elegan', 'bergaya_elegan'], ['gaya korea', 'gaya_korea'], 
 ['bergaya korea' , 'gaya_korea'],  ['bergaya retro', 'bergaya_retro'],
 ['bergaya sexy', 'bergaya_sexy'], ['bergaya vintage', 'bergaya_vintage'],  ['gaya retro', 'gaya_retro'],
 ['gaun midi bodycon', 'gaun_midi_bodycon'], ['gaun midi', 'gaun_midi'], 
 ['gaun pesta', 'gaun_pesta'], ['gaun wanita', 'gaun_wanita'], 
 ['gaun maxi bodycon', 'gaun_maxi_bodycon'], ['gaun maxi', 'gaun_maxi'],    
 ['hem asimetris', 'hem_asimetris'], ['high waist', 'high_waist'], 
 ['lengan panjang', 'lengan_panjang'],  ['tanpa lengan', 'tanpa_lengan'],  ['lengan pendek', 'lengan_pendek'],
 ['long sleeve', 'long_sleeve'], ['short sleeve', 'short_sleeve'],
 ['model longgar', 'model_longgar'],  ['model backless', 'model_backless'],    
 ['motif bunga', 'motif_bunga'],  ['motif floral', 'motif_floral'] ,  ['motif print', 'motif_print'], 
 ['motif garis', 'motif_garis'], ['musim panas', 'musim_panas'],  ['musim gugur', 'musim_gugur'], 
 ['musim semi', 'musim_semi'],  ['autumn winter', 'autumn_winter'], ['summer autumn', 'summer_autumn'],
 ['off shoulder', 'off_shoulder'], ['pesta cocktail malam', 'pesta_cocktail_malam'],
 ['pesta cocktail club', 'pesta_cocktail_club'], ['pesta cocktail', 'pesta_cocktail'], ['pesta malam', 'pesta_malam'], 
 ['cocktail club', 'cocktail_club'],  ['cocktail party', 'cocktail_party'],   
 ['potongan longgar', 'potongan_longgar'],
 ['party dress', 'party_dress'], ['party dresses', 'party_dress'], ['dress party', 'party_dress'],    
 ['polos wanita', 'polos_wanita'], ['polos putih' , 'polos_putih'], ['polos biru' , 'polos_biru'],
 ['polos musim', 'polos_musim'], ['print bunga', 'print_bunga'], ['slim fit', 'slim_fit'],
 ['tali spaghetti', 'tali_spaghetti'], ['t shirt' , 't_shirt'], ['ukuran besar', 'ukuran_besar'], 
 ['warna biru', 'warna_biru'], ['warna hitam', 'warna_hitam'], ['warna merah', 'warna_merah'],
 ['warna polos', 'warna_polos'], ['warna putih', 'warna_putih'],  ['women casual', 'women_casual'],  
 ['dress polos', 'dress_polos'], ['dress retro', 'dress_retro'],  ['dress summer', 'dress_summer']    ] 

#************Basic Data Cleaning Function: Remove puncutation and extra spaces*************

def basic_data_cleaning_function (df, words_to_del):
    df["title"] = df['title'].str.replace('[^\w\s]',' ') # remove punctuations
    df["title"] = df['title'].str.replace('  ',' ')      # remove double space
    df["title"] = df['title'].str.replace('   ',' ')     # remove tripple space
    words_to_del=words_to_del.split()
    for w in words_to_del:        
        df['title']=df['title'].replace( {w : ''} , regex=True)
    return df

#***********Additional Data Cleaning Function: Reduce variance of certain key Words**********

def additional_data_cleaning_function (df, distinct_words, words_to_distill ):
    for w in words_to_distill:
        t=time.time()
        to_distill= [w_var for w_var in distinct_words if w in w_var ]
        to_distill= [w_var for w_var in to_distill if w_var!= w]
        print("Number of words to distill for", w,  "=", len(to_distill))
        for i in to_distill:
            df['title']=df['title'].replace( {i : w} , regex=True)
        print("Done distilling", w,  "Time:", round((time.time()-t),3), "secs")
    return df

#*********Function Call to Feature Engineering**********************************************

def feature_engineering_function (df, words_to_engineer):
  
    t = time.time()  
    for eq in words_to_engineer:
        df['title']=df['title'].replace( {eq[0] : eq[1]} , regex=True) #replace 1st word with 2nd word for each word pairs.
    print ("Done Feature Engineering, Time elapse: ", round((time.time()-t),3), "secs")
    return df

#************Main Training Function**********************************************************
#  train_func will receive model (A,or B), df (beauty_df, fashion_df or mobile_df), weight (None of 'balanced')
#  gram and ngram (1 to 8) and max_feat (max features for CountVectorizer) as inputs, 
#  and return validation predictions, test set predictions, Overall Accuracy, and length of test set as outputs. 

def train_func(model, df, cat, weight, gram,n_gram, max_feat):
    t = time.time()
    train_df = df[df['Category'].notnull()]
    test_df = df[df['Category'].isnull()]
    
    #---------convert words to vectors--------------------------------------
    vec = CountVectorizer(max_features = max_feat, ngram_range = (gram , n_gram))    
    train_text_counts= vec.fit_transform(train_df['title'])
    test_text_counts= vec.transform(test_df['title'])
    train_text_counts= normalize(train_text_counts)  
    test_text_counts= normalize(test_text_counts)
 
    print("Done Transforming", cat, "Vectorizer. (", gram, ",", n_gram,") grams Shape:",train_text_counts.shape,  "Time elapse: ", round((time.time()-t),3), "secs") 
    t = time.time()
    
    #---------Stratified KFold Splits-----------------------------------------    
    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)   
    
    print ("START TRAINING MODEL", model, ",", cat, "Category")    
    valid_pred=pd.DataFrame(columns=['actual']) # initialize an empty df to capture validation results
    test_pred=test_df[['itemid']]

    Overall_Accuracy=0 
    #---------Start Training-----------------------------------------    
    for n_fold, (Train_id_array, Eval_id_array) in enumerate(folds.split(train_text_counts, train_df['Category'].astype('category'))):
        t = time.time()
        X_Train, y_Train  = train_text_counts[Train_id_array], train_df['Category'].iloc[Train_id_array]
        X_Eval,  y_Eval   = train_text_counts[Eval_id_array], train_df['Category'].iloc[Eval_id_array]
    
        test_len=len(y_Eval)
        #clf = MultinomialNB().fit(X_Train, y_Train) # for debugging purpose
        clf=LinearSVC(class_weight=weight, max_iter=20, random_state=42).fit(X_Train, y_Train)
        
        validation_prediction=  clf.predict(X_Eval)          # Predict on Validation Set
        test_set_prediction = clf.predict(test_text_counts)  # Predict on Test Set 
        test_pred[model +'_Pred_' + str(n_fold + 1)]=test_set_prediction  
        
        accuracy=metrics.accuracy_score(y_Eval, validation_prediction)
        print("Done Training", cat, "fold", n_fold + 1, ", Accuracy:", accuracy,"Time elapse: ", round((time.time()-t),3), "secs")   
        t = time.time()      

        d = {'Validation Prediction': validation_prediction, 'actual': y_Eval}
        valid_pred_0 = pd.DataFrame(data=d)
        valid_pred= pd.concat([valid_pred,valid_pred_0]) #combine validation prediction of different folds
        
        Overall_Accuracy=Overall_Accuracy + accuracy
    
    Overall_Accuracy=Overall_Accuracy/num_folds
    print("OVERALL ACCURACY FOR MODEL",model, cat, "Category=", Overall_Accuracy )  
    
    return (valid_pred,test_pred, Overall_Accuracy, test_len )  

#**************Random Forest Classifier Model Stacking Function*****************************

def stacking_function(train_df, test_df):
    num_folds=10
    feat      = [col for col in train_df.columns if col not in ['actual']]
    feat_test = [col for col in test_df.columns if col not in ['itemid'] ]    
    folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    
    print("START TRAINING STACKING MODEL, train shape:",train_df.shape, "test shape:", test_df.shape)
    stacking_accuracy=0 #initialize stacking accuracy as zero first. 
    for n_fold, (Train_id_array, Eval_id_array) in enumerate(folds.split(train_df[feat], train_df['actual'])):
        t = time.time()
        X_Train, y_Train  = train_df[feat].iloc[Train_id_array], train_df['actual'].iloc[Train_id_array]
        X_Eval,  y_Eval   = train_df[feat].iloc[Eval_id_array], train_df['actual'].iloc[Eval_id_array]  

        clf=RandomForestClassifier(n_estimators=30, random_state=42).fit(X_Train, y_Train)
        valid_pred=  clf.predict(X_Eval)            # Predict on Validation Set
        accuracy=metrics.accuracy_score(y_Eval, valid_pred)
        stacking_accuracy= stacking_accuracy + accuracy    
        print("Done Training fold", n_fold + 1, ". Time elapse: ", round((time.time()-t),3), "secs. Training Accuracy:", accuracy )   
        t = time.time()
        test_pred = clf.predict_proba(test_df[feat_test])  # Predict on Test Set
        if n_fold !=0:
                test_pred=test_pred + test_pred
            
    overall_accuracy=stacking_accuracy/num_folds    
    test_pred= (test_pred / num_folds )       # take the average of predict_prob across folds
    test_pred= np.argmax(test_pred, axis=1)   # get the predictions of the max average probability each row

    print("Done Training. Time elapse: ", round((time.time()-t),3), "secs. Overall Accuracy:", overall_accuracy)        
    submission_df=test_df[['itemid']]
    submission_df['Category']=test_pred       

    #**********Create Prediction file *********************************************************************
    prediction_submit = submission_df.astype(int) # make sure predictions are in integer format
    prediction_submit = prediction_submit.set_index('itemid') # just to make sure there is no extra column of running numbers
    prediction_submit.to_csv(Submission_Version)

#**********Function to Display Overall Model training Accuracy *********************************************

def model_accuracy(md, b_acc, b_test_len, f_acc, f_test_len,m_acc, m_test_len):
    o_acc= (b_acc*b_test_len + f_acc*f_test_len + m_acc*m_test_len)/(b_test_len + m_test_len  +f_test_len)
    print ("TRAINING OVERALL ACCURACY FOR MODEL", md, ": ", o_acc )  
    
#*************Calling Basic Data Cleaning Function: *******************************************************

print ("Starting Basic Data Cleaning ...")
t = time.time() 
beauty_df = basic_data_cleaning_function (beauty_df, beauty_words_to_del)
fashion_df = basic_data_cleaning_function(fashion_df, fashion_words_to_del)
mobile_df = basic_data_cleaning_function (mobile_df, beauty_words_to_del)
print ("Done Basic Data Cleaning, Time elapse: ", round((time.time()-t),3), "secs")

#*************Calling Additional Data Cleaning Function: *************************************************
 
print ("Starting Additional Cleaning beauty_df")
beauty_df = additional_data_cleaning_function (beauty_df,beauty_distinct_words, beauty_words_to_distill)
print ("Starting Additional Cleaning fashion_df")
fashion_df = additional_data_cleaning_function (fashion_df, fashion_distinct_words, fashion_words_to_distill)
print ("Starting Additional Cleaning mobile_df")
mobile_df = additional_data_cleaning_function (mobile_df, mobile_distinct_words, mobile_words_to_distill)                 

#*************Calling Feature Engineering Function: ******************************************************** 

print ("Starting beauty_df Feature Engineering")
beauty_df = feature_engineering_function (beauty_df, beauty_words_to_engineer)
print ("Starting fashion_df Feature Engineering")
fashion_df = feature_engineering_function (fashion_df, fashion_words_to_engineer )
print ("Starting mobile_df Feature Engineering")
mobile_df = feature_engineering_function (mobile_df, mobile_words_to_engineer )  

#*************Evaluate Cleaning and Feature Engineering Results: ******************************************* 

beauty_distinct_words_after=' '.join(beauty_df['title']).split()
fashion_distinct_words_after=' '.join(fashion_df['title']).split()
mobile_distinct_words_after=' '.join(mobile_df['title']).split()

print ("Total Distinct Words in beauty_df After Cleaning & Feature Engineering: ", len(set(beauty_distinct_words_after)))
print ("Total Distinct Words in fashion_df After Cleaning & Feature Engineering: ", len(set(fashion_distinct_words_after)))
print ("Total Distinct Words in mobile_df After Cleaning & Feature Engineering: ", len(set(mobile_distinct_words_after)))
print ("TOTAL CLEANING & FEATURE ENGINEERING TIME: ", round((time.time()-t0),3), "secs")

#**********Function Call to build Model A: *****************************************************************

b_valid_pred_A, b_test_pred_A, b_acc, b_test_len = train_func('A', beauty_df, 'Beauty', None, 1,6,     1000000 )     
f_valid_pred_A, f_test_pred_A, f_acc, f_test_len  = train_func('A', fashion_df, 'Fashion', None, 1,8,  1400000 )
m_valid_pred_A, m_test_pred_A, m_acc, m_test_len = train_func('A', mobile_df,  'Mobile', None,  1,4,    200000 )
model_accuracy('A', b_acc, b_test_len, f_acc, f_test_len,m_acc, m_test_len)

#**********Function Call to build Model B: ****************************************************************

b_valid_pred_B, b_test_pred_B, b_acc, b_test_len = train_func('B', beauty_df, 'Beauty',  'balanced', 1,1, None) 
f_valid_pred_B, f_test_pred_B, f_acc, f_test_len = train_func('B', fashion_df,'Fashion', 'balanced', 1,1, None)
m_valid_pred_B, m_test_pred_B, m_acc, m_test_len = train_func('B', mobile_df, 'Mobile',  'balanced', 1,1, None)
model_accuracy('B', b_acc, b_test_len, f_acc, f_test_len, m_acc, m_test_len)

#**********Combine Test Set Predictions from Model A & B for stacking****************************************

test_pred_A = pd.concat([b_test_pred_A, f_test_pred_A, m_test_pred_A])
test_pred_A = test_pred_A.set_index('itemid')

test_pred_B = pd.concat([b_test_pred_B, f_test_pred_B, m_test_pred_B])
test_pred_B = test_pred_B.set_index('itemid')

test_pred_combined=pd.concat([test_pred_A, test_pred_B], axis=1) #combine test_pred for model A & B together
test_pred_combined=test_pred_combined.reset_index()

#**********Organize and Combine Validation Predictions from Model A & B for Stacking************************

valid_pred_A= pd.concat([b_valid_pred_A, f_valid_pred_A, m_valid_pred_A])
for i in np.arange(num_folds): #duplicate valid_pred X 20 times for Model stacking purpose
    valid_pred_A['A_val_fol_'+str(i+1)]=valid_pred_A['Validation Prediction']    
valid_pred_A= valid_pred_A.drop(['Validation Prediction','actual'], axis=1) #only one training label is needed.

valid_pred_B= pd.concat([b_valid_pred_B, f_valid_pred_B, m_valid_pred_B])
for i in np.arange(num_folds): #duplicate valid_pred X 20 times for Model stacking purpose
    valid_pred_B['B_val_fol_'+str(i+1)]=valid_pred_B['Validation Prediction'] 
valid_pred_B=valid_pred_B.drop(['Validation Prediction'], axis=1)

valid_pred_combined= pd.concat([valid_pred_A,valid_pred_B], axis=1) #combine valid_pred for model A & B together

#**********Function Call to Random Forest Classifier for Model Stacking**************************************

stacking_function(valid_pred_combined, test_pred_combined)

print ("Training total run time: ", round((time.time()-t0),3), "secs")