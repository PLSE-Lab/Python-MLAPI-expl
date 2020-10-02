#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('../input/credit-default-prediction-ai-big-data/train.csv')
test=pd.read_csv('../input/credit-default-prediction-ai-big-data/test.csv')


# In[ ]:


import re
[int(x) for x in str.split(' ') if x.isdigit()] #[0]

def getnumbers(string):
    retstr=[]
    for ci,xi in enumerate(string):
        #print(xi)
        if xi==np.nan:
            print()
        else:
            retstr.append( [int(s) for s in re.findall(r'-?\d+\.?\d*', xi)][0] )
    return retstr

train['Years in current job']=getnumbers( train['Years in current job'].fillna('10 Year').values )
test['Years in current job']=getnumbers( test['Years in current job'].fillna('10 Year').values )


# In[ ]:


train['Home Ownership'].unique()


# In[ ]:


def vulleeg(data):
    data['NetVal']=0
    for xi in range(len(data)):
        tempv=data.loc[xi,'Current Credit Balance']
        if data.loc[xi,'Current Loan Amount']==99999999.0:
            data.loc[xi,'Current Loan Amount']=tempv
        if data.loc[xi,'Term']=='Short Term':
            data.loc[xi,'Term']=24
        else:
            data.loc[xi,'Term']=120
        if data.loc[xi,'Home Ownership']=='Own Home':
            data.loc[xi,'NetVal']=data.loc[xi,'Annual Income']*7 - data.loc[xi,'Current Credit Balance']
        if data.loc[xi,'Home Ownership']=='Home Mortgage':
            data.loc[xi,'NetVal']=data.loc[xi,'Annual Income']*7 - data.loc[xi,'Current Credit Balance']

    data['Term2']=data['Current Loan Amount']/(data['Monthly Debt']+1)
    data['Term3']=data['Current Loan Amount']/(data['Annual Income']+1)
    data['ratio']=data['Maximum Open Credit']/(data['Annual Income']+1)
    data['Annual Income']=np.log(data['Annual Income']+1)
    data['Maximum Open Credit']=np.log(data['Maximum Open Credit']+1)
    data['Current Loan Amount']=np.log(data['Current Loan Amount']+1)
    data['Current Credit Balance']=np.log(data['Current Credit Balance']+1)
    data['Monthly Debt']=np.log(data['Monthly Debt']+1)
    data['NetVal']=np.log(data['NetVal']+1)

    data=data.replace(np.inf,999)
    data=data.replace(-np.inf,-999)
    
    
    return data
        
        
           
train=vulleeg(train)
test=vulleeg(test)
train


# In[ ]:


len(train['Credit Default'].dropna())


# In[ ]:


def kluster2(data,grbvar,label,nummercl,level):
    '''nummercl < ncol'''

    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report    
    from scipy import spatial
    import time
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA
    from umap import UMAP  # knn lookalike of tSNE but faster, so scales up
    from sklearn.manifold import TSNE,Isomap,SpectralEmbedding,spectral_embedding,LocallyLinearEmbedding,MDS #limit number of records to 100000

    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC,NuSVC
    import xgboost as xgb

    simdata=data[data[label].isnull()==False].drop([label],axis=1)

    #   Label encoding or remove string data
    ytrain=data[label]
    if True: 
        from category_encoders.cat_boost import CatBoostEncoder
        CBE_encoder = CatBoostEncoder()
        cols=[ci for ci in data.columns if ci not in ['index',label]]
        coltype=data.dtypes
        featured=[ci for ci in cols]
        ytrain=data[label]
        CBE_encoder.fit(data[:len(simdata)].drop(label,axis=1), ytrain[:len(simdata)])
        data=CBE_encoder.transform(data.drop(label,axis=1))
        data[label]=ytrain
        
    #add random columns
    for xi in range(5):
        labnm='rand'+str(xi)
        data[labnm]=np.random.randint(0,10,size=(len(data), 1))
    #find mean per label
    train_me=data.drop([grbvar],axis=1).groupby(label).mean()        
    #imput
    kol=data.drop(label,axis=1).columns
    from sklearn.experimental import enable_iterative_imputer  
    from sklearn.impute import IterativeImputer
    print( len(data.dropna()),len(data[data[label]!=np.nan]))
    if len(data.dropna())<len(data[label].dropna()):
        print('impute empty data')
        data = IterativeImputer(random_state=0).fit_transform(data.drop(label,axis=1))
    data = pd.DataFrame(data,columns=kol)   
    data[label]=ytrain.values
    #cosin similarity transform

    print(train_me)
    
    simdata=data[data[label].isnull()==False].drop([grbvar,label],axis=1)
    ytrain=data[data[label].isnull()==False][label]
    simtest=data[data[label].isnull()==True].drop([grbvar,label],axis=1)
    ytest=np.random.randint(0,1,size=(len(simtest), 1))  #fill data not used
    iddata=data[grbvar].astype('int')
    submit=data[data[label].isnull()==True][[grbvar,label]]
    print(submit.columns,submit.describe())
    if len(simtest)==0:
        simtest=simdata[int(len(simdata)*0.9):]
        ytest=ytrain[int(len(simdata)*0.9):]

    print(simdata.shape,simtest.shape,data.shape,ytrain.shape)
    #train_se=data.groupby('label').std()
    train_cs2=cosine_similarity(simdata,train_me)
    test_cs2=cosine_similarity(simtest,train_me)
    dicto={ np.round(i,1) : ytrain.unique()[i] for i in range(0, len(ytrain.unique()))} #print(clf.classes_)
    ypred=pd.Series(np.argmax(train_cs2,axis=1)).map(dicto)
    
    print('cosinesimilarity direction' ,classification_report(ytrain.values, ypred)  )
    
    trainmu=pd.DataFrame( simdata.values-simdata.values.mean(axis=1)[:,None])
    testmu=pd.DataFrame( simtest.values-simtest.values.mean(axis=1)[:,None])
    
    trainmu[label]=ytrain
    trainme2=trainmu.groupby(label).mean()    
    #spatial 0.79

    def adjcos_dist(size, matrix, matrixm):
        distances = np.zeros((len(matrix),size))
        M_u = matrix.mean(axis=1)
        m_sub = matrix - M_u[:,None]
        for first in range(0,len(matrix)):
            for sec in range(0,size):
                distance = spatial.distance.cosine(m_sub[first],matrixm[sec])
                distances[first,sec] = distance
        return distances

    trainsp2=adjcos_dist(len(trainme2),trainmu.drop(label,axis=1).values,trainme2.values)
    testsp2=adjcos_dist(len(trainme2),testmu.values,trainme2.values)
    
    print(trainsp2.shape,trainme2.shape,simdata.shape)
    print('cosinesimilarity distance', classification_report(ytrain.values, pd.Series(np.argmin(trainsp2,axis=1)).map(dicto)  )  )
    # blended with three classifiers random Forest
    classifier=[
                RandomForestClassifier(n_jobs=4),
                LogisticRegression(n_jobs=4),
                xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11,n_jobs=4),
                SVC(probability=True),
                KNeighborsClassifier(n_neighbors=3),
                PCA(n_components=nummercl,random_state=0,whiten=True),
                #TruncatedSVD(n_components=nummercl, n_iter=7, random_state=42),
                #FastICA(n_components=nummercl,random_state=0),
    ]
    simdata2=np.hstack((train_cs2,trainsp2))
    simtest2=np.hstack((test_cs2,testsp2))
    kol2=['x'+str(xi) for xi in range(nummercl)]+['y'+str(xi) for xi in range(nummercl)]
    for clf in classifier:
        #clf = RandomForestClassifier(n_jobs=4) #GaussianProcessClassifier()#
        print(simdata.shape,ytrain.shape,simtest.shape,data.shape,simdata2.shape,simtest2.shape)
        print(simtest2)

        try:
            clf.fit(simdata, ytrain)
            train_tr=clf.predict_proba(simdata)
            test_tr=clf.predict_proba(simtest)
        except:
            clf.fit(simdata.append(simtest))
            train_tr=clf.transform(simdata)
            test_tr=clf.transform(simtest)
            
        #dicto={ i : clf.classes_[i] for i in range(0, len(clf.classes_) ) } #print(clf.classes_)
        ypred=pd.Series(np.argmax(train_tr,axis=1)).map(dicto)
        print(str(clf)[:10] ,classification_report(ytrain.values, ypred)  )
        simdata2=np.hstack((simdata2,train_tr))
        simtest2=np.hstack((simtest2,test_tr))
        kol2=kol2+[str(clf)[:3]+str(xi) for xi in range(nummercl)]
    #concat data
    simdata=pd.DataFrame(simdata2,columns=kol2)
    simtest=pd.DataFrame(simtest2,columns=kol2)

    #plotimg2=pd.DataFrame(train_cs2,columns=['x'+str(xi) for xi in range(nummercl)])
    clusters = [PCA(n_components=nummercl,random_state=0,whiten=True),
                TruncatedSVD(n_components=nummercl, n_iter=7, random_state=42),
                FastICA(n_components=nummercl,random_state=0),
                Isomap(n_components=nummercl),
                LocallyLinearEmbedding(n_components=nummercl),
                SpectralEmbedding(n_components=nummercl),
                #MDS(n_components=nummercl),
                TSNE(n_components=3,random_state=0),
                UMAP(n_neighbors=nummercl,n_components=10, min_dist=0.3,metric='minkowski'),
                #NMF(n_components=nummercl,random_state=0),                
                ] 
    clunaam=['PCA','tSVD','ICA','Iso','LLE','Spectr','tSNE','UMAP','NMF']
    
    clf = RandomForestClassifier()
    clf.fit(simdata, ytrain)
        
    print('rFor pure',classification_report(ytrain,clf.predict(simdata))) 
    print(clf.predict(simtest))
    submit[label]=clf.predict(simtest)
    submit[label]=submit[label].astype('int')
    submit[grbvar]=iddata#submit[grbvar].values.astype('int')
    
    submit[[grbvar,label]].to_csv('submission.csv',index=False)
            
    for cli in clusters:
        print(cli)
        clunm=clunaam[clusters.index(cli)] #find naam
        
        if str(cli)[:3]=='NMF':
            maxmin=np.array([simdata.min(),simtest.min()])
            simdata=simdata-maxmin.min()+1
        svddata = cli.fit_transform(simdata.append(simtest))  #totale test
        
        #test what method is best
        #ttrain=pd.DataFrame(svddata)
        #ttrain[label]=ytrain
        #clustertechniques2(ttrain.reset_index(),label,'index') #.append(test)
        
        
        km = KMeans(n_clusters=nummercl, random_state=0)
        km.fit_transform(svddata)
        cluster_labels = km.labels_
        cluster_labels = pd.DataFrame(cluster_labels, columns=[label])
        #print(cluster_labels.shape) # train+test ok
        pd.DataFrame(svddata[:len(simdata)]).plot.scatter(x=0,y=1,c=ytrain.values,colormap='viridis')
        print(clunm,'kmeans_labelmean',cluster_labels.mean())   
        submit[label]=cluster_labels[len(simdata):]
        submit[label]=submit[label]#.astype('int')
        submit[[grbvar,label]].to_csv('Clu'+str(cli)[:5]+'kmean.csv',index=False)
        print('kmean'+str(cli)[:10],submit[[grbvar,label]].groupby(label).count() )
        clf= xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11,n_jobs=4)
        clf.fit(svddata[:len(simdata)], ytrain)
        print(clunm+'+xgb',classification_report(ytrain,clf.predict(svddata[:len(simdata)])))        
        #pd.DataFrame(svddata).plot.scatter(x=0,y=1,c=clf.predict(svddata),colormap='viridis')
        submit[label]=clf.predict(svddata[len(simdata):])
        #print(submit[submit[:,:].isnull()],submit[submit[label]==np.inf])
        submit[label]=submit[label].astype('int')
        submit[[grbvar,label]].to_csv('submitxgbkl_'+str(cli)[:5]+'.csv',index=False)
        print('xgb'+str(cli)[:10],submit[[grbvar,label]].groupby(label).count() )
    
        plt.show()

        #clusdata=pd.concat([pd.DataFrame(grbdata.reset_index()[grbvar]), cluster_labels], axis=1)
        #if len(grbdata)<3: 
        #    data['Clu'+clunm+str(level)]=cluster_labels.values
            
        #else:
        #    data=data.merge(clusdata,how='left',left_on=grbvar,right_on=grbvar)
        confmat=confusion_matrix ( ytrain,cluster_labels[:len(simdata)])
        dicti={}
        for xi in range(len(confmat)):
            #print(np.argmax(confmat[xi]),confmat[xi])
            dicti[xi]=np.argmax(confmat[xi])
        #print(dicti)
        #print('Correlation\n',confusion_matrix ( ytrain,cluster_labels[:len(ytrain)]))
        #print(clunm+'+kmean clusterfit', classification_report(ytrain.map(dicti), cluster_labels[:len(simdata)])  )   
        invdict = {np.round(value,1): key for key, value in dicti.items()}
        #print(invdict)
        submit[label]=cluster_labels[len(simdata):].values
        #print(cluster_labels[len(simdata):])
        #print(submit.describe().T)
        ytest=submit[label].astype('int')
        submit[label]=ytest.map(invdict)#.astype('int')
        submit[[grbvar,label]].to_csv('submit'+str(cli)[:5]+'kmean.csv',index=False)
        print('kmean'+str(cli)[:10],submit[[grbvar,label]].groupby(label).count() )        
    return data



#train2=kluster(plotimg[:10000].reset_index(),'index','label',10,1)
train2=kluster2( train.append(test,ignore_index=True),'Id','Credit Default',len(train['Credit Default'].unique() ),1)


# def kluster(data,grbvar,label,nummercl,level):
#     '''nummercl < ncol'''
# 
#     from sklearn.cluster import KMeans
#     from sklearn.metrics.pairwise import cosine_similarity
#     from sklearn.metrics import confusion_matrix
#     from sklearn.metrics import classification_report    
#     from scipy import spatial
#     
#     import matplotlib.pyplot as plt
#     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis
#     from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA
#     from umap import UMAP  # knn lookalike of tSNE but faster, so scales up
#     from sklearn.manifold import TSNE,Isomap,SpectralEmbedding,spectral_embedding,LocallyLinearEmbedding,MDS #limit number of records to 100000
# 
#     from sklearn.gaussian_process import GaussianProcessClassifier
# 
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.svm import SVC
# 
#     #   Label encoding or remove string data
#     ytrain=data[label]
#     if True: 
#         from category_encoders.cat_boost import CatBoostEncoder
#         CBE_encoder = CatBoostEncoder()
#         cols=[ci for ci in data.columns if ci not in ['index',label]]
#         coltype=data.dtypes
#         featured=[ci for ci in cols]
#         ytrain=data[label]
#         data = CBE_encoder.fit_transform(data.drop(label,axis=1), ytrain)
#         data[label]=ytrain
#     #imput
#     from sklearn.experimental import enable_iterative_imputer  
#     from sklearn.impute import IterativeImputer
#     data = IterativeImputer(random_state=0).fit_transform(data.drop(label,axis=1))
# 
#         
#     #cosin similarity transform
# 
#     train_me=data.groupby(label).mean()
#     simdata=data.drop(label,axis=1)
#     #train_se=data.groupby('label').std()
#     train_cs2=cosine_similarity(simdata,train_me)
#     print('cosinesimilarity direction' ,classification_report(ytrain.values, np.argmax(train_cs2,axis=1))  )
#     
#     trainmu=pd.DataFrame( simdata.values-simdata.values.mean(axis=1)[:,None])
#     trainmu[label]=ytrain
#     trainme2=trainmu.groupby(label).mean()    
#     #spatial 0.79
# 
#     def adjcos_dist(size, matrix, matrixm):
#         distances = np.zeros((len(matrix),size))
#         M_u = matrix.mean(axis=1)
#         m_sub = matrix - M_u[:,None]
#         for first in range(0,len(matrix)):
#             for sec in range(0,size):
#                 distance = spatial.distance.cosine(m_sub[first],matrixm[sec])
#                 distances[first,sec] = distance
#         return distances
# 
#     trainsp2=adjcos_dist(len(trainme2),trainmu.drop(label,axis=1).values,trainme2.values)
#     print(trainsp2.shape,trainme2.shape,simdata.shape)
#     print('cosinesimilarity distance', classification_report(ytrain.values, np.argmin(trainsp2,axis=1))  )
#     # blended with random Forest
#     
#     clf = GaussianProcessClassifier()#RandomForestClassifier()
#     clf.fit(simdata, ytrain)
#     train_tr=clf.predict_proba(simdata)
#     print('rFor' ,classification_report(ytrain.values, np.argmax(train_tr,axis=1))  )
# 
#     clf = LogisticRegression()
#     clf.fit(simdata, ytrain)
#     train_lo=clf.predict_proba(simdata)
#     print('logist' ,classification_report(ytrain.values, np.argmax(train_lo,axis=1))  )
# 
#     clf = SVC(kernel='rbf', gamma=0.7,probability=True)
#     clf.fit(simdata, ytrain)
#     train_sv=clf.predict_proba(simdata)
#     print('SVC' ,classification_report(ytrain.values, np.argmax(train_sv,axis=1))  )
# 
#     #concat data
#     plotimg=pd.DataFrame(1-trainsp2,columns=['y'+str(xi) for xi in range(nummercl)])
#     plotimg2=pd.DataFrame(train_cs2,columns=['x'+str(xi) for xi in range(nummercl)])
#     plotimg3=pd.DataFrame(train_tr,columns=['w'+str(xi) for xi in range(nummercl)])
#     plotimg4=pd.DataFrame(train_lo,columns=['v'+str(xi) for xi in range(nummercl)])
#     plotimg5=pd.DataFrame(train_sv,columns=['u'+str(xi) for xi in range(nummercl)])
#     for xi in plotimg2.columns:
#         plotimg[xi]=plotimg2[xi]
#     for xi in plotimg3.columns:
#         plotimg[xi]=plotimg3[xi]
#     for xi in plotimg4.columns:
#         plotimg[xi]=plotimg4[xi]
#     for xi in plotimg5.columns:
#         plotimg[xi]=plotimg5[xi]
#         
#     simdata=plotimg
#     #simdata = cosine_similarity(simdata)
#     clusters = [PCA(n_components=nummercl,random_state=0,whiten=True),
#                 TruncatedSVD(n_components=nummercl, n_iter=7, random_state=42),
#                 FastICA(n_components=nummercl,random_state=0),
#                 NMF(n_components=nummercl,random_state=0),
#                 Isomap(n_components=nummercl),
#                 LocallyLinearEmbedding(n_components=nummercl),
#                 SpectralEmbedding(n_components=nummercl),
#                 #MDS(n_components=nummercl),
#                 TSNE(n_components=3,random_state=0),
#                 UMAP(n_neighbors=nummercl,n_components=10, min_dist=0.3,metric='minkowski'),
#                 ] 
#     clunaam=['PCA','tSVD','ICA','NMF','Iso','LLE','Spectr','tSNE','UMAP']
#     
#     #grbdata=data.groupby(grbvar).mean()
#     #simdata = cosine_similarity(grbdata.fillna(0))
#     #print(grbdata.shape,simdata.shape)
#     #if len(grbdata)<3:
#     #    simdata=data.drop(grbvar,axis=1)
#     #    simdata=simdata.dot(simdata.T)
#     #    from sklearn import preprocessing
#     #    simdata = preprocessing.MinMaxScaler().fit_transform(simdata)
# 
#     clf = RandomForestClassifier()
#     clf.fit(simdata, ytrain)
#     print('rFor pure',classification_report(ytrain,clf.predict(simdata))) 
#     
#     for cli in clusters:
#         print(cli)
#         clunm=clunaam[clusters.index(cli)] #find naam
#         if clunm=='NMF':
#             simdata=simdata-simdata.min()+1
#         svddata = cli.fit_transform(simdata)
#         
#         #test what method is best
#         #ttrain=pd.DataFrame(svddata)
#         #ttrain[label]=ytrain
#         #clustertechniques2(ttrain.reset_index(),label,'index') #.append(test)
#         
#         
#         km = KMeans(n_clusters=nummercl, random_state=0)
#         km.fit_transform(svddata)
#         cluster_labels = km.labels_
#         clulabel='Clu'+clunm+str(level)
#         cluster_labels = pd.DataFrame(cluster_labels, columns=[clulabel])
#         print(cluster_labels.shape)
#         pd.DataFrame(svddata).plot.scatter(x=0,y=1,c=ytrain.values,colormap='viridis')
#         print(clunm,'kmeans_labelmean',cluster_labels.mean())        
#         from sklearn.ensemble import RandomForestClassifier
#         from sklearn.svm import LinearSVC,SVC
#         import xgboost as xgb
# 
#         clf = SVC()#RandomForestClassifier()
#         clf= xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11,n_jobs=4)
#         clf.fit(svddata, ytrain)
#         print(clunm+'+xgb',classification_report(ytrain,clf.predict(svddata)))        
#         #pd.DataFrame(svddata).plot.scatter(x=0,y=1,c=clf.predict(svddata),colormap='viridis')
#         
# 
#         plt.show()
# 
#         #clusdata=pd.concat([pd.DataFrame(grbdata.reset_index()[grbvar]), cluster_labels], axis=1)
#         #if len(grbdata)<3: 
#         #    data['Clu'+clunm+str(level)]=cluster_labels.values
#             
#         #else:
#         #    data=data.merge(clusdata,how='left',left_on=grbvar,right_on=grbvar)
#         confmat=confusion_matrix ( ytrain,cluster_labels)
#         dicti={}
#         for xi in range(len(confmat)):
#             #print(np.argmax(confmat[xi]),confmat[xi])
#             dicti[xi]=np.argmax(confmat[xi])
#         #print('Correlation\n',confusion_matrix ( ytrain,cluster_labels))
#         print(clunm+'+kmean clusterfit', classification_report(ytrain.map(dicti), cluster_labels)  )            
#     return data
# 
# 
# 
# #train2=kluster(plotimg[:10000].reset_index(),'index','label',10,1)
# #train2=kluster(train,'Id','Credit Default',len(train['Credit Default'].unique() ),1)
# 

# In[ ]:


train.append(test)


# def kluster2(data,grbvar,label,nummercl,level):
#     '''nummercl < ncol'''
# 
#     from sklearn.cluster import KMeans
#     from sklearn.metrics.pairwise import cosine_similarity
#     from sklearn.metrics import confusion_matrix
#     from sklearn.metrics import classification_report    
#     from scipy import spatial
#     
#     import matplotlib.pyplot as plt
#     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis
#     from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA
#     from umap import UMAP  # knn lookalike of tSNE but faster, so scales up
#     from sklearn.manifold import TSNE,Isomap,SpectralEmbedding,spectral_embedding,LocallyLinearEmbedding,MDS #limit number of records to 100000
# 
#     from sklearn.gaussian_process import GaussianProcessClassifier
# 
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.svm import SVC,NuSVC
# 
#     #   Label encoding or remove string data
#     ytrain=data[label]
#     if True: 
#         from category_encoders.cat_boost import CatBoostEncoder
#         CBE_encoder = CatBoostEncoder()
#         cols=[ci for ci in data.columns if ci not in ['index',label]]
#         coltype=data.dtypes
#         featured=[ci for ci in cols]
#         ytrain=data[label]
#         data = CBE_encoder.fit_transform(data.drop(label,axis=1), ytrain)
#         data[label]=ytrain
#         
#     #add random columns
#     for xi in range(5):
#         labnm='rand'+str(xi)
#         data[labnm]=np.random.randint(0,10,size=(len(data), 1))
#     #find mean per label
#     train_me=data.groupby(label).mean()        
#     #imput
#     kol=data.drop(label,axis=1).columns
#     from sklearn.experimental import enable_iterative_imputer  
#     from sklearn.impute import IterativeImputer
#     data = IterativeImputer(random_state=0).fit_transform(data.drop(label,axis=1))
#     data = pd.DataFrame(data,columns=kol)   
#     data[label]=ytrain.values
#     #cosin similarity transform
# 
#     print(train_me)
#     simdata=data[data[label].isnull()==False].drop([label],axis=1)
#     ytrain=data[data[label].isnull()==False][label]
#     simtest=data[data[label].isnull()==True].drop([label],axis=1)
#     ytest=ytrain.sample(len(simtest))  #fill data not used
#     iddata=data[grbvar]
#     submit=data[data[label].isnull()==True]
#     #print(submit.columns,submit.describe())
# 
#     
#     print(simdata.shape,simtest.shape,data.shape,ytrain.shape)
#     #train_se=data.groupby('label').std()
#     train_cs2=cosine_similarity(simdata,train_me)
#     test_cs2=cosine_similarity(simtest,train_me)
#     print('cosinesimilarity direction' ,classification_report(ytrain.values, np.argmax(train_cs2,axis=1))  )
#     
#     trainmu=pd.DataFrame( simdata.values-simdata.values.mean(axis=1)[:,None])
#     testmu=pd.DataFrame( simtest.values-simtest.values.mean(axis=1)[:,None])
#     
#     trainmu[label]=ytrain
#     trainme2=trainmu.groupby(label).mean()    
#     #spatial 0.79
# 
#     def adjcos_dist(size, matrix, matrixm):
#         distances = np.zeros((len(matrix),size))
#         M_u = matrix.mean(axis=1)
#         m_sub = matrix - M_u[:,None]
#         for first in range(0,len(matrix)):
#             for sec in range(0,size):
#                 distance = spatial.distance.cosine(m_sub[first],matrixm[sec])
#                 distances[first,sec] = distance
#         return distances
# 
#     trainsp2=adjcos_dist(len(trainme2),trainmu.drop(label,axis=1).values,trainme2.values)
#     testsp2=adjcos_dist(len(trainme2),testmu.values,trainme2.values)
#     
#     print(trainsp2.shape,trainme2.shape,simdata.shape)
#     print('cosinesimilarity distance', classification_report(ytrain.values, np.argmin(trainsp2,axis=1))  )
#     # blended with three classifiers random Forest
#     
#     clf = GaussianProcessClassifier()#RandomForestClassifier()
#     clf.fit(simdata, ytrain)
#     train_tr=clf.predict_proba(simdata)
#     test_tr=clf.predict_proba(simtest)
#     print('Gauss' ,classification_report(ytrain.values, np.argmax(train_tr,axis=1))  )
# 
#     clf = LogisticRegression()
#     clf.fit(simdata, ytrain)
#     train_lo=clf.predict_proba(simdata)
#     test_lo=clf.predict_proba(simtest)
#     
#     print('logist' ,classification_report(ytrain.values, np.argmax(train_lo,axis=1))  )
# 
#     clf = NuSVC(probability=True)
#     clf.fit(simdata, ytrain)
#     train_sv=clf.predict_proba(simdata)
#     test_sv=clf.predict_proba(simtest)
#     
#     print('SVC' ,classification_report(ytrain.values, np.argmax(train_sv,axis=1))  )
# 
#     #concat data
#     kol2=['x'+str(xi) for xi in range(nummercl)]+['y'+str(xi) for xi in range(nummercl)]+['w'+str(xi) for xi in range(nummercl)]+['v'+str(xi) for xi in range(nummercl)]+['u'+str(xi) for xi in range(nummercl)]
#     simdata=pd.DataFrame(np.hstack((train_cs2,trainsp2,train_tr,train_lo,train_sv)),columns=kol2)
#     simtest=pd.DataFrame(np.hstack((test_cs2,testsp2,test_tr,test_lo,test_sv)),columns=kol2)
# 
#     #plotimg2=pd.DataFrame(train_cs2,columns=['x'+str(xi) for xi in range(nummercl)])
#     clusters = [PCA(n_components=nummercl,random_state=0,whiten=True),
#                 TruncatedSVD(n_components=nummercl, n_iter=7, random_state=42),
#                 FastICA(n_components=nummercl,random_state=0),
#                 NMF(n_components=nummercl,random_state=0),
#                 Isomap(n_components=nummercl),
#                 LocallyLinearEmbedding(n_components=nummercl),
#                 SpectralEmbedding(n_components=nummercl),
#                 #MDS(n_components=nummercl),
#                 TSNE(n_components=3,random_state=0),
#                 UMAP(n_neighbors=nummercl,n_components=10, min_dist=0.3,metric='minkowski'),
#                 ] 
#     clunaam=['PCA','tSVD','ICA','NMF','Iso','LLE','Spectr','tSNE','UMAP']
#     
#     clf = RandomForestClassifier()
#     clf.fit(simdata, ytrain)
#     print('rFor pure',classification_report(ytrain,clf.predict(simdata))) 
#     
#     submit[label]=clf.predict(simtest)
#     submit[label]=submit[label].astype('int')
#     submit[grbvar]=submit[grbvar].astype('int')
#     
#     submit[[grbvar,label]].to_csv('submit_rfor.csv',index=False)
#             
#     for cli in clusters:
#         print(cli)
#         clunm=clunaam[clusters.index(cli)] #find naam
#         if clunm=='NMF':
#             simdata=simdata-simdata.min()+1
#         svddata = cli.fit_transform(simdata.append(simtest))
#         
#         #test what method is best
#         #ttrain=pd.DataFrame(svddata)
#         #ttrain[label]=ytrain
#         #clustertechniques2(ttrain.reset_index(),label,'index') #.append(test)
#         
#         
#         km = KMeans(n_clusters=nummercl, random_state=0)
#         km.fit_transform(svddata)
#         cluster_labels = km.labels_
#         cluster_labels = pd.DataFrame(cluster_labels, columns=[label])
#         print(cluster_labels.shape)
#         pd.DataFrame(svddata[:len(simdata)]).plot.scatter(x=0,y=1,c=ytrain.values,colormap='viridis')
#         print(clunm,'kmeans_labelmean',cluster_labels.mean())   
#         submit[label]=cluster_labels[len(simdata):]
#         submit[label]=submit[label].astype('int')
#         submit[[grbvar,label]].to_csv('Clu'+str(cli)[:3]+'kmean.csv',index=False)
#         print('kmean'+str(cli)[:10],submit[[grbvar,label]].groupby(label).count() )
#         from sklearn.ensemble import RandomForestClassifier
#         from sklearn.svm import LinearSVC,SVC
#         import xgboost as xgb
# 
#         clf = NuSVC()#RandomForestClassifier()
#         clf= xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11,n_jobs=4)
#         clf.fit(svddata[:len(simdata)], ytrain)
#         print(clunm+'+xgb',classification_report(ytrain,clf.predict(svddata[:len(simdata)])))        
#         #pd.DataFrame(svddata).plot.scatter(x=0,y=1,c=clf.predict(svddata),colormap='viridis')
#         submit[label]=clf.predict(svddata[len(simdata):])
#         submit[label]=submit[label].astype('int')
#     
#         submit[[grbvar,label]].to_csv('submitkl_'+str(cli)[:3]+'.csv',index=False)
#         print('xgb'+str(cli)[:10],submit[[grbvar,label]].groupby(label).count() )
#     
#         plt.show()
# 
#         #clusdata=pd.concat([pd.DataFrame(grbdata.reset_index()[grbvar]), cluster_labels], axis=1)
#         #if len(grbdata)<3: 
#         #    data['Clu'+clunm+str(level)]=cluster_labels.values
#             
#         #else:
#         #    data=data.merge(clusdata,how='left',left_on=grbvar,right_on=grbvar)
#         confmat=confusion_matrix ( ytrain,cluster_labels[:len(simdata)])
#         dicti={}
#         for xi in range(len(confmat)):
#             #print(np.argmax(confmat[xi]),confmat[xi])
#             dicti[xi]=np.argmax(confmat[xi])
#         #print('Correlation\n',confusion_matrix ( ytrain,cluster_labels))
#         print(clunm+'+kmean clusterfit', classification_report(ytrain.map(dicti), cluster_labels[:len(simdata)])  )   
#         invdict = {value: key for key, value in dicti.items()}
#         submit[label]=cluster_labels[len(simdata):]
#         ytest=submit[label].astype('int')
#         submit[label]=ytest.map(invdict)#.astype('int')
#         submit[[grbvar,label]].to_csv('Clu'+str(cli)[:3]+'kmean.csv',index=False)
#         print('kmean'+str(cli)[:10],submit[[grbvar,label]].groupby(label).count() )        
#     return data
# 
# 
# 
# #train2=kluster(plotimg[:10000].reset_index(),'index','label',10,1)
# train2=kluster2(train.append(test),'Id','Credit Default',len(train['Credit Default'].unique() ),1)
# 

# In[ ]:


train.shape,test.shape


# In[ ]:



def data2sparse(data):
    #print(data.info())
    typekolom=data.dtypes
    nummercol=[x for x in data.columns if typekolom[x]!='object']
    objcol=[x for x in data.columns if typekolom[x]=='object']
    print(nummercol)
    from scipy.sparse import coo_matrix,vstack,hstack,csr_matrix
    totalSPNr=coo_matrix(data[nummercol])
    sparse=pd.DataFrame( totalSPNr.col,columns=['col'] )
    sparse['row']=pd.DataFrame( totalSPNr.row )
    sparse['data']=pd.DataFrame(totalSPNr.data)
    sparse=sparse.dropna()
    totalSPN=coo_matrix((sparse.data,(sparse.col,sparse.row)))
    from sklearn.preprocessing import OneHotEncoder
    enc=OneHotEncoder()
    ohenc=enc.fit_transform(data[objcol])
    totalSPN=vstack([totalSPN,ohenc.T] )
    kolom=[x for x in nummercol]+[y for y in enc.get_feature_names(objcol)]
    return pd.DataFrame( csr_matrix(totalSPN.transpose()).todense(),columns=kolom)

data2sparse(train)


# In[ ]:


#train['total']=train['Home Ownership']+' '+train['Purpose']+' '+train['Term']
#test['total']=test['Home Ownership']+' '+test['Purpose']+' '+test['Term']
kolom=[x for x in train.columns if x not in ['Id','total','Home Ownership','Purpose','Term','total','Credit Default']]
kolom


# In[ ]:


def ALSforecast(train,test,ytrain,ytest,textcolumn,valuecolumns,indexv,label):
    print('input data',len(train),len(test),len(ytrain),len(ytest))
    from scipy.sparse import coo_matrix,hstack,csr_matrix
    def data2sparse(data):
        #print(data.info())
        typekolom=data.dtypes
        nummercol=[x for x in data.columns if typekolom[x]!='object']
        objcol=[x for x in data.columns if typekolom[x]=='object']
        print(nummercol)
        from scipy.sparse import coo_matrix,vstack,hstack,csr_matrix
        totalSPNr=coo_matrix(data[nummercol])
        sparse=pd.DataFrame( totalSPNr.col,columns=['col'] )
        sparse['row']=pd.DataFrame( totalSPNr.row )
        sparse['data']=pd.DataFrame(totalSPNr.data)
        sparse=sparse.dropna()
        totalSPN=coo_matrix((sparse.data,(sparse.col,sparse.row)))
        from sklearn.preprocessing import OneHotEncoder
        enc=OneHotEncoder()
        ohenc=enc.fit_transform(data[objcol])
        totalSPN=vstack([totalSPN,ohenc.T] )
        kolom=[x for x in nummercol]+[y for y in enc.get_feature_names(objcol)]
        return csr_matrix(totalSPN.transpose()),kolom
    
    print(textcolumn)    
    #_________________________________________________________________
    #prepare data
    # random dataenricher
    if False:
            cols=[ci for ci in train.columns if ci not in [indexv,'index',label,textcolumn,'target']]
            coltype=train.dtypes

            for ci in cols:
                if (coltype[ci]!="object"  ):
                    train[ci]=train[ci].astype('float32')
                    print(ci)
                    for di in cols[cols.index(ci)+1:]:
                        if (coltype[di]!="object"  ):
                            s2=(train[ci]*train[di]).astype('float32')
                            if np.abs(ytrain.corr(s2))>0.15:
                                train[ci+'x'+di]=s2
                                test[ci+'x'+di]=(test[ci]*test[di]).astype('float32')
                                valuecolumns=valuecolumns+[ci+'x'+di]
                            s2=(train[ci]/(train[di]+1.1)).astype('float32')
                            if np.abs(ytrain.corr(s2))>0.15:
                                train[ci+'/'+di]=s2
                                test[ci+'/'+di]=(test[ci]/(test[di]+1)  ).astype('float32') 
                                valuecolumns=valuecolumns+[ci+'/'+di]
                            s2=(train[ci]*np.log(train[di]+1.1)).astype('float32')
                            if np.abs(ytrain.corr(s2))>0.15:
                                train[ci+'log'+di]=s2
                                test[ci+'log'+di]=(test[ci]*np.log(test[di]+1)  ).astype('float32') 
                                valuecolumns=valuecolumns+[ci+'log'+di]

    df=train.append(test)   
    ratings,kolom=data2sparse(df.drop([label],axis=1))
    #transform embed regression forecast
    if False:
        #dropcolumns=[x for x in df.columns if x not in valuecolumns]
        X=ratings[:len(train)]
        Z=ratings[len(train):]
        Y=Z
        #print(np.linalg.pinv( X.T.dot(X)) )
        XtXi=np.linalg.pinv(np.dot(X.T,X).toarray()) 
        print(XtXi.shape,X.shape,Y.shape)
        from scipy.sparse import coo_matrix,vstack,hstack,csr_matrix
        for li in range(1,int(len(train)/len(test)+1)):
            Y = vstack((Y,Z))#np.concatenate((Y, Z))
            print(Y.shape)
        Y=Y[:len(train)]
        XtXiXt=np.dot(csr_matrix(XtXi),X.T) 
        XtXiXtZ=np.dot(XtXiXt,Y)
        print(XtXiXtZ.shape,X.shape,ratings.shape)
        Yhat=pd.DataFrame( np.dot(ratings,XtXiXtZ).toarray(),columns=kolom ,index=df.index) 
        Yhat[indexv]=df[indexv]     #restore index
        Yhat[label+'__Yh']=df[label]  #embed forecast
        Yhat[label]=df[label]       #restore label
        for ci in valuecolumns:
            df[ci+'_rest']=df[ci]-Yhat[ci]
            #dtrain[ci+'_ratio']=dtrain[ci]/(Yhat[ci]+1)
        print('transformed splitted basic/rest',df.shape)

    
    #valuecolumns=[x for x in train.columns if x not in [textcolumn,indexv]]
    print('expanded',df.shape,valuecolumns)
    #data to sparse vectorizer
    #ratings=data2sparse(df.drop(label,axis=1))
    #try:
    if False:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        ratings=vectorizer.fit_transform(df[textcolumn].fillna(' ')  )
        print(ratings.shape,'tfidf')
        ratings=hstack([ratings,data2sparse(df[valuecolumns])] )
        print(ratings.shape,'stacked')
    #except:
        #try:
        #    ratings=data2sparse(df[valuecolumns])
        #except:
        #    ratings=vectorizer.fit_transform(df[textcolumn].fillna(' ')  )                    
    print('rating matrix ready',ratings.shape)
    
    #_________________________________________________________________
    #find hidden data
    #ALS
    import tqdm
    import time
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking 
    from implicit.nearest_neighbours import (BM25Recommender, CosineRecommender,TFIDFRecommender, bm25_weight)
    #filter for NMF
    #min_rating=0.0
    #ratings.data[ratings.data < min_rating] = 0
    #ratings.eliminate_zeros()
    #
    ratings.data = np.ones(len(ratings.data))
    # generate a recommender model based off the input params
    model = AlternatingLeastSquares()  #ALS model                --------------------------------------------- change possible
    model = BayesianPersonalizedRanking()  #Baysian ranking model
    # lets weight these models by bm25weight.
    print("weighting matrix by bm25_weight")
    ratings = (bm25_weight(ratings, B=0.9) * 5).tocsr()
    print("training model %s", model)
    start = time.time()
    model.fit(ratings)
    print("trained model '%s' in %s", time.time() - start)
    result=model.fit(ratings)
    print('user',model.user_factors.shape,'item', model.item_factors.shape)
    
    #_________________________________________________________________
    #print clustering graph on factors
    #from umap import UMAP  # knn lookalike of tSNE but faster, so scales up
    #from sklearn.manifold import TSNE #limit number of records to 100000
    from matplotlib import pyplot as plt
    #X_total_clu = UMAP(n_neighbors=2,n_components=10, min_dist=0.3,metric='minkowski').fit_transform(model.item_factors)
    #X_total_clu = TSNE().fit_transform(model.item_factors)
    #print(X_total_clu.shape)
    #plt.scatter(X_total_clu[:len(train),0],X_total_clu[:len(train),1] ,c=ytrain.values,cmap='prism')
    #plt.show()
    
    #_________________________________________________________________
    reconstr=np.dot(model.item_factors,model.user_factors.T)
    print(reconstr.shape)
    
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,SGDClassifier,LogisticRegression
    import xgboost as xgb
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier,MLPRegressor
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_absolute_error
    from sklearn.svm import SVC, LinearSVC,NuSVC

    classifiers = [KNeighborsClassifier(n_neighbors=2,n_jobs=-1),
                   xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11,n_jobs=4),
                   SGDClassifier(average=True,max_iter=100),
                   DecisionTreeClassifier(),
                   LogisticRegression( solver="lbfgs",multi_class ='auto'),#solver="lbfgs",max_iter=500,n_jobs=-1,multi_class ='auto'),
                   MLPClassifier(alpha=0.510,activation='logistic'),
                   RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1, oob_score=True),
                   ExtraTreesClassifier(n_estimators=10, max_depth=50, min_samples_split=5, min_samples_leaf=1, random_state=None, min_impurity_decrease=1e-7),     
                   PassiveAggressiveClassifier(max_iter=50, tol=1e-3,n_jobs=-1),
                   MLPClassifier(),
                   SVC(),
                   
                    ]

    for classif in classifiers:
        #classif=LogisticRegression( solver="lbfgs",max_iter=500,n_jobs=-1,multi_class ='auto')

        classif.fit(reconstr[:len(train)],ytrain ) #model.item_factors[:len(train)]
        trainpredi=classif.predict(reconstr[:len(train)])
        testpredi=classif.predict(reconstr[len(train):]) 
        plt.scatter(x=trainpredi, y=ytrain, marker='.', alpha=0.3)
        print('classifier',str(classif))
        print('compare train.algo ratio, train mean -test mean',ytrain.mean()/trainpredi.mean(),trainpredi.mean(),testpredi.mean())
        plt.show()   
        
        submit = pd.DataFrame({indexv: test[indexv],label: testpredi})
        
        submit[label]=submit[label].astype('int')
        if str(classif)[:3]=='SVC':
            filenaam='submission.csv' #str(classif)[:3]+
        else:
            filenaam=str(classif)[:3]+'submission.csv' #
        submit.to_csv(path_or_buf =filenaam, index=False)

        try:
            cm = confusion_matrix(trainpredi, ytrain )
            cmtest = confusion_matrix(testpredi, ytest)
            print("Confusion Matrix: \n", cm)
            print("Test Confusion Matrix: \n", cmtest)
            print("Classification Report: \n", classification_report(trainpredi, ytrain ))
            print("Test Classification Report: \n", classification_report(testpredi, ytest ))
        
        except:
            cm = confusion_matrix(trainpredi, ytrain )
            print('mean abs error train -test', mean_absolute_error(trainpredi,ytrain))
            
            print("Confusion Matrix: \n", cm)
            print("Classification Report: \n", classification_report(trainpredi,ytrain ))
    return submit


colom=[x for x in train.columns if x not in ['qid','question_text'] ]

temp=ALSforecast(train,test,train['Credit Default'],train['Credit Default'],'total',kolom,'Id','Credit Default')


# In[ ]:


# imputer for handling missing values
for ci in train.columns:
    coltype=train.dtypes
    if (coltype[ci]!="object"):
        train[ci]=train[ci].fillna(train[ci].median())
        if ci !='Credit Default':
            test[ci]=test[ci].fillna(train[ci].median())

