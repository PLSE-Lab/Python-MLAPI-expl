import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.linalg import sqrtm
from copy import deepcopy
from sklearn.linear_model import LogisticRegression

##########################################################################


usecols1 = ['fecha_dato', 'ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

train = pd.read_csv("../input/train_ver2.csv", usecols=usecols1)
train = train[train['fecha_dato']=="2016-05-28"].drop("fecha_dato", axis = 1)
true = deepcopy(train)

#############################################################################

users = true['ncodpers'].tolist()
true.drop('ncodpers', axis=1, inplace=True)

items = true.columns.tolist()
user_index = {}
for i in range(len(users)):
    user_index[users[i]] = i

trueMat = np.array(true)
del true
print("users dict formed")

##########################################################################

cols = ['sexo','age','indext','renta','ind_nuevo','indresi']
limit_rows   = 4000000
train = pd.read_csv("../input/train_ver2.csv", usecols = cols, nrows= limit_rows)

def formatting(df, nanfix):
    df.replace(to_replace={'indresi':{'S':1,'N':0}, 'indext':{'S':1,'N':0},
        'sexo':{'H':1,'V':0}}, inplace=True)
    print("step one")
    print(train.isnull().any())
    if nanfix==True:
        df['renta'].fillna(value= df['renta'].mean(), inplace=True)
        print("1")
        df['sexo'].fillna(value= 0, inplace=True)
        print("2")
        df['ind_nuevo'].fillna(value= 0, inplace=True)
        print("3")
        #df['indrel'].fillna(value= 1, inplace=True)
        #print("4")
        df['indresi'].fillna(value= 0, inplace=True)
        print("5")
        df['indext'].fillna(value= 0, inplace=True)
        print("6")
        #df['ind_actividad_cliente'].fillna(value= 0, inplace=True)
        #print("7")
        #df['antiguedad'].fillna(value= 3, inplace=True)
        #print("8")
        


    print("step two")
    df.replace(to_replace=' NA', value=0, inplace=True)
    df.replace(to_replace='         NA',value=0, inplace=True)
    df.replace(to_replace='         NA',value=0, inplace=True)
    df.replace(to_replace='     NA',value=0, inplace=True)

    return df
    #print(train.isnull().any())

train = formatting(train, nanfix=True)

"""
for i in range(len(train)):
    a = list(train.ix[i,:])
    try:
        list1=[float(x) for x in a]
    except ValueError:
        print(a," on line",i)
    
"""
print("train loaded")

fieldcols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

#fields = pd.read_csv("../input/train_ver2.csv", usecols = fieldcols)
testing = pd.read_csv("../input/test_ver2.csv", usecols = ['ncodpers']+cols)
print("test length: ", len(testing))
testusers = testing['ncodpers'].tolist()
testing.drop('ncodpers', axis=1, inplace=True)
testing = testing[cols]
print("hoolah")
test = formatting(testing, nanfix=False)
test.fillna(0, inplace=True)
print("hoolah")

del testing

pred = {}
pred['ncodpers'] = testusers

#print(train)
#print(test)

print("test loaded")

for col in fieldcols:
    print(col)
    fields = pd.read_csv("../input/train_ver2.csv", usecols = [col], nrows=limit_rows)
    fields.fillna(0, inplace=True)
    model = LogisticRegression()
    model.fit(train, fields[col])
    colpred = model.predict_proba(test)
    print(colpred)
    pred[col] = [colpred[i][1] for i in range(len(colpred))]
    
    #print(col,"fit done")

#print(pred)
predDf = pd.DataFrame(pred)
testMat = np.array(predDf[fieldcols])
#testMat = np.ma.masked_where(trueMat[:,:]==1,testMat[:,:]).filled(fill_value = -999)

def max_items(x, leest):
    out = []

    for xx in x:
        if leest[xx] != -999: # setting a threshold
            out.append(items[xx])

    return out

i=0
for user in testusers:
    j = user_index[user]
    testMat[i,:] = np.ma.masked_where(trueMat[j,:]==1,testMat[i,:]).filled(fill_value = -999)
    i += 1

############################################################################
del pred
del predDf
del test
del train

print("first part completed")
#train = pd.read_csv("../input/train_ver2.csv", usecols=usecols1)
#train1 = train[train['fecha_dato']=="2016-05-28"].drop("fecha_dato", axis = 1)
#train2 = train[train['fecha_dato']=="2016-04-28"].drop("fecha_dato", axis = 1)
test = pd.read_csv("../input/test_ver2.csv")
test = test[['ncodpers']]
#del train
#print("datasets loaded")

############################################################################

def reorder(train):
    train.index = train['ncodpers'].tolist()
    train.drop('ncodpers', axis=1, inplace=True)
    train = train.reindex(users)
    return train

#train3 = reorder(train3)
#train2 = reorder(train2)
#train1 = reorder(train1)
###################### COMPUTING THE SVD ##########################

def svd(train, trueMat, k):
    utilMat = np.array(train)

    mask = np.isnan(utilMat)
    masked_arr=np.ma.masked_array(utilMat, mask)
    item_means=np.mean(masked_arr, axis=0)
    utilMat = masked_arr.filled(item_means)

    x = np.tile(item_means, (utilMat.shape[0],1))

    utilMat = utilMat - x

    print(utilMat)

    U, s, V=np.linalg.svd(utilMat, full_matrices=False)
    s=np.diag(s)
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]

    s_root=sqrtm(s)

    Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)
    UsV = np.dot(Usk, skV)

    UsV = UsV + x

    UsV = np.ma.masked_where(trueMat==1,UsV).filled(fill_value=-999)

    print("svd done")
    return UsV

trueMat = svd(train=trueMat,trueMat=trueMat,k=4)

"""
def max_items_2(x,y,z):
    deect = {}
    for i in range(len(x)):
        deect[x[i]] = (i, z[x[i]])
    for i in range(len(y)):
        if y[i] in deect:
            deect[y[i]] = (i+deect[x[i]][0], z[x[i]])
        else:
            deect[y[i]] = (i + 12, z[y[i]])
        
    leest = []
    for d in deect:
        leest.append((d, deect[d][0], deect[d][1]))
    
    for i in range(1,len(leest)):
        j = i
        
        while j>0 and (leest[j-1][1]>leest[j][1] or (leest[j][1] == leest[j-1][1] and leest[j][2] > leest[j-1][2])):
            temp = leest[j]
            leest[j] = leest[j-1]
            leest[j-1] = temp
            
            j=j-1
    
    
    out = []
    ind = leest[:7]
    for xx in ind:
        #if leest[xx]>0.001: # setting a threshold
        out.append(items[xx[0]])

    return out
"""

pred = []
print(testMat.shape, trueMat.shape)

n = 7
i = 0
for user in testusers:
    j = user_index[user]
    max_proba = testMat[i,:].argsort()[-1:][::-1][0]
    trueMat[j,:] = np.ma.masked_where(testMat[i,:] < max_proba*0.0000000000000001, trueMat[j,:]).filled(fill_value=-999)
    #p = max_items_2(trueMat[j,:].argsort()[-12:][::-1], testMat[i,:].argsort()[-12:][::-1], testMat[i,:])
    p = max_items(trueMat[j,:].argsort()[-7:][::-1], trueMat[j,:])
    print(len(p))
    pred.append(" ".join(p))
    i += 1
    
test['added_products'] = pred
test.to_csv('sub.csv', index=False)





