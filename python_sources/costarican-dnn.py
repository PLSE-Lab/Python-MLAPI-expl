#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_path = '../input'
train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
train_data.head(3)


# <TABLE>
#     <TR><TD>v2a1</TD><TD>Monthly rent payment</TD></TR>
#     <TR><TD>hacdor</TD><TD>=1 Overcrowding by bedrooms</TD></TR>
#     <TR><TD>rooms</TD><TD> number of all rooms in the house</TD></TR>
#     <TR><TD>hacapo</TD><TD>=1 Overcrowding by rooms</TD></TR>
#     <TR><TD>v14a</TD><TD>=1 has bathroom in the household</TD></TR>
#     <TR><TD>refrig</TD><TD>=1 if the household has refrigerator</TD></TR>
#     <TR><TD>v18q</TD><TD>owns a tablet</TD></TR>
#     <TR><TD>v18q1</TD><TD>number of tablets household owns</TD></TR>
#     <TR><TD>r4h1</TD><TD>Males younger than 12 years of age</TD></TR>
#     <TR><TD>r4h2</TD><TD>Males 12 years of age and older</TD></TR>
#     <TR><TD>r4h3</TD><TD>Total males in the household</TD></TR>
#     <TR><TD>r4m1</TD><TD>Females younger than 12 years of age</TD></TR>
#     <TR><TD>r4m2</TD><TD>Females 12 years of age and older</TD></TR>
#     <TR><TD>r4m3</TD><TD>Total females in the household</TD></TR>
#     <TR><TD>r4t1</TD><TD>persons younger than 12 years of age</TD></TR>
#     <TR><TD>r4t2</TD><TD>persons 12 years of age and older</TD></TR>
#     <TR><TD>r4t3</TD><TD>Total persons in the household</TD></TR>
#     <TR><TD>tamhog</TD><TD>size of the household</TD></TR>
#     <TR><TD>tamviv</TD><TD>number of persons living in the household</TD></TR>
#     <TR><TD>escolari</TD><TD>years of schooling</TD></TR>
#     <TR><TD>rez_esc</TD><TD>Years behind in school</TD></TR>
#     <TR><TD>hhsize</TD><TD>household size</TD></TR>
#     <TR><TD>paredblolad</TD><TD>=1 if predominant material on the outside wall is block or brick</TD></TR>
#     <TR><TD>paredzocalo</TD><TD>"=1 if predominant material on the outside wall is socket (wood, zinc or absbesto"</TD></TR>
#     <TR><TD>paredpreb</TD><TD>=1 if predominant material on the outside wall is prefabricated or cement</TD></TR>
#     <TR><TD>pareddes</TD><TD>=1 if predominant material on the outside wall is waste material</TD></TR>
#     <TR><TD>paredmad</TD><TD>=1 if predominant material on the outside wall is wood</TD></TR>
#     <TR><TD>paredzinc</TD><TD>=1 if predominant material on the outside wall is zink</TD></TR>
#     <TR><TD>paredfibras</TD><TD>=1 if predominant material on the outside wall is natural fibers</TD></TR>
#     <TR><TD>paredother</TD><TD>=1 if predominant material on the outside wall is other</TD></TR>
#     <TR><TD>pisomoscer</TD><TD>"=1 if predominant material on the floor is mosaic, ceramic, terrazo"</TD></TR>
#     <TR><TD>pisocemento</TD><TD>=1 if predominant material on the floor is cement</TD></TR>
#     <TR><TD>pisoother</TD><TD>=1 if predominant material on the floor is other</TD></TR>
#     <TR><TD>pisonatur</TD><TD>=1 if predominant material on the floor is  natural material</TD></TR>
#     <TR><TD>pisonotiene</TD><TD>=1 if no floor at the household</TD></TR>
#     <TR><TD>pisomadera</TD><TD>=1 if predominant material on the floor is wood</TD></TR>
#     <TR><TD>techozinc</TD><TD>=1 if predominant material on the roof is metal foil or zink</TD></TR>
#     <TR><TD>techoentrepiso</TD><TD>"=1 if predominant material on the roof is fiber cement, mezzanine "</TD></TR>
#     <TR><TD>techocane</TD><TD>=1 if predominant material on the roof is natural fibers</TD></TR>
#     <TR><TD>techootro</TD><TD>=1 if predominant material on the roof is other</TD></TR>
#     <TR><TD>cielorazo</TD><TD>=1 if the house has ceiling</TD></TR>
#     <TR><TD>abastaguadentro</TD><TD>=1 if water provision inside the dwelling</TD></TR>
#     <TR><TD>abastaguafuera</TD><TD>=1 if water provision outside the dwelling</TD></TR>
#     <TR><TD>abastaguano</TD><TD>=1 if no water provision</TD></TR>
#     <TR><TD>public</TD><TD>"=1 electricity from CNFL, ICE, ESPH/JASEC"</TD></TR>
#     <TR><TD>planpri</TD><TD>=1 electricity from private plant</TD></TR>
#     <TR><TD>noelec</TD><TD>=1 no electricity in the dwelling</TD></TR>
#     <TR><TD>coopele</TD><TD>=1 electricity from cooperative</TD></TR>
#     <TR><TD>sanitario1</TD><TD>=1 no toilet in the dwelling</TD></TR>
#     <TR><TD>sanitario2</TD><TD>=1 toilet connected to sewer or cesspool</TD></TR>
#     <TR><TD>sanitario3</TD><TD>=1 toilet connected to  septic tank</TD></TR>
#     <TR><TD>sanitario5</TD><TD>=1 toilet connected to black hole or letrine</TD></TR>
#     <TR><TD>sanitario6</TD><TD>=1 toilet connected to other system</TD></TR>
#     <TR><TD>energcocinar1</TD><TD>=1 no main source of energy used for cooking (no kitchen)</TD></TR>
#     <TR><TD>energcocinar2</TD><TD>=1 main source of energy used for cooking electricity</TD></TR>
#     <TR><TD>energcocinar3</TD><TD>=1 main source of energy used for cooking gas</TD></TR>
#     <TR><TD>energcocinar4</TD><TD>=1 main source of energy used for cooking wood charcoal</TD></TR>
#     <TR><TD>elimbasu1</TD><TD>=1 if rubbish disposal mainly by tanker truck</TD></TR>
#     <TR><TD>elimbasu2</TD><TD>=1 if rubbish disposal mainly by botan hollow or buried</TD></TR>
#     <TR><TD>elimbasu3</TD><TD>=1 if rubbish disposal mainly by burning</TD></TR>
#     <TR><TD>elimbasu4</TD><TD>=1 if rubbish disposal mainly by throwing in an unoccupied space</TD></TR>
#     <TR><TD>elimbasu5</TD><TD>"=1 if rubbish disposal mainly by throwing in river, creek or sea"</TD></TR>
#     <TR><TD>elimbasu6</TD><TD>=1 if rubbish disposal mainly other</TD></TR>
#     <TR><TD>epared1</TD><TD>=1 if walls are bad</TD></TR>
#     <TR><TD>epared2</TD><TD>=1 if walls are regular</TD></TR>
#     <TR><TD>epared3</TD><TD>=1 if walls are good</TD></TR>
#     <TR><TD>etecho1</TD><TD>=1 if roof are bad</TD></TR>
#     <TR><TD>etecho2</TD><TD>=1 if roof are regular</TD></TR>
#     <TR><TD>etecho3</TD><TD>=1 if roof are good</TD></TR>
#     <TR><TD>eviv1</TD><TD>=1 if floor are bad</TD></TR>
#     <TR><TD>eviv2</TD><TD>=1 if floor are regular</TD></TR>
#     <TR><TD>eviv3</TD><TD>=1 if floor are good</TD></TR>
#     <TR><TD>dis</TD><TD>=1 if disable person</TD></TR>
#     <TR><TD>male</TD><TD>=1 if male</TD></TR>
#     <TR><TD>female</TD><TD>=1 if female</TD></TR>
#     <TR><TD>estadocivil1</TD><TD>=1 if less than 10 years old</TD></TR>
#     <TR><TD>estadocivil2</TD><TD>=1 if free or coupled uunion</TD></TR>
#     <TR><TD>estadocivil3</TD><TD>=1 if married</TD></TR>
#     <TR><TD>estadocivil4</TD><TD>=1 if divorced</TD></TR>
#     <TR><TD>estadocivil5</TD><TD>=1 if separated</TD></TR>
#     <TR><TD>estadocivil6</TD><TD>=1 if widow/er</TD></TR>
#     <TR><TD>estadocivil7</TD><TD>=1 if single</TD></TR>
#     <TR><TD>parentesco1</TD><TD>=1 if household head</TD></TR>
#     <TR><TD>parentesco2</TD><TD>=1 if spouse/partner</TD></TR>
#     <TR><TD>parentesco3</TD><TD>=1 if son/doughter</TD></TR>
#     <TR><TD>parentesco4</TD><TD>=1 if stepson/doughter</TD></TR>
#     <TR><TD>parentesco5</TD><TD>=1 if son/doughter in law</TD></TR>
#     <TR><TD>parentesco6</TD><TD>=1 if grandson/doughter</TD></TR>
#     <TR><TD>parentesco7</TD><TD>=1 if mother/father</TD></TR>
#     <TR><TD>parentesco8</TD><TD>=1 if father/mother in law</TD></TR>
#     <TR><TD>parentesco9</TD><TD>=1 if brother/sister</TD></TR>
#     <TR><TD>parentesco10</TD><TD>=1 if brother/sister in law</TD></TR>
#     <TR><TD>parentesco11</TD><TD>=1 if other family member</TD></TR>
#     <TR><TD>parentesco12</TD><TD>=1 if other non family member</TD></TR>
#     <TR><TD>idhogar</TD><TD>Household level identifier</TD></TR>
#     <TR><TD>hogar_nin</TD><TD>Number of children 0 to 19 in household</TD></TR>
#     <TR><TD>hogar_adul</TD><TD>Number of adults in household</TD></TR>
#     <TR><TD>hogar_mayor</TD><TD># of individuals 65+ in the household</TD></TR>
#     <TR><TD>hogar_total</TD><TD># of total individuals in the household</TD></TR>
#     <TR><TD>dependency</TD><TD>Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)</TD></TR>
#     <TR><TD>edjefe</TD><TD>years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0</TD></TR>
#     <TR><TD>edjefa</TD><TD>years of education of female head of household, based on the interaction of escolari (years of education), >head of household and gender, yes=1 and no=0</TD></TR>
#     <TR><TD>meaneduc</TD><TD>average years of education for adults (18+)</TD></TR>
#     <TR><TD>instlevel1</TD><TD>=1 no level of education</TD></TR>
#     <TR><TD>instlevel2</TD><TD>=1 incomplete primary</TD></TR>
#     <TR><TD>instlevel3</TD><TD>=1 complete primary</TD></TR>
#     <TR><TD>instlevel4</TD><TD>=1 incomplete academic secondary level</TD></TR>
#     <TR><TD>instlevel5</TD><TD>=1 complete academic secondary level</TD></TR>
#     <TR><TD>instlevel6</TD><TD>=1 incomplete technical secondary level</TD></TR>
#     <TR><TD>instlevel7</TD><TD>=1 complete technical secondary level</TD></TR></TD></TR>
#     <TR><TD>instlevel8</TD><TD>=1 undergraduate and higher education</TD></TR>
#     <TR><TD>instlevel9</TD><TD>=1 postgraduate higher education</TD></TR>
#     <TR><TD>bedrooms</TD><TD>number of bedrooms</TD></TR>
#     <TR><TD>overcrowding</TD><TD># persons per room</TD></TR>
#     <TR><TD>tipovivi1</TD><TD>=1 own and fully paid house</TD></TR>
#     <TR><TD>tipovivi2</TD><TD>"=1 own, paying in installments"</TD></TR>
#     <TR><TD>tipovivi3</TD><TD>=1 rented</TD></TR>
#     <TR><TD>tipovivi4</TD><TD>=1 precarious</TD></TR>
#     <TR><TD>tipovivi5</TD><TD>"=1 other(assigned, borrowed)"</TD></TR>
#     <TR><TD>computer</TD><TD>=1 if the household has notebook or desktop computer</TD></TR>
#     <TR><TD>television</TD><TD>=1 if the household has TV</TD></TR>
#     <TR><TD>mobilephone</TD><TD>=1 if mobile phone</TD></TR>
#     <TR><TD>qmobilephone</TD><TD># of mobile phones</TD></TR>
#     <TR><TD>lugar1</TD><TD>=1 region Central</TD></TR>
#     <TR><TD>lugar2</TD><TD>=1 region Chorotega</TD></TR>
#     <TR><TD>lugar3</TD><TD>=1 region PacÃ­fico central</TD></TR>
#     <TR><TD>lugar4</TD><TD>=1 region Brunca</TD></TR>
#     <TR><TD>lugar5</TD><TD>=1 region Huetar AtlÃ¡ntica</TD></TR>
#     <TR><TD>lugar6</TD><TD>=1 region Huetar Norte</TD></TR>
#     <TR><TD>area1</TD><TD>=1 zona urbana</TD></TR>
#     <TR><TD>area2</TD><TD>=2 zona rural</TD></TR>
#     <TR><TD>age</TD><TD>Age in years</TD></TR>
#     <TR><TD>SQBescolari</TD><TD>escolari squared</TD></TR>
#     <TR><TD>SQBage</TD><TD>age squared</TD></TR>
#     <TR><TD>SQBhogar_total</TD><TD>hogar_total squared</TD></TR>
#     <TR><TD>SQBedjefe</TD><TD>edjefe squared</TD></TR>
#     <TR><TD>SQBhogar_nin</TD><TD>hogar_nin squared</TD></TR>
#     <TR><TD>SQBovercrowding</TD><TD>overcrowding squared</TD></TR>
#     <TR><TD>SQBdependency</TD><TD>dependency squared</TD></TR>
#     <TR><TD>SQBmeaned</TD><TD>square of the mean years of education of adults (>=18) in the household</TD></TR>
#     <TR><TD>agesq</TD><TD>Age squared</TD></TR>
# </TABLE>

# In[ ]:


tablets = []
for owns, num_tablets in zip(train_data['v18q'], train_data['v18q1']):
    if owns == 0:
        tablets += [0]
    else:
        tablets += [num_tablets]
        
train_data['v18q1'] = tablets


# In[ ]:


train_data.head(3)

v2a1, rez_esc, meaneduc, SQBmeaned has NaN values. edjefe, edjefa has object values.  Nan value in meaneduc and SQBmeaned can fill by edjefe and edjefa. dependency has object values. dependency = no to 0, yes to mean value can be used.
Id and idhoger can be dropped.
# In[ ]:


tmp_educ = []
sq_tmp_educ = []

for efe, efa, meduc, sq in zip(train_data['edjefe'], train_data['edjefa'], train_data['meaneduc'], train_data['SQBmeaned']):
    new_educ = meduc
    if new_educ != new_educ:
        if efa == "no":
            if efe == "no":
                new_educ = 0.0
            else:
                new_educ = float(efe)
        else:
            if efe == "no":
                new_educ = float(efa)
            else:
                new_educ = float(efe) + float(efa)
    if meduc != new_educ:
        print(meduc, ",", new_educ)
    tmp_educ += [new_educ]

    sq_tmp_educ += [new_educ ** 2]
        
train_data['meaneduc'] = tmp_educ
train_data['SQBmeaned'] = sq_tmp_educ


# In[ ]:


from sklearn.preprocessing import Imputer

v2a1 = []
rez_esc = []
for rentpay, rez in zip(train_data['v2a1'], train_data['rez_esc']):
    if rentpay != rentpay:
        v2a1 += [0]
    else:
        v2a1 += [rentpay]

    if rez != rez:
        rez_esc += [0]
    else:
        rez_esc += [rez]

        
#train_data['v2a1'] = v2a1
train_data['rez_esc'] = rez_esc


# In[ ]:


train_data.info(verbose=True, null_counts=True)


# In[ ]:


depend = []
for dependency, children, olds, total in zip(train_data['dependency'], train_data['hogar_nin'], train_data['hogar_mayor'], train_data['hogar_total']):
    calc_depend = False
    if depend != depend:
        calc_depend = True
    elif (dependency == "yes" or dependency == "no"):
        calc_depend = True

    #Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
    if calc_depend:
        i = (children + olds) / (total - children - olds)
    else:
        i = float(dependency)

    depend += [i]

train_data['dependency'] = depend

chw = []
for nin, adul in zip(train_data['hogar_nin'], train_data['hogar_adul']):
    if adul == 0:
        chw += [nin * 2]
    else:
        chw += [nin / adul]

train_data['child_weight'] = (train_data['hogar_nin'] + train_data['hogar_mayor']) / train_data['hogar_total']
train_data['child_weight2'] = chw
train_data['child_weight3'] = train_data['r4t1'] / train_data['r4t3']
train_data['work_power'] = train_data['dependency'] * train_data['hogar_adul']
train_data['SQBworker'] = train_data['hogar_adul'] ** 2
train_data['rooms_per_person'] = train_data['rooms'] / (train_data['tamviv'])
train_data['bedrooms_per_room'] = train_data['bedrooms'] / train_data['rooms']
train_data['female_weight'] = train_data['r4m3'] / train_data['r4t3']


# In[ ]:


#Predict v2a1 for household.
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

v2a1_drop = ['edjefe', 'edjefa', 'v2a1', 'Id']
train_hh = train_data.query('parentesco1 == 1').drop('Target', axis=1)
v2a1_train_tmp = train_hh.query('v2a1 == v2a1')
v2a1_train = v2a1_train_tmp.drop(v2a1_drop, axis=1)
v2a1_train = v2a1_train.drop('idhogar', axis=1)
v2a1_train_target = v2a1_train_tmp['v2a1'].copy()
#std_scaler = StandardScaler()
#std_scaler.fit(v2a1_train)
#v2a1_train = std_scaler.transform(v2a1_train)

v2a1_test_tmp = train_hh.query('v2a1 != v2a1').drop(v2a1_drop, axis=1)
v2a1_test = v2a1_test_tmp.drop('idhogar', axis=1)
forest = GradientBoostingRegressor(n_estimators=400, learning_rate=0.2, max_depth=5, random_state=0)
forest.fit(v2a1_train, v2a1_train_target)
print("score: ", forest.score(v2a1_train, v2a1_train_target))
v2a1_train_pred = forest.predict(v2a1_train)
forest_mse = mean_squared_error(v2a1_train_target, v2a1_train_pred)
print("RMSE: ", np.sqrt(forest_mse))


# In[ ]:



v2a1_pred = pd.DataFrame({'ID': v2a1_test_tmp['idhogar'], 'v2a1_pred': forest.predict(v2a1_test)})

v2a1 = []
for rent, idhh in zip(train_data['v2a1'], train_data['idhogar']):
    if rent != rent:
        i = 0
        for rent_p, index in zip(v2a1_pred['v2a1_pred'], v2a1_pred['ID']):
            if index == idhh:
                i = rent_p
                break
        if i == 0:
            for rent_org, index in zip(v2a1_train_target, v2a1_train_tmp['idhogar']):
                if index == idhh:
                    i = rent_org
        if i < 0:
            i = 0
        v2a1 += [i]
    else:
        v2a1 += [rent]

train_data['v2a1'] = v2a1


# In[ ]:


train_data['hogar_total'].value_counts()


# In[ ]:


corr_matrix = train_data.corr()
corr_matrix['Target'].sort_values(ascending=False)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train = train_data.drop('Target', axis=1)
y_train = train_data['Target'].copy()

#X_train, X_validate, y_train, y_validate = train_test_split(X, y, random_state=0)


# In[ ]:


X_train = X_train.drop(['Id', 'idhogar', 'edjefe', 'edjefa'], axis=1)


# In[ ]:


#data augmentation
from sklearn.utils import shuffle

X_train_da = pd.concat([X_train, X_train, X_train])
#X_train_da = pd.concat([X_train_da, X_train])

#X_train['Noise1'] = np.random.rand()
#X_train['Noise2'] = np.random.rand()
X_train_da['age'] += np.random.randint(2) + np.random.randint(2) - 2
X_train_da['SQBage'] = X_train_da['age'] ** 2
X_train_da['hogar_total'] += np.random.randint(3) - 1
X_train_da['SQBhogar_total'] = X_train['hogar_total'] ** 2
X_train_da['v2a1'] += np.random.randint(10) * 1000 - 5000

X_train_da = pd.concat([X_train, X_train_da])

y_train = pd.concat([y_train, y_train, y_train, y_train])
#y_train = pd.concat([y_train, y_train])

X_train_da, y_train = shuffle(X_train_da, y_train)


# In[ ]:


X_train_dummy = pd.get_dummies(X_train_da)


# In[ ]:


X_train_dummy.info(verbose=True, null_counts=True)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='mean')
imputer.fit(X_train_dummy)
X_train_dummy = imputer.transform(X_train_dummy)

scaler = MinMaxScaler()
scaler.fit(X_train_dummy)
X_train_scaled = scaler.transform(X_train_dummy)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers

n_classes = 5

y_train_keras = keras.utils.to_categorical(y_train, n_classes)

(n_samples, n_features) = X_train_scaled.shape

model = Sequential()
model.add(Dense(units=500, activation="relu", input_shape=(n_features, )))
model.add(Dropout(0.2))
model.add(Dense(units=500, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=500, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=n_classes, activation="softmax"))
          
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train_keras, epochs=100, validation_split=0.1, batch_size=n_samples, verbose=0)
y_train_pred = model.predict_classes(X_train_scaled, verbose=0)
f1_score(y_train, y_train_pred, average='macro')


# In[ ]:


plt.ylim(0, max(np.r_[history.history['val_loss'], history.history['loss']]))
plt.plot(history.history['val_loss'], c='red')
plt.plot(history.history['loss'], c='green')


# In[ ]:


#X_validate = X_validate.drop(['Id', 'idhogar', 'edjefe', 'edjefa'], axis=1)
#X_validate['Noise1'] = 0
#X_validate['Noise2'] = 0
#X_validate_dummy = pd.get_dummies(X_validate)
#X_validate_dummy = imputer.transform(X_validate_dummy)
#X_validate_scaled = scaler.transform(X_validate_dummy)


# In[ ]:



#print("score: ", mlp.score(X_validate_scaled, y_validate))
#y_validate_pred = mlp.predict(X_validate_scaled)

#y_validate_pred = model.predict_classes(X_validate_scaled, verbose=0)

#f1_score(y_validate, y_validate_pred, average='macro')


# In[ ]:


test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))

tablets = []
for owns, num_tablets in zip(test_data['v18q'], test_data['v18q1']):
    if owns == 0:
        tablets += [0]
    else:
        tablets += [num_tablets]
        
test_data['v18q1'] = tablets

tmp_educ = []
sq_tmp_educ = []

for efe, efa, meduc, sq in zip(test_data['edjefe'], test_data['edjefa'], test_data['meaneduc'], test_data['SQBmeaned']):
    new_educ = meduc
    if new_educ != new_educ:
        if efa == "no":
            if efe == "no":
                new_educ = 0.0
            else:
                new_educ = float(efe)
        else:
            if efe == "no":
                new_educ = float(efa)
            else:
                new_educ = float(efe) + float(efa)
    tmp_educ += [new_educ]

    sq_tmp_educ += [new_educ ** 2]
        
test_data['meaneduc'] = tmp_educ
test_data['SQBmeaned'] = sq_tmp_educ

v2a1 = []
rez_esc = []
for rentpay, rez in zip(test_data['v2a1'], test_data['rez_esc']):
    if rentpay != rentpay:
        v2a1 += [0]
    else:
        v2a1 += [rentpay]

    if rez != rez:
        rez_esc += [0]
    else:
        rez_esc += [rez]

#test_data['v2a1'] = v2a1
test_data['rez_esc'] = rez_esc

depend = []
for dependency, children, olds, total in zip(test_data['dependency'], test_data['hogar_nin'], test_data['hogar_mayor'], test_data['hogar_total']):
    calc_depend = False
    if depend != depend:
        calc_depend = True
    elif (dependency == "yes" or dependency == "no"):
        calc_depend = True

    #Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
    if calc_depend:
        i = (children + olds) / (total - children - olds)
    else:
        i = float(dependency)

    depend += [i]

test_data['dependency'] = depend

chw = []
for nin, adul in zip(test_data['hogar_nin'], test_data['hogar_adul']):
    if adul == 0:
        chw += [nin * 2]
    else:
        chw += [nin/adul]

test_data['child_weight'] = (test_data['hogar_nin'] + test_data['hogar_mayor']) / test_data['hogar_total']
test_data['child_weight2'] = chw
test_data['child_weight3'] = test_data['r4t1'] / test_data['r4t3']
test_data['work_power'] = test_data['dependency'] * test_data['hogar_adul']
test_data['SQBworker'] = test_data['hogar_adul'] ** 2
test_data['rooms_per_person'] = test_data['rooms'] / (test_data['tamviv'])
test_data['bedrooms_per_room'] = test_data['bedrooms'] / test_data['rooms']
test_data['female_weight'] = test_data['r4m3'] / test_data['r4t3']

train_hh = test_data.query('parentesco1 == 1')
v2a1_test_tmp = train_hh.query('v2a1 != v2a1').drop(v2a1_drop, axis=1)
v2a1_test = v2a1_test_tmp.drop('idhogar', axis=1)

(num_target, num_col) = v2a1_test.shape

if (num_target > 0):
    v2a1_pred = pd.DataFrame({'ID': v2a1_test_tmp['idhogar'], 'v2a1_pred': forest.predict(v2a1_test)})

    v2a1 = []
    for rent, idhh in zip(test_data['v2a1'], test_data['idhogar']):
        if rent != rent:
            i = 0
            for rent_p, index in zip(v2a1_pred['v2a1_pred'], v2a1_pred['ID']):
                if index == idhh:
                    i = rent_p
                    break
            if i == 0:
                for rent_org, index in zip(v2a1_train_target, v2a1_train_tmp['idhogar']):
                    if index == idhh:
                        i = rent_org
            if i < 0:
                i = 0
            v2a1 += [i]
        else:
             v2a1 += [rent]

    test_data['v2a1'] = v2a1


test_data_drop = test_data.drop(['Id', 'idhogar', 'edjefe', 'edjefa'], axis=1)
#test_data_drop['Noise1'] = np.random.rand()
#test_data_drop['Noise2'] = 0
test_data_dummy = pd.get_dummies(test_data_drop)
test_data_dummy = imputer.transform(test_data_dummy)
test_data_scaled = scaler.transform(test_data_dummy)


# In[ ]:


#y_pred = mlp.predict(test_data_scaled)
y_pred = model.predict_classes(test_data_scaled, verbose=0)


# In[ ]:


result = pd.DataFrame({'Id':test_data['Id'], 'Target':y_pred})
result.to_csv('result1.csv', index=False)

