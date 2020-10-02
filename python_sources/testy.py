#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import timeit

df = pd.read_csv('.\train.csv', sep=',')

random_state = 5

removeDied   = False
sortDatetime = False
doGraphs = True
doFinalOutput = False

# Prep the data

def DataMunge_NullRecords(df):
    # remove rows with nulls in the columns
    # there aren't many records removed by this so it seems worthwhile
    oldrowcount = df.shape[0]
    df = df[pd.notnull(df['SexuponOutcome'])]
    print "...Removed %i rows for null entries" % int(oldrowcount - df.shape[0])
    return df

def DataMunge_Died(df):
    if removeDied and 'OutcomeType' in df.columns:
        oldrowcount = df.shape[0]
        df = df.loc[df['OutcomeType'] != "Died"]
        print "...Removed %i rows for 'Died' entries" % int(oldrowcount - df.shape[0])
    return df

def DataMunge_SexData(df):
    # manage the Sex data
    df['SexuponOutcome'] = df['SexuponOutcome'].str.lower()
    df['Sex_Female'] = df.SexuponOutcome.str.contains(r'\bfemale')
    df['Sex_Fixed'] = df.SexuponOutcome.str.contains(r'spayed|neutered')
    df['Sex_Unknown'] = df.SexuponOutcome.str.contains(r'\bunknown')
    print "...Converted Sex attributes"
    return df

def DataMunge_NameLength(df):
#    df['Unnamed'] = df['Name'].isnull()
    df['NameLen'] = df['Name'].str.len()
    df['NameLen'].fillna(0, inplace=True)
    df['NameLen'] = df['NameLen'] / df['NameLen'].max() # normalize to 0-1.0
    print "...Converted Name Length attributes"
    return df
    
def DataMunge_Color(df):
    df['MixedColor'] = df.Color.str.contains('/')
    print "...Converted Color attributes"
    return df

def DataMunge_BreedHair(df):
    # manage the Breed data
    df['Breed'] = df['Breed'].str.lower() # convert all to lower case
    
    df['Hair_Short'] = df['Breed'].str.contains('shorthair')
#    df['Hair_Med'] = df['Breed'].str.contains('medium hair')
#    df['Hair_Long'] = df['Breed'].str.contains('longhair')

    df['Breed'] = df['Breed'].str.replace(' shorthair','')
    df['Breed'] = df['Breed'].str.replace(' medium hair','')
    df['Breed'] = df['Breed'].str.replace(' longhair','')
    df['Breed'] = df['Breed'].str.replace(' semi-longhair','')
    df['Breed'] = df['Breed'].str.replace(' wirehair','')
    df['Breed'] = df['Breed'].str.replace(' rough','')
    df['Breed'] = df['Breed'].str.replace(' smooth coat','')
    df['Breed'] = df['Breed'].str.replace(' smooth','')
    df['Breed'] = df['Breed'].str.replace(' black/tan','')
    df['Breed'] = df['Breed'].str.replace('black/tan ','')
    df['Breed'] = df['Breed'].str.replace(' flat coat','')
    df['Breed'] = df['Breed'].str.replace('flat coat ','')
    df['Breed'] = df['Breed'].str.replace(' coat','')
    
    # add a 'mix' column
    df['BreedMixUnknown'] = 0
    df['BreedMixUnknown'] = df['Breed'].str.contains(' mix')
    df['Breed'] = df['Breed'].str.replace(' mix','')
#    df['BreedMixKnown'] = 0
#    df['BreedMixKnown'] = df['Breed'].str.contains('/')
#    df['MixedBreed'] = df['Breed'].str.contains(' mix') | df['Breed'].str.contains('/')
    
    # I should make this more robust
    breeds = df['Breed'].str.split(pat="/", expand=True)
    df['Breed1'] = breeds[0]
    df['Breed2'] = breeds[1]
    return df

def DataMunge_CatBreeds(df):
    # now categorize the cats
    # this is somewhat arbitrary based on frequency
    if df['AnimalType'].str.contains('Cat').any():
        df['Cat_Domestic']  = df['Breed'].str.contains('domestic')  & df['AnimalType'].str.contains('Cat')
        df['Cat_Siamese']   = df['Breed'].str.contains('siamese')   & df['AnimalType'].str.contains('Cat')
#        df['Cat_Snowshoe']  = df['Breed'].str.contains('snowshoe')  & df['AnimalType'].str.contains('Cat')
#        df['Cat_MaineCoon'] = df['Breed'].str.contains('maine coon')& df['AnimalType'].str.contains('Cat')
#        df['Cat_Manx']      = df['Breed'].str.contains('manx')      & df['AnimalType'].str.contains('Cat')
#        df['Cat_Himalayan'] = df['Breed'].str.contains('himalayan') & df['AnimalType'].str.contains('Cat')
#        df['Cat_Persian']   = df['Breed'].str.contains('persian')   & df['AnimalType'].str.contains('Cat')
    return df

def DefineDogBreeds():
    # Assigns the dog breeds to AKC breeds, with some manual adjustments
    breeds = ['Blue Lacy','Queensland Heeler','Rhod Ridgeback','Retriever','Chinese Sharpei','Black Mouth Cur','Catahoula','Staffordshire','Affenpinscher','Afghan Hound','Airedale Terrier','Akita','Australian Kelpie','Alaskan Malamute','English Bulldog','American Bulldog','American English Coonhound','American Eskimo Dog (Miniature)','American Eskimo Dog (Standard)','American Eskimo Dog (Toy)','American Foxhound','American Hairless Terrier','American Staffordshire Terrier','American Water Spaniel','Anatolian Shepherd Dog','Australian Cattle Dog','Australian Shepherd','Australian Terrier','Basenji','Basset Hound','Beagle','Bearded Collie','Beauceron','Bedlington Terrier','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Bernese Mountain Dog','Bichon Frise','Black and Tan Coonhound','Black Russian Terrier','Bloodhound','Bluetick Coonhound','Boerboel','Border Collie','Border Terrier','Borzoi','Boston Terrier','Bouvier des Flandres','Boxer','Boykin Spaniel','Briard','Brittany','Brussels Griffon','Bull Terrier','Bull Terrier (Miniature)','Bulldog','Bullmastiff','Cairn Terrier','Canaan Dog','Cane Corso','Cardigan Welsh Corgi','Cavalier King Charles Spaniel','Cesky Terrier','Chesapeake Bay Retriever','Chihuahua','Chinese Crested Dog','Chinese Shar Pei','Chinook','Chow Chow',"Cirneco dell'Etna",'Clumber Spaniel','Cocker Spaniel','Collie','Coton de Tulear','Curly-Coated Retriever','Dachshund','Dalmatian','Dandie Dinmont Terrier','Doberman Pinsch','Doberman Pinscher','Dogue De Bordeaux','English Cocker Spaniel','English Foxhound','English Setter','English Springer Spaniel','English Toy Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Finnish Spitz','Flat-Coated Retriever','French Bulldog','German Pinscher','German Shepherd','German Shorthaired Pointer','German Wirehaired Pointer','Giant Schnauzer','Glen of Imaal Terrier','Golden Retriever','Gordon Setter','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Greyhound','Harrier','Havanese','Ibizan Hound','Icelandic Sheepdog','Irish Red and White Setter','Irish Setter','Irish Terrier','Irish Water Spaniel','Irish Wolfhound','Italian Greyhound','Japanese Chin','Keeshond','Kerry Blue Terrier','Komondor','Kuvasz','Labrador Retriever','Lagotto Romagnolo','Lakeland Terrier','Leonberger','Lhasa Apso','L_wchen','Maltese','Manchester Terrier','Mastiff','Miniature American Shepherd','Miniature Bull Terrier','Miniature Pinscher','Miniature Schnauzer','Neapolitan Mastiff','Newfoundland','Norfolk Terrier','Norwegian Buhund','Norwegian Elkhound','Norwegian Lundehund','Norwich Terrier','Nova Scotia Duck Tolling Retriever','Old English Sheepdog','Otterhound','Papillon','Parson Russell Terrier','Pekingese','Pembroke Welsh Corgi','Petit Basset Griffon Vend_en','Pharaoh Hound','Plott','Pointer','Polish Lowland Sheepdog','Pomeranian','Standard Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo Pequeno','Portuguese Water Dog','Pug','Puli','Pyrenean Shepherd','Rat Terrier','Redbone Coonhound','Rhodesian Ridgeback','Rottweiler','Russell Terrier','St. Bernard','Saluki','Samoyed','Schipperke','Scottish Deerhound','Scottish Terrier','Sealyham Terrier','Shetland Sheepdog','Shiba Inu','Shih Tzu','Siberian Husky','Silky Terrier','Skye Terrier','Sloughi','Smooth Fox Terrier','Soft-Coated Wheaten Terrier','Spanish Water Dog','Spinone Italiano','Staffordshire Bull Terrier','Standard Schnauzer','Sussex Spaniel','Swedish Vallhund','Tibetan Mastiff','Tibetan Spaniel','Tibetan Terrier','Toy Fox Terrier','Treeing Walker Coonhound','Vizsla','Weimaraner','Welsh Springer Spaniel','Welsh Terrier','West Highland White Terrier','Whippet','Wire Fox Terrier','Wirehaired Pointing Griffon','Wirehaired Vizsla','Xoloitzcuintli','Yorkshire Terrier']
    groups = ['Herding','Herding','Hound','Sporting','Non-Sporting','Herding','Herding','Terrier','Toy','Hound','Terrier','Working','Working','Working','Non-Sporting','Non-Sporting','Hound','Non-Sporting','Non-Sporting','Toy','Hound','Terrier','Terrier','Sporting','Working','Herding','Herding','Terrier','Hound','Hound','Hound','Herding','Herding','Terrier','Herding','Herding','Herding','Herding','Herding','Working','Non-Sporting','Hound','Working','Hound','Hound','Working','Herding','Terrier','Hound','Non-Sporting','Herding','Working','Sporting','Herding','Sporting','Toy','Terrier','Terrier','Non-Sporting','Working','Terrier','Working','Working','Herding','Toy','Terrier','Sporting','Toy','Toy','Non-Sporting','Working','Non-Sporting','Hound','Sporting','Sporting','Herding','Non-Sporting','Sporting','Hound','Non-Sporting','Terrier','Working','Working','Working','Sporting','Hound','Sporting','Sporting','Toy','Herding','Sporting','Herding','Non-Sporting','Sporting','Non-Sporting','Working','Herding','Sporting','Sporting','Working','Terrier','Sporting','Sporting','Working','Working','Working','Hound','Hound','Toy','Hound','Herding','Sporting','Sporting','Terrier','Sporting','Hound','Toy','Toy','Non-Sporting','Terrier','Working','Working','Sporting','Sporting','Terrier','Working','Non-Sporting','Non-Sporting','Toy','Terrier','Working','Herding','Terrier','Toy','Terrier','Working','Working','Terrier','Herding','Hound','Non-Sporting','Terrier','Sporting','Herding','Hound','Toy','Terrier','Toy','Herding','Hound','Hound','Hound','Sporting','Herding','Toy','Non-Sporting','Non-Sporting','Toy','Hound','Working','Toy','Herding','Herding','Terrier','Hound','Hound','Working','Terrier','Working','Hound','Working','Non-Sporting','Hound','Terrier','Terrier','Herding','Non-Sporting','Toy','Working','Toy','Terrier','Hound','Terrier','Terrier','Herding','Sporting','Terrier','Working','Sporting','Herding','Working','Non-Sporting','Non-Sporting','Toy','Hound','Sporting','Sporting','Sporting','Terrier','Terrier','Hound','Terrier','Sporting','Sporting','Non-Sporting','Toy']
    breeds = [element.lower() for element in breeds]

    breeds.append('jack russell terrier');      groups.append('Terrier')
    breeds.append('american pit bull terrier'); groups.append('Terrier')
    breeds.append('anatol shepherd');           groups.append('Working')
    breeds.append('plott hound');               groups.append('Hound')
    breeds.append('redbone hound');             groups.append('Hound')
    breeds.append('pit bull');                  groups.append('Pit Bull')
    breeds.append('pbgv');                      groups.append('Terrier')
    breeds.append('hound');                     groups.append('Hound')
    breeds.append('softed wheaten terrier');    groups.append('Terrier')
    breeds.append('wire hair fox terrier');     groups.append('Terrier')
    breeds.append('english pointer');           groups.append('Sporting')
    breeds.append('english coonhound');         groups.append('Hound')
    breeds.append('carolina dog');              groups.append('Sporting')
    breeds.append('bruss griffon');             groups.append('Toy')
    breeds.append('bluetick hound');            groups.append('Hound')
    breeds.append('dogo argentino');            groups.append('Sporting')
    breeds.append('alaskan husky');             groups.append('Working')
    breeds.append('german pointer');            groups.append('Sporting')
    breeds.append('west highland');             groups.append('Terrier')
    breeds.append('american eskimo');           groups.append('Non-Sporting')
    breeds.append('chesa bay retr');            groups.append('Sporting')
    return breeds, groups

DogBreeds = DefineDogBreeds()

def DataMunge_DogBreeds(df):
    breeds = DogBreeds[0]; groups = DogBreeds[1]
    breed_dict = {}
    if len(breeds) != len(groups):
        print "length of 'breeds' and 'groups' mismatch!!"
    else:
        for i in range(len(breeds)):
            breed_dict[breeds[i]] = groups[i]
        df['map1'] = df.Breed1.map(breed_dict)
        df['map2'] = df.Breed2.map(breed_dict)
    
        dog_groups = np.unique(groups)
        for group in dog_groups:
            df[group] = 0
            df[group] += df['map1'] == group 
            df[group] += df['map2'] == group 
    
        for group in dog_groups:
            df[group] = df[group].replace(2,1)
    
        df.pop('map1')
        df.pop('map2')
    
        print "...Converted Breed attributes"
    return df
    
def DataMunge_Age(df):
    # normalize the ages
    zeroAge = 0.3
    df['AgeuponOutcome'].fillna('1 year', inplace=True) # deal with NaN entries
    df['Age_value'] = df['AgeuponOutcome'].str.replace(pat=r'\D',repl=r'')
    df['Age_value'] = pd.to_numeric(df['Age_value'])
    df['Age_value'] += (df['Age_value'] == 0) * zeroAge # for entries of 0 we pick the median
    
    df['Age'] = 0
    df['Age'] += df['AgeuponOutcome'].str.contains('day')  * 1.0 * (df['Age_value'] * 1)
    df['Age'] += df['AgeuponOutcome'].str.contains('week') * 1.0 * (df['Age_value'] * 7)
    df['Age'] += df['AgeuponOutcome'].str.contains('month')* 1.0 * (df['Age_value'] * 30)
    df['Age'] += df['AgeuponOutcome'].str.contains('year') * 1.0 * (df['Age_value'] * 365)
    df['Age_log'] = np.log(df['Age'])  # convert to log scale
    df['Age_log'] = df['Age_log'] / df['Age_log'].max() # normalize to 0-1.0
    df['Age'] = df['Age'] / df['Age'].max() # normalize to 0-1.0

    df['Age_category'] = 0    
    df['Age_category'] += df['AgeuponOutcome'].str.contains('year') * 0.5
    df['Age_category'] += (df['AgeuponOutcome'].str.contains('year') &            df['Age_value'] >= 6) * 1.0

    df.pop('AgeuponOutcome')
#    df.pop('Age')
    df.pop('Age_value')

    print "...Converted Age attributes"
    return df

# Converting bools to ints"
boolConvert = {True:1, False:0}

def DataMunge_Final(df):
    df = df.applymap(lambda x: boolConvert.get(x,x))
        
    df.pop('Name')
    df.pop('AnimalType')
    df.pop('Breed')
    df.pop('Breed1')
    df.pop('Breed2')
    df.pop('SexuponOutcome')
    df.pop('Color')

    if sortDatetime:
        # sort the dataframe by date (do this before splitting off the y vector)
        df.sort_values(by='DateTime',inplace=True)
        print "...Sorted Datetime"
    df.pop('DateTime')

    print "...Dropped unnecessary columns"
    return df


# this is needed to suppress warning when putting all the munges into a definition
pd.options.mode.chained_assignment = None  # default='warn'

def DoTheMunge(df):
    df = DataMunge_NullRecords(df)
    df = DataMunge_Died(df)
    df = DataMunge_SexData(df)
    df = DataMunge_NameLength(df)
    df = DataMunge_Color(df)
    df = DataMunge_BreedHair(df)
    df = DataMunge_CatBreeds(df)
    df = DataMunge_DogBreeds(df)
    df = DataMunge_Age(df)
    df = DataMunge_Final(df)
    return df
    
print; print "Now munging the TRAINING data..."
print "Starting dataframe shape is %s" % str(df.shape)
df = DoTheMunge(df)

y = df['OutcomeType']        # this is our target
df.pop('OutcomeType')        # now drop it from the inputs
df.pop('OutcomeSubtype')

print "Finished Dataframe shape is %s" % str(df.shape)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
         df.drop("AnimalID",axis=1), y, train_size=0.7, random_state = random_state)

class_labels = list(np.unique(y).astype(str))
feature_labels = list(X_train.columns.values)  # could also use list(X_train)

print
print 'Analysis stats:'
print '   Training set has %i records' % X_train.shape[0]
print '   Testing  set has %i records' % X_test.shape[0]
print '   There are %i attributes' % X_train.shape[1]

from sklearn.ensemble import GradientBoostingClassifier

n_estimator = 100

classifiers = [
    ("GradientBoosting", GradientBoostingClassifier(learning_rate=0.075,n_estimators=120,max_features=0.5,max_leaf_nodes=30, random_state=5))
    ]

if 1:
    print; print "Running classifiers..."
    results = []
    for name, clf in classifiers:
        print("Training %s ..." % name),
        segment_start = timeit.default_timer()
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        segment_end = timeit.default_timer()
        print ' (%.1f sec)' % (segment_end - segment_start)
        results.append([name, log_loss(y_test, y_prob), (segment_end - segment_start)])

    print
    print 'Classifier              log_loss time'
    print '-----------------------  ------  ----'
    for result in results:
        print "{0:25}{1:.4f}\t{2:.1f}".format(result[0], result[1], result[2])

if 1 or doGraphs:
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.plot()
    plt.barh(pos, feature_importance[sorted_idx], align='center')
#    sns.barplot(feature_importance[sorted_idx], pos, palette="BuGn_d", orient='h')
    cols = X_train.columns.values
    plt.yticks(pos, cols[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

if 0 or doFinalOutput:
    testdf = pd.read_csv('C:\Users\ATS\Desktop\Asher\AnimalShelter\\test.csv', sep=',')
    print; print "Now munging the test data..."
    print "Starting dataframe shape is %s" % str(testdf.shape)

    testdf = DoTheMunge(testdf)
    print "Finished dataframe shape is %s" % str(testdf.shape)

    print; print "Training on the FULL dataset to build final predicter..."
    clf = GradientBoostingClassifier(learning_rate=0.075,n_estimators=120,max_features=0.5,max_leaf_nodes=30, random_state=5)
    clf.fit(df.drop("AnimalID", axis=1), y)
    test_pred = clf.predict_proba(testdf.drop("ID",axis=1))

# write results to csv
if 0 or doFinalOutput:
    final = []
    header = ['ID', 'Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
#    header = []
#    header.append("AnimalID")
#    for label in class_labels:
#        header.append(label)
    final.append(header)

    row = 0    
    for pred in np.array(test_pred).tolist():
        index = testdf.iloc[[row]].index[0] # gets the df index value
        animalid = int(testdf.loc[index,"ID"]) # gets the id from the orig df
        if removeDied:
            final.append([animalid,pred[0],0.007,pred[1],pred[2],pred[3]])
        else:
            final.append([animalid,pred[0],pred[1],pred[2],pred[3],pred[4]])
        row += 1

    csvdf = pd.DataFrame(final)
    csvdf.to_csv("mysubmission.csv",index=False,header=False)

    print "Output file is written.  And we're DONE!!!"


# In[ ]:




