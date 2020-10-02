import numpy as np
import pandas as pd
import pylab as p
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC




df = pd.read_csv('../input/xAPI-Edu-Data.csv')
#My analyzes prediction first data is classified into total number of males and females with lowlevel,mediumlevel
#and high level stageid ,ofcourse sum of these variable should be total numbers of records
malelowlevel=0
malemediumlevel=0
malehighlevel=0
femalelowlevel=0
femalemediumlevel=0
femalehighlevel=0

#points given to student in raisedhands+VisITedResources+AnnouncementsView+Discussion coloums are out of 100 so the sum of these
#coloum would be 400 and 75% of 400 is 300 
#according to my logic if the sum of raisedhands+VisITedResources+AnnouncementsView+Discussion is greater than 300 student is top performer
dosum=0


#count of top performer male in lowerlevel is stored in topMlowerlevel same goes with other 5 variables
topMLowerlevel =0
topMMiddleSchool=0
topMHighSchool=0
topFLowerlevel =0
topFMiddleSchool=0
topFHighSchool=0

topperformer = []
index =2

for g,sid,raisedhands,VisITedResources,AnnouncementsView,Discussion in zip(df.gender,df.StageID,df.raisedhands,df.VisITedResources,df.AnnouncementsView,df.Discussion):
	if(g == 'M' and sid == 'lowerlevel'):
		malelowlevel=malelowlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topMLowerlevel = topMLowerlevel +1
			topperformer.append(index)
		index=index+1

			

	if(g == 'M' and sid == 'MiddleSchool'):
		malemediumlevel=malemediumlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topMMiddleSchool = topMMiddleSchool +1
			topperformer.append(index)
		index=index+1

	if(g == 'M' and sid == 'HighSchool'):
		malehighlevel=malehighlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topMHighSchool = topMHighSchool +1
			topperformer.append(index)
		index=index+1


	if(g == 'F' and sid == 'lowerlevel'):
		femalelowlevel=femalelowlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topFLowerlevel = topFLowerlevel +1
			topperformer.append(index)
		index=index+1

	if(g == 'F' and sid == 'MiddleSchool'):
		femalemediumlevel=femalemediumlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topFMiddleSchool = topFMiddleSchool +1
			topperformer.append(index)
		index=index+1

	if(g == 'F' and sid == 'HighSchool'):
		femalehighlevel=femalehighlevel+1
		dosum=0
		dosum = raisedhands+VisITedResources+AnnouncementsView+Discussion
		if(dosum >= 300):
			topFHighSchool = topFHighSchool +1
			topperformer.append(index)
		index=index+1

	
print("These are the top performer from my side:")
print("\n")
print(topperformer)
print("\n")
print("Further data is classified into how much top performer are in each category")
print("\n")
print("Analyzes: Male with lowlevel:","there are top",topMLowerlevel,"Performer Students out of",malelowlevel)
print("Analyzes: Male with Mediumlevel:","there are top",topMMiddleSchool,"Performer Students out of",malemediumlevel)
print("Analyzes: Male with Highlevel:","there are top",topMHighSchool,"Performer Students out of",malehighlevel)
print("Analyzes: Female with lowlevel:","there are top",topFLowerlevel,"Performer Students out of",femalelowlevel)
print("Analyzes: female with Mediumlevel:","there are top",topFMiddleSchool,"Performer Students out of",femalemediumlevel)
print("Analyzes: female with Highlevel:","there are top",topFHighSchool,"Performer Students out of",femalehighlevel)




#LogisticRegression
subset_df = df[['gender', 'raisedhands','VisITedResources','Discussion','AnnouncementsView', 'StudentAbsenceDays','Class']]
#make a column find where class =High put as 1 , where class =Low and class =Medium put as 0
subset_df['Performance'] = subset_df.Class.map({'H':1, 'L':0, 'M': 0})
#where gender = female put 0 and if male put 1
subset_df['IsMale'] = subset_df.gender.map({'F':0, 'M': 1})
#where student absences is above7 put as 1 and under 7 put as 0
subset_df['MoreThan7Absences'] = subset_df.StudentAbsenceDays.map({'Above-7':1, 'Under-7': 0})
subset_df = subset_df.drop(['Class','StudentAbsenceDays','gender'],axis=1)
#now we have converted categorical data into numerical
temp_set = subset_df
temp_set = temp_set.drop(['Performance'],axis=1)
X = temp_set.values
y = subset_df.Performance.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1337)
#X_train , X_test contains raisedhands,visit.r,discussion,AnnouncementsView,ismale and morethan7absences values
#y_train ,y_test contains performances values

reg = LogisticRegression()
y_pred = reg.fit(X_train, y_train).predict(X_test)
print("Score of logistic regression",reg.score(X_test,y_test)) #around 0.775
print("Prediction matrix of logistic regression")
print(reg.predict(X))
#prediction matrix tells us that 7 last student from down is included in top performer
#which has raisedhands= 85 visitresources =88 discussion =79 and announcmentview =70 
#absences under 7 and a medium school student this data shows us that student will definelty a 
#good performer
#reference link:http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

print("\n")


#Randomforest
#we do use same training and testing set for random forest
radm = RandomForestClassifier()
radm.fit(X_train,y_train)
print("Score of Random forest",radm.score(X_test,y_test)) #around 0.7 - 0.8
print("Prediction matrix of Random forest")
print(radm.predict(X))
#according to randomforest prediction matrix tells us that 13 last from down is included in top performer
#if we check the values in csv file so 87,93,63,60 under 7 means 1 and High class means 1
#reference link:http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit

print("\n")

#SVM
#we do use same training and testing set for SVM
svm = SVC()
svm.fit(X_train,y_train)
print("Score of SVM",svm.score(X_test,y_test)) #around 0.691
print("Prediction matrix of SVM")
print(svm.predict(X))
#according to SVM predication matrix tells us that 14 from last down is included in top performer
#if we check the values in csv file so 80,82,64,58 under7 absences means 1 and high class means 1 which is not bad result
#reference link http://scikit-learn.org/stable/modules/svm.html