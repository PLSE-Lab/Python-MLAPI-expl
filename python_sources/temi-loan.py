# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from flask import Flask, render_template,session,redirect,url_for,request
from flask_wtf import FlaskForm
from wtforms import StringField,BooleanField,DateTimeField,RadioField,SelectField,TextField,TextAreaField,SubmitField,IntegerField
from wtforms.validators import DataRequired
#Machine Learning Library
import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
 
import warnings                        # To ignore any warnings warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer


app=Flask(__name__)

app.config['SECRET_KEY']='mykey'

class InfoForm(FlaskForm):
    xName=StringField ('Welcome, What is your name?', validators=[DataRequired()])
    xGender=RadioField ('What Gender are you ?',choices=[('Male','Male'),('Female','Female')])
    xMarried=SelectField(u'Are you married',choices=[('Yes','Yes'),('No','No')])
    xDepend=SelectField(u'How many dependants do yo have?',choices=[('0','1'),('1','1'),('2','2'),('3+','Over 3')])
    xEducation=SelectField(u'Education Level',choices=[('Graduate','Graduate'),('Not Graduate','Not A Graduate')])
   
    xSelf_Employ=RadioField ('Are you Self Employed?',choices=[('Yes','Yes'),('No','No')])
    xApplicantIncome=IntegerField('kindly disclose you Monthly income',validators=[DataRequired()])
    
    xCoapplicantIncome=IntegerField('kindly disclose the coapplicant Monthly income',validators=[DataRequired()])
    
    xLoanAmount=IntegerField('kindly disclose the amount you need',validators=[DataRequired()])

    xLoan_Amount_Term=SelectField(u'How long do you need the loan for',choices=[('12','12'),('36','36'),('60','60'),('84','84'),('120','120'),('180','180'),('240','240'),('300','300'),('360','360'),('480','480')])

    xCredit=SelectField(u'Have you taken Loan before',choices=[('1','Yes'),('0','No')])

    xPropty=SelectField(u'Which of these propery area do you stay?',choices=[('Urban','Urban'),('Semiurban','Semiurban'),('Rural','Rural')])




    submit=SubmitField('Submit')

@app.route('/', methods=['GET','POST'])
def index():
        form= InfoForm()
        if form.validate_on_submit():
            session['xName']=form.xName.data
            session['xGender']=form.xGender.data
            session['xMarried']=form.xMarried.data
            session['xDepend']=form.xDepend.data
            session['xEducation']=form.xEducation.data
            session['xSelf_Employ']=form.xSelf_Employ.data
            session['xApplicantIncome']=form.xApplicantIncome.data
            session['xCoapplicantIncome']=form.xCoapplicantIncome.data
            session['xLoanAmount']=form.xLoanAmount.data
            session['xLoan_Amount_Term']=form.xLoan_Amount_Term.data
            session['xCredit']=form.xCredit.data
            session['xPropty']=form.xPropty.data

            return redirect(url_for('thankyou'))

        return render_template('windex.html', form=form)


@app.route('/thankyou')

def thankyou():
    
    train=pd.read_csv("C:\\Users\\TEMILOLUWA\\Downloads\\train_u6lujuX_CVtuZ9i.csv") 
    test=pd.read_csv("C:\\Users\\TEMILOLUWA\\Downloads\\test_Y3wMUE5_7gLdaTN.csv")
    test.head()
   # train_original=train.copy() 
   # test_original=test.copy()

    train.isnull().sum()
    train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
    train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
    train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
    train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
    train.isnull().sum()

    train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
    train.isnull().sum()
    test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
    test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
    test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
    test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
    test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
    test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
    #test.isnull().sum()
    train=train.drop('Loan_ID',axis=1) 
    test=test.drop('Loan_ID',axis=1)
    X = train.drop('Loan_Status',1) 
    y = train.Loan_Status
    X=pd.get_dummies(X) 
    train=pd.get_dummies(train) 
    test=pd.get_dummies(test)
    from sklearn.model_selection import train_test_split
    x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

    from sklearn.linear_model import LogisticRegression 
    from sklearn.metrics import accuracy_score
    modelz = LogisticRegression() 
    modelz.fit(x_train, y_train)
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,          penalty='l2', random_state=1, solver='liblinear', tol=0.0001,          verbose=0, warm_start=False)
    pred_cv = modelz.predict(x_cv)
    accuracy_score(y_cv,pred_cv)
    from sklearn.externals import joblib
    joblib.dump(modelz, 'model.pkl')
    modelz= joblib.load('model.pkl')

    # Saving the data columns from training
    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    #print("Models columns dumped!")
    cv=CountVectorizer()
    mypredictions=0
    if request.method == 'POST':
            namequery=request.form['form.xName.data','form.xGender.data','form.xMarried.data','form.xDepend.data','form.xEducation.data','form.xSelf_Employ.data','form.xApplicantIncome.data','form.xCoapplicantIncome.data','form.xLoanAmount.data','form.xLoan_Amount_Term.data','form.xCredit.data','form.xPropty.data']
            data=[namequery]

            vect=cv.transform(data).toarray()

            mypredictions= modelz.predict(vect)

            return mypredictions

            


    

    return render_template('thankyou.html',mypredictionsd=mypredictions )


 


if __name__=='__main__':
        app.run(debug=True)
   
    
