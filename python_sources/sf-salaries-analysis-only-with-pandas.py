import pandas as pd
import numpy as np
df = pd.read_csv("../input/sf-salaries/Salaries.csv")

#information about column and data types
print(df.info())
#displays the top 5 rows of the dataset
print(df.head())

#change case of all job titles in order to bring about uniformity
df["JobTitle"]=df["JobTitle"].str.upper()

#these two columns do not have any values in them, hence we drop them"
df.drop(columns=["Notes","Status"],inplace=True)
print(df.info())

print("\ndetails about the person with the highest pay+benefits")
s=df["TotalPayBenefits"].idxmax()
print(df.loc[s])

print("\ndescription about the BasePay,OvertimePay,Benefits and TotalPayBenefits")
print(round(df['BasePay'].describe()))
print(round(df['OvertimePay'].describe()))
print(round(df["Benefits"].describe()))
print(round(df["TotalPayBenefits"].describe()))

print("\ndisplays mean benefits ,overtimepay, otherpay and totalpay+benefits year-wise")
a=round((df.groupby("Year"))['Benefits','OvertimePay','OtherPay',"TotalPayBenefits"].aggregate('mean'))
print(a)

print("display the number of unique job titles")
print(df['JobTitle'].nunique())
print("display the top 15 most common jobs")
print(df['JobTitle'].value_counts().head(15))
print("\ndisplays top 5 most common jobs year-wise")
a=df["Year"].unique()
for i in a:
    print(i)
    print(df[df["Year"]==i]["JobTitle"].value_counts().head(5))

#print("\nextract names of top ten jobs")
top_ten_occupations = df["JobTitle"].value_counts().head(10).index
#print(top_ten_occupations)
print("displays mean benefits ,overtimepay, otherpay and totalpay+benefits job-wise")
salaries_averages_by_occupation = (df[df.JobTitle.isin(top_ten_occupations)]
                                   .groupby('JobTitle')[['Benefits', 'OvertimePay','TotalPayBenefits']].aggregate('mean'))
print(salaries_averages_by_occupation)

print("\ndisplay the top 10 job titles with with highest mean TotalPayBenefits")
print(round((df.groupby("JobTitle"))['TotalPayBenefits'].mean().nlargest(10)))
print("\ndisplay the top 10 job titles with with lowest mean TotalPayBenefits")
print(round((df.groupby("JobTitle"))['TotalPayBenefits'].mean().nsmallest(10)))

def chief_string(title):
    if 'CHIEF' in title:
        return True
    else:
        return False
def public_string(title):
    if 'PUBLIC' in title:
        return True
    else:
        return False
def police_string(title):
    if 'POLICE' in title:
        return True
    else:
        return False
def fire_string(title):
    if 'FIRE' in title:
        return True
    else:
        return False

print("\nPeople with chief in jobtitle ",df[df['JobTitle'].apply(lambda x: chief_string(x))]["JobTitle"].count())
print("People working in public related sectors ",df[df['JobTitle'].apply(lambda x: public_string(x))]["JobTitle"].count())
print("People working in Police department ",df[df['JobTitle'].apply(lambda x: police_string(x))]["JobTitle"].count())
print("People working in Fire department ",df[df['JobTitle'].apply(lambda x: fire_string(x))]["JobTitle"].count())

print("\nmean Total pay + benefits : ")
print("Chief ",round(df[df['JobTitle'].apply(lambda x: chief_string(x))]["TotalPayBenefits"].mean()))
print("Public sector ",round(df[df['JobTitle'].apply(lambda x: public_string(x))]["TotalPayBenefits"].mean()))
print("Police department ",round(df[df['JobTitle'].apply(lambda x: police_string(x))]["TotalPayBenefits"].mean()))
print("Fire department ",round(df[df['JobTitle'].apply(lambda x: fire_string(x))]["TotalPayBenefits"].mean()))

print("\nHere you can see the difference in mean pay scale for public sector jobs vs all jobs")
print("Public Sector")
print(round(df[df['JobTitle'].apply(lambda x: public_string(x))].groupby("Year")
            ['Benefits','OvertimePay','OtherPay',"TotalPayBenefits"].mean()))
print("All Jobs")
print(round((df.groupby("Year"))['Benefits','OvertimePay','OtherPay',"TotalPayBenefits"].aggregate('mean')))

print("\nHere you can see that the difference in mean overtimepay received by the police department and fire department")
print("Police department ",round(df[df['JobTitle'].apply(lambda x: police_string(x))]["OvertimePay"].mean()))
print("Fire department ",round(df[df['JobTitle'].apply(lambda x: fire_string(x))]["OvertimePay"].mean()))
