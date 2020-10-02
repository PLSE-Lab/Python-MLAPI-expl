print("This file explains the data-generation process from the original dataset (http://www.nature.com/articles/sdata201555#data-records) to the Kaggle's one")
print("")
print("This source file is commented because it is not made to be runned in Kaggle.com")

'''
import pandas as pd
import datetime
parse = lambda x: datetime.datetime.fromtimestamp(float(int(x)/1000.))


## From the original, Open sourced CDRs to the Kaggle's ones
for i in range(1,8):
    df = pd.read_csv('data/original_CDRs/sms-call-internet-mi-2013-11-0{}.txt'.format(i), sep='\t', encoding="utf-8-sig", names=['CellID', 'datetime', 'cell2Province', 'Province2cell'], parse_dates=['datetime'], date_parser=parse)
    df = df.set_index('datetime')
    df = df.groupby([pd.TimeGrouper(freq='60Min'), 'CellID', 'countrycode']).sum()
    df = df.round(4)
    df.to_csv('data/generated_files/sms-call-internet-mi-2013-11-0{}.csv'.format(i))


## From the original, Open sourced CDRs to the Kaggle's ones
for i in range(1,8):
    df = pd.read_csv('data/original_CDRs/mi-to-provinces-2013-11-0{}.txt'.format(i), sep='\t', encoding="utf-8-sig", names=['CellID', 'provinceName', 'datetime', 'cell2Province', 'Province2cell'], parse_dates=['datetime'], date_parser=parse)
    df = df.set_index('datetime')
    df = df.groupby([pd.TimeGrouper(freq='60Min'), 'CellID', 'provinceName']).sum()
    df = df.round(4)
    df.to_csv('data/generated_files/mi-to-provinces-2013-11-0{}.csv'.format(i))


## From the original, Open sourced ISTAT census files to the Kaggle's ones
df_prov = pd.DataFrame({})
for i in range(1,21):
    df = pd.read_csv('data/census/R{}_indicatori_2011_localita.csv'.format(str.zfill(str(i),2)), sep=';', encoding = "ISO-8859-1")
    df = df.groupby(['PROVINCIA']).sum()
    df = df.drop(['CODREG', 'CODPRO', 'CODCOM', 'PROCOM', 'LOC2011', 'CODLOC', 'TIPOLOC', 'AMPLOC', 'CAPOLUOGO'], 1)
    df_prov = df_prov.append(df)
df_prov.to_csv('data/generated_files/ISTAT_census_variables_2011.csv')
'''