'''
Dated: Nov03-2018
Auhor: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for data preprocessing row-wise time series data into columnar time series data using sql-style pandas statements
Results:
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input")) 

def data_preprocessing_sql(row_wise_table):
    #let's do some sql with pandas 
    #select all values of 'Shipment Units' where product is '2 Door Bottom Mount'
    #select id from airports where ident = 'KLAX'	airports[airports.ident == 'KLAX'].id
    print("Checking for nulls:\n", row_wise_table.isnull().sum())
    print('\nStandardizing column names')
    row_wise_table.columns = [c.lower().replace(' ', '_') for c in row_wise_table.columns]
    print(row_wise_table.columns)
    print('Done')
    print('Begin SQLing with pandas')
    print("SELECT shipment_units FROM row_wise_table WHERE product == '2 Door Bottom Mount'")

    '''df.loc[df['column_name'] == some_value]'''
    #print(row_wise_table.loc[row_wise_table['product']=='2 Door Bottom Mount'].shipment_units)
    print("Extracting monthly aggregates for 2_Door_Bottom_Mount")
    col_2_Door_Bottom_Mount_weekly = pd.Series()
    col_2_Door_Bottom_Mount_monthly = pd.Series()
    col_2_Door_Bottom_Mount_weekly = row_wise_table.loc[row_wise_table['product']=='2 Door Bottom Mount'].shipment_units
    col_2_Door_Bottom_Mount_weekly.reset_index(inplace=True, drop=True)
    col_2_Door_Bottom_Mount_monthly = col_2_Door_Bottom_Mount_weekly.groupby(col_2_Door_Bottom_Mount_weekly.index // 4 * 4).sum()
    col_2_Door_Bottom_Mount_monthly.reset_index(inplace=True, drop=True)
    #print('col_2_Door_Bottom_Mount_weekly[:30]\n', col_2_Door_Bottom_Mount_weekly[:30])
    #print('col_2_Door_Bottom_Mount_monthly[:8]\n', col_2_Door_Bottom_Mount_monthly[:8])
    print(len(col_2_Door_Bottom_Mount_weekly))
    print(len(col_2_Door_Bottom_Mount_monthly))
    
    print("Extracting monthly aggregates for Built_in_Ovens")
    col_Built_in_Ovens_weekly = pd.Series()
    col_Built_in_Ovens_monthly = pd.Series()
    col_Built_in_Ovens_weekly = row_wise_table.loc[row_wise_table['product']=='Built-in Ovens'].shipment_units
    col_Built_in_Ovens_weekly.reset_index(inplace=True, drop=True)
    col_Built_in_Ovens_monthly = col_Built_in_Ovens_weekly.groupby(col_Built_in_Ovens_weekly.index // 4 * 4).sum()
    col_Built_in_Ovens_monthly.reset_index(inplace=True, drop=True)
    #print('col_Built_in_Ovens_weekly[:30]\n', col_Built_in_Ovens_weekly[:30])
    #print('col_Built_in_Ovens_monthly[:8]\n', col_Built_in_Ovens_monthly[:8])
    print(len(col_Built_in_Ovens_weekly))
    print(len(col_Built_in_Ovens_monthly))    
    
    print("Extracting monthly aggregates for Cooktops")
    col_Cooktops_weekly = pd.Series()
    col_Cooktops_monthly = pd.Series()
    col_Cooktops_weekly = row_wise_table.loc[row_wise_table['product']=='Cooktops'].shipment_units
    col_Cooktops_weekly.reset_index(inplace=True, drop=True)
    col_Cooktops_monthly = col_Cooktops_weekly.groupby(col_Cooktops_weekly.index // 4 * 4).sum()
    col_Cooktops_monthly.reset_index(inplace=True, drop=True)
    #print('col_Cooktops_weekly[:30]\n', col_Cooktops_weekly[:30])
    #print('col_Cooktops_monthly[:8]\n', col_Cooktops_monthly[:8])
    print(len(col_Cooktops_weekly))
    print(len(col_Cooktops_monthly))
    
    print("Extracting monthly aggregates for Dishwasher")
    col_Dishwasher_weekly = pd.Series()
    col_Dishwasher_monthly = pd.Series()
    col_Dishwasher_weekly = row_wise_table.loc[row_wise_table['product']=='Dishwasher'].shipment_units
    col_Dishwasher_weekly.reset_index(inplace=True, drop=True)
    col_Dishwasher_monthly = col_Dishwasher_weekly.groupby(col_Dishwasher_weekly.index // 4 * 4).sum()
    col_Dishwasher_monthly.reset_index(inplace=True, drop=True)
    #print('col_Dishwasher_weekly[:30]\n', col_Dishwasher_weekly[:30])
    #print('col_Dishwasher_monthly[:8]\n', col_Dishwasher_monthly[:8])
    print(len(col_Dishwasher_weekly))
    print(len(col_Dishwasher_monthly))
    
    print("Extracting monthly aggregates for Dryer")
    col_Dryer_weekly = pd.Series()
    col_Dryer_monthly = pd.Series()
    col_Dryer_weekly = row_wise_table.loc[row_wise_table['product']=='Dryer'].shipment_units
    col_Dryer_weekly.reset_index(inplace=True, drop=True)
    col_Dryer_monthly = col_Dryer_weekly.groupby(col_Dryer_weekly.index // 4 * 4).sum()
    col_Dryer_monthly.reset_index(inplace=True, drop=True)
    #print('col_Dryer_weekly[:30]\n', col_Dryer_weekly[:30])
    #print('col_Dryer_monthly[:8]\n', col_Dryer_monthly[:8])
    print(len(col_Dryer_weekly))
    print(len(col_Dryer_monthly))
    
    print("Extracting monthly aggregates for Free_Standing_Ranges")
    col_Free_Standing_Ranges_weekly = pd.Series()
    col_Free_Standing_Ranges_monthly = pd.Series()
    col_Free_Standing_Ranges_weekly = row_wise_table.loc[row_wise_table['product']=='Free Standing Ranges'].shipment_units
    col_Free_Standing_Ranges_weekly.reset_index(inplace=True, drop=True)
    col_Free_Standing_Ranges_monthly = col_Free_Standing_Ranges_weekly.groupby(col_Free_Standing_Ranges_weekly.index // 4 * 4).sum()
    col_Free_Standing_Ranges_monthly.reset_index(inplace=True, drop=True)
    #print('col_Free_Standing_Ranges_weekly[:30]\n', col_Free_Standing_Ranges_weekly[:30])
    #print('col_Free_Standing_Ranges_monthly[:8]\n', col_Free_Standing_Ranges_monthly[:8])
    print(len(col_Free_Standing_Ranges_weekly))
    print(len(col_Free_Standing_Ranges_monthly))
    
    print("Extracting monthly aggregates for Freezer")
    col_Freezer_weekly = pd.Series()
    col_Freezer_monthly = pd.Series()
    col_Freezer_weekly = row_wise_table.loc[row_wise_table['product']=='Freezer'].shipment_units
    col_Freezer_weekly.reset_index(inplace=True, drop=True)
    col_Freezer_monthly = col_Freezer_weekly.groupby(col_Freezer_weekly.index // 4 * 4).sum()
    col_Freezer_monthly.reset_index(inplace=True, drop=True)
    #print('col_Freezer_weekly[:30]\n', col_Freezer_weekly[:30])
    #print('col_Freezer_monthly[:8]\n', col_Freezer_monthly[:8])
    print(len(col_Freezer_weekly))
    print(len(col_Freezer_monthly))
    
    print("Extracting monthly aggregates for French_Door")
    col_French_Door_weekly = pd.Series()
    col_French_Door_monthly = pd.Series()
    col_French_Door_weekly = row_wise_table.loc[row_wise_table['product']=='French Door'].shipment_units
    col_French_Door_weekly.reset_index(inplace=True, drop=True)
    col_French_Door_monthly = col_French_Door_weekly.groupby(col_French_Door_weekly.index // 4 * 4).sum()
    col_French_Door_monthly.reset_index(inplace=True, drop=True)
    #print('col_French_Door_weekly[:30]\n', col_French_Door_weekly[:30])
    #print('col_French_Door_monthly[:8]\n', col_French_Door_monthly[:8])
    print(len(col_French_Door_weekly))
    print(len(col_French_Door_monthly))
    
    print("Extracting monthly aggregates for Front_Load")
    col_Front_Load_weekly = pd.Series()
    col_Front_Load_monthly = pd.Series()
    col_Front_Load_weekly = row_wise_table.loc[row_wise_table['product']=='Front Load'].shipment_units
    col_Front_Load_weekly.reset_index(inplace=True, drop=True)
    col_Front_Load_monthly = col_Front_Load_weekly.groupby(col_Front_Load_weekly.index // 4 * 4).sum()
    col_Front_Load_monthly.reset_index(inplace=True, drop=True)
    #print('col_Front_Load_weekly[:30]\n', col_Front_Load_weekly[:30])
    #print('col_Front_Load_monthly[:8]\n', col_Front_Load_monthly[:8])
    print(len(col_Front_Load_weekly))
    print(len(col_Front_Load_monthly))
    
    print("Extracting monthly aggregates for MHC")
    col_MHC_weekly = pd.Series()
    col_MHC_monthly = pd.Series()
    col_MHC_weekly = row_wise_table.loc[row_wise_table['product']=='MHC'].shipment_units
    col_MHC_weekly.reset_index(inplace=True, drop=True)
    col_MHC_monthly = col_MHC_weekly.groupby(col_MHC_weekly.index // 4 * 4).sum()
    col_MHC_monthly.reset_index(inplace=True, drop=True)
    #print('col_MHC_weekly[:30]\n', col_MHC_weekly[:30])
    #print('col_MHC_monthly[:8]\n', col_MHC_monthly[:8])
    print(len(col_MHC_weekly))
    print(len(col_MHC_monthly))
    
    print("Extracting monthly aggregates for Side_by_Side")
    col_Side_by_Side_weekly = pd.Series()
    col_Side_by_Side_monthly = pd.Series()
    col_Side_by_Side_weekly = row_wise_table.loc[row_wise_table['product']=='Side by Side'].shipment_units
    col_Side_by_Side_weekly.reset_index(inplace=True, drop=True)
    col_Side_by_Side_monthly = col_Side_by_Side_weekly.groupby(col_Side_by_Side_weekly.index // 4 * 4).sum()
    col_Side_by_Side_monthly.reset_index(inplace=True, drop=True)
    #print('col_Side_by_Side_weekly[:30]\n', col_Side_by_Side_weekly[:30])
    #print('col_Side_by_Side_monthly[:8]\n', col_Side_by_Side_monthly[:8])
    print(len(col_Side_by_Side_weekly))
    print(len(col_Side_by_Side_monthly))
    
    print("Extracting monthly aggregates for Top_Load")
    col_Top_Load_weekly = pd.Series()
    col_Top_Load_monthly = pd.Series()
    col_Top_Load_weekly = row_wise_table.loc[row_wise_table['product']=='Top Load'].shipment_units
    col_Top_Load_weekly.reset_index(inplace=True, drop=True)
    col_Top_Load_monthly = col_Top_Load_weekly.groupby(col_Top_Load_weekly.index // 4 * 4).sum()
    col_Top_Load_monthly.reset_index(inplace=True, drop=True)
    #print('col_Top_Load_weekly[:30]\n', col_Top_Load_weekly[:30])
    #print('col_Top_Load_monthly[:8]\n', col_Top_Load_monthly[:8])
    print(len(col_Top_Load_weekly))
    print(len(col_Top_Load_monthly))
    
    print("Extracting monthly aggregates for Top_Mount")
    col_Top_Mount_weekly = pd.Series()
    col_Top_Mount_monthly = pd.Series()
    col_Top_Mount_weekly = row_wise_table.loc[row_wise_table['product']=='Top Mount'].shipment_units
    col_Top_Mount_weekly.reset_index(inplace=True, drop=True)
    col_Top_Mount_monthly = col_Top_Mount_weekly.groupby(col_Top_Mount_weekly.index // 4 * 4).sum()
    col_Top_Mount_monthly.reset_index(inplace=True, drop=True)
    #print('col_Top_Mount_weekly[:30]\n', col_Top_Mount_weekly[:30])
    #print('col_Top_Mount_monthly[:8]\n', col_Top_Mount_monthly[:8])
    print(len(col_Top_Mount_weekly))
    print(len(col_Top_Mount_monthly))
    
    '''Add all these 13 series as 13 columns to a new dataframe'''
    col_wise_table = pd.DataFrame()
    col_wise_table['2 Door Bottom Mount'] = col_2_Door_Bottom_Mount_monthly
    col_wise_table['Built-in Ovens'] = col_Built_in_Ovens_monthly
    col_wise_table['Cooktops'] = col_Cooktops_monthly
    col_wise_table['Dishwasher'] = col_Dishwasher_monthly
    col_wise_table['Dryer'] = col_Dryer_monthly
    col_wise_table['Free Standing Ranges'] = col_Free_Standing_Ranges_monthly
    col_wise_table['Freezer'] = col_Freezer_monthly
    col_wise_table['French Door'] = col_French_Door_monthly
    col_wise_table['Front Load'] = col_Front_Load_monthly
    col_wise_table['MHC'] = col_MHC_monthly
    col_wise_table['Side by Side'] = col_Side_by_Side_monthly
    col_wise_table['Top Load'] = col_Top_Load_monthly
    col_wise_table['Top Mount'] = col_Top_Mount_monthly
    
    print(col_wise_table.info())
    print(col_wise_table.describe())
    print(col_wise_table.head())
    print(col_wise_table.tail())
    print(col_wise_table.sample(5))
    return col_wise_table

if __name__ == '__main__':
    '''Source code for data preprocessing row-wise time series data into columnar time series data 
    using sql-style pandas statements'''
    #read in the data from csv file
    row_wise_table = pd.read_csv("../input/Sales-Shipment-Retail-Weekly-data-v2.csv", sep=',')
    #pass dataframe to data_preprocessing_sql(df)
    col_wise_table = data_preprocessing_sql(row_wise_table)
    col_wise_table.to_csv('Sales-Shipment-Retail-Monthly-Column-wise-Time-Series.csv')
    

    
