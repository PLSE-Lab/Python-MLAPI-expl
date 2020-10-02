#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_test_data = [train_data, test_data]

# MSSubClass: Identifies the type of dwelling involved in the sale.	

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

MSZoning: Identifies the general zoning classification of the sale.
		
       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density
	
LotFrontage: Linear feet of street connected to property

LotArea: Lot size in square feet

Street: Type of road access to property

       Grvl	Gravel	
       Pave	Paved
       	
Alley: Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
		
LotShape: General shape of property

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular
       
LandContour: Flatness of the property

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression
		
Utilities: Type of utilities available
		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	
	
LotConfig: Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property
	
LandSlope: Slope of property
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
	
Neighborhood: Physical locations within Ames city limits

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
			
Condition1: Proximity to various conditions
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
Condition2: Proximity to various conditions (if more than one is present)
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
BldgType: Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
	
HouseStyle: Style of dwelling
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level
	
OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
		
YearBuilt: Original construction date

YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)

RoofStyle: Type of roof

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed
		
RoofMatl: Roof material

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
		
Exterior1st: Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
Exterior2nd: Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
	
MasVnrArea: Masonry veneer area in square feet

ExterQual: Evaluates the quality of the material on the exterior 
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
ExterCond: Evaluates the present condition of the material on the exterior
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
Foundation: Type of foundation
		
       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete	
       Slab	Slab
       Stone	Stone
       Wood	Wood
		
BsmtQual: Evaluates the height of the basement

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
		
BsmtCond: Evaluates the general condition of the basement

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
	
BsmtExposure: Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
	
BsmtFinType1: Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
		
BsmtFinSF1: Type 1 finished square feet

BsmtFinType2: Rating of basement finished area (if multiple types)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinSF2: Type 2 finished square feet

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating
		
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace	
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace
		
HeatingQC: Heating quality and condition

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
CentralAir: Central air conditioning

       N	No
       Y	Yes
		
Electrical: Electrical system

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix	Mixed
		
1stFlrSF: First Floor square feet
 
2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms

BsmtHalfBath: Basement half bathrooms

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

Kitchen: Kitchens above grade

KitchenQual: Kitchen quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       	
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality (Assume typical unless deductions are warranted)

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only
		
Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
		
GarageType: Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
		
GarageYrBlt: Year garage was built
		
GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
		
GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
GarageCond: Garage condition

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
PavedDrive: Paved driveway

       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
		
WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet

EnclosedPorch: Enclosed porch area in square feet

3SsnPorch: Three season porch area in square feet

ScreenPorch: Screen porch area in square feet

PoolArea: Pool area in square feet

PoolQC: Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
		
Fence: Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
	
MiscFeature: Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
		
MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
		
SaleCondition: Condition of sale

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)

# ## Explore Data

# In[ ]:


train_data.info()


# ### there are 1460 rows 
# ### how many are nulls in train data?

# In[ ]:


#finding no of null values and the column name which contains null value
print("no of null values in each column of train set is\n",train_data.isna().sum())
nan_val= []
for i in train_data.columns:
    if(train_data[i].isna().sum() > 1000):
        nan_val.append(i)
print("\nno of columns with null values are\n",len(nan_val))      


# <b>we have <u>19</u> columns with null values and their name can be found in list nan_val,out of which 4 have more than 1000 null values and it is better to drop them  
# 

# ### how many are nulls in train data?

# In[ ]:


print("no of null values in each column of train set is\n",train_data.isna().sum())
nan_val= []
for i in test_data.columns:
    if(test_data[i].isna().sum() > 1000):
        nan_val.append(i)
print("\nno of columns with null values are\n",len(nan_val))   


# <b>we have <u>33</u> columns with null values and their name can be found in list nan_val,out of which 4 have more than 1000 null values and it is better to drop them  
# 

# In[ ]:


nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum() >1000):
        nan_val1.append(i)
#print("\nno of columns with null values are\n",len(nan_val1))    
print(nan_val1)


# In[ ]:


train_data.drop(columns = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'],inplace = True)
test_data.drop(columns = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'],inplace = True)


# In[ ]:


nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<500 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)

BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
# In[ ]:


#for dataset in train_test_data:
#print(test_data['MasVnrType'].mode(),'\n')
#print(train_data.groupby('MasVnrType')['MasVnrArea'].mean())
#print(test_data.groupby('MasVnrType')['MasVnrArea'].mean())
train_data['MasVnrType'].fillna("None",inplace = True)
test_data['MasVnrType'].fillna("None",inplace = True)


# In[ ]:


nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


train_data['MasVnrArea'].fillna(train_data.groupby('MasVnrType')['MasVnrArea'].transform('mean'),inplace = True)
test_data['MasVnrArea'].fillna(train_data.groupby('MasVnrType')['MasVnrArea'].transform('mean'),inplace = True)


# In[ ]:


nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


#quality = train_data[['PavedDrive','SalePrice']]
#quality.groupby('PavedDrive').mean().plot.bar()
#print(quality.groupby('PavedDrive').mean())


# In[ ]:


re = train_data.corr()
re['LotFrontage']


# In[ ]:


A=train_data.LotFrontage
Anan=A[~np.isnan(A)] # Remove the NaNs

sns.distplot(Anan,hist=True,bins = 10)


# In[ ]:


train_data['LotFrontage'].fillna(train_data.LotFrontage.mean(),inplace = True)
test_data['LotFrontage'].fillna(train_data.LotFrontage.mean(),inplace = True)


# In[ ]:


print(train_data['LotFrontage'].isna().sum())
print(test_data['LotFrontage'].isna().sum())


# In[ ]:


nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['BsmtQual'].value_counts())
print(test_data['BsmtQual'].value_counts())
train_data.BsmtQual.mode()[0]


# In[ ]:


train_data['BsmtQual'].fillna('None',inplace = True)
test_data['BsmtQual'].fillna('None',inplace = True)


# In[ ]:


nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['BsmtCond'].value_counts())
print(test_data['BsmtCond'].value_counts())
train_data['BsmtCond'].fillna("None",inplace = True)
test_data['BsmtCond'].fillna("None",inplace = True)
nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['BsmtExposure'].value_counts())
print(test_data['BsmtExposure'].value_counts())
train_data['BsmtExposure'].fillna("None",inplace = True)
test_data['BsmtExposure'].fillna("None",inplace = True)
nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['BsmtFinType2'].value_counts())
print(test_data['BsmtFinType2'].value_counts())
train_data['BsmtFinType2'].fillna("None",inplace = True)
test_data['BsmtFinType2'].fillna("None",inplace = True)
nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['Electrical'].value_counts())
print(test_data['Electrical'].value_counts())
train_data['Electrical'].fillna(train_data['Electrical'].mode()[0],inplace = True)
test_data['Electrical'].fillna(test_data['Electrical'].mode()[0],inplace = True)
nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['GarageType'].value_counts())
print(test_data['GarageType'].value_counts())
train_data['GarageType'].fillna("None",inplace = True)
test_data['GarageType'].fillna("None",inplace = True)
nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['GarageYrBlt'].value_counts())
print(test_data['GarageYrBlt'].value_counts())
train_data['GarageYrBlt'].fillna(0,inplace = True)
test_data['GarageYrBlt'].fillna(0,inplace = True)
nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['GarageFinish'].value_counts())
print(test_data['GarageFinish'].value_counts())
train_data['GarageFinish'].fillna("None",inplace = True)
test_data['GarageFinish'].fillna("None",inplace = True)
nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['GarageQual'].value_counts())
print(test_data['GarageQual'].value_counts())
train_data['GarageQual'].fillna("None",inplace = True)
test_data['GarageQual'].fillna("None",inplace = True)
nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['GarageCond'].value_counts())
print(test_data['GarageCond'].value_counts())
train_data['GarageCond'].fillna("None",inplace = True)
test_data['GarageCond'].fillna("None",inplace = True)
nan_val1= []
for i in test_data.columns:
    if(train_data[i].isna().sum()<300 and train_data[i].isna().sum()!=0 ):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(train_data['BsmtFinType1'].value_counts())
print(test_data['BsmtFinType1'].value_counts())
train_data['BsmtFinType1'].fillna("None",inplace = True)
test_data['BsmtFinType1'].fillna("None",inplace = True)
nan_val1= []


# In[ ]:


for i in test_data.columns:
    if(train_data[i].isna().sum()>300):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


train_data['FireplaceQu'].fillna("None",inplace = True)
test_data['FireplaceQu'].fillna("None",inplace = True)


# In[ ]:


nan_val1 = []
for i in train_data.columns:
    if(train_data[i].isna().sum()>50):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


import seaborn as sns


# In[ ]:


nan_val1 = []
for i in test_data.columns:
    if(test_data[i].isnull().sum()>0):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


re['GarageCars']


# In[ ]:


train_data['GarageArea'].fillna(0,inplace = True)
test_data['GarageArea'].fillna(0,inplace = True)


# In[ ]:


a=(test_data.groupby('GarageArea')['GarageCars'].min())
b=(test_data.groupby('GarageArea')['GarageCars'].max())
c=(test_data.groupby('GarageArea')['GarageCars'].mean())
print(a,b,c)
#sns.distplot(test_data['GarageArea'])


# In[ ]:


test_data['GarageCars'].fillna(test_data.groupby('GarageArea')['GarageCars'].transform('mean'),inplace = True)


# In[ ]:


nan_val1 = []
for i in test_data.columns:
    if(test_data[i].isna().sum()>0):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


test_data.Exterior1st.fillna(test_data.Exterior1st.mode()[0],inplace = True)


# In[ ]:


test_data.Exterior2nd.fillna(test_data.Exterior2nd.mode()[0],inplace = True)
nan_val1 = []
for i in test_data.columns:
    if(test_data[i].isna().sum()>0):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


(test_data.BsmtFinSF1.fillna(0,inplace = True))
(test_data.BsmtFinSF2.fillna(0,inplace = True))


# In[ ]:


nan_val1 = []
for i in test_data.columns:
    if(test_data[i].isna().sum()>0):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


test_data.KitchenQual.fillna(test_data.KitchenQual.mode()[0],inplace = True)
test_data.KitchenQual.value_counts()


# In[ ]:


test_data.MSZoning.fillna(test_data.MSZoning.mode()[0],inplace = True)
test_data.MSZoning.isna().sum()


# In[ ]:


nan_val1 = []
for i in test_data.columns:
    if(test_data[i].isna().sum()>0):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


test_data.Utilities.fillna("AllPub",inplace = True)
nan_val1 = []
for i in test_data.columns:
    if(test_data[i].isna().sum()>0):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


test_data.TotalBsmtSF.fillna(test_data.TotalBsmtSF.mean(),inplace = True)
test_data.Functional.fillna("Typ",inplace = True)
nan_val1 = []
for i in test_data.columns:
    if(test_data[i].isna().sum()>0):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


print(test_data.BsmtUnfSF.mean())
test_data.BsmtUnfSF.fillna(0,inplace= True)
nan_val1 = []
for i in test_data.columns:
    if(test_data[i].isna().sum()>0):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


test_data.BsmtHalfBath.fillna(0.0,inplace = True)
test_data.BsmtFullBath.fillna(0.0,inplace=True)
test_data.SaleType.fillna(test_data.SaleType.mode()[0],inplace=True)


# In[ ]:


nan_val1 = []
for i in train_data.columns:
    if(train_data[i].isna().sum()>0):
        nan_val1.append(i)
print(nan_val1)


# In[ ]:


colsn=[]  
for i in test_data.columns:
    if(test_data[i].dtype == 'object'):
        colsn.append(i)
len(colsn)


# In[ ]:


test_data.shape


# In[ ]:


train_data.shape


# ## Encoding

# In[ ]:


from sklearn import preprocessing


# In[ ]:


train_copy = train_data.copy()
test_copy = test_data.copy()
enc = pd.concat([train_copy,test_copy],sort = False)
enc.shape


# In[ ]:


"""l = ['a','b','s','a']
le = preprocessing.LabelEncoder()
le.fit(l)
list(le.transform(l))
l = []
for i in test_data['SaleType']:
    l.append(i)
y = list(le.fit_transform(l))"""


# In[ ]:


le = preprocessing.LabelEncoder()
for i in colsn:
    v =[]
    for j in enc[i]:
        v.append(j)
    
    enc[i].replace(v,list(le.fit_transform(v)),inplace = True)


# In[ ]:


enc.dtypes


# In[ ]:


train_df = enc.iloc[:1460, :]
test_df = enc.iloc[1460:,:]


# In[ ]:


train_df.shape


# In[ ]:


test_df.drop('SalePrice',axis = 1,inplace = True)


# In[ ]:


test_df.isna().sum()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn import linear_model
xtrain = train_df.drop(['SalePrice'],axis = 1)
ytrain = train_df['SalePrice']


# In[ ]:


rfc = RandomForestRegressor(n_estimators=900)
rfc.fit(xtrain,ytrain)
y_pred = rfc.predict(test_df)
y_pred


# In[ ]:


import xgboost
regressor=xgboost.XGBRegressor()


from sklearn.model_selection import RandomizedSearchCV


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv.fit(xtrain,ytrain)


# In[ ]:


#from sklearn.model_selection import GridSearchCV
#grid_cv = GridSearchCV(estimator=regressor,
#            param_grid=hyperparameter_grid,
#            cv=2, 
#            scoring = 'neg_mean_absolute_error',n_jobs = 100,
#            verbose = 5, 
#            return_train_score = True,
#            )
#grid_cv.fit(xtrain,ytrain)


# In[ ]:


random_cv.best_estimator_


# In[ ]:


#cl = xgboost.XGBRegressor(base_score=1, booster='gbtree', colsample_bylevel=1,
#             colsample_bynode=1, colsample_bytree=1, gamma=0,
#             importance_type='gain', learning_rate=0.1, max_delta_step=0,
#             max_depth=3, min_child_weight=4, missing=None, n_estimators=1100,
#             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#             silent=None, subsample=1, verbosity=1)
#cl.fit(xtrain,ytrain)
#y_pred2 = cl.predict(test_df)
#y_pred2


# In[ ]:


#Linear Regression
lm = linear_model.LinearRegression()
model = lm.fit(xtrain,ytrain)
y_pred3 = lm.predict(test_df)
y_pred3+20000


# In[ ]:


#for i in colsn:
 #   print(train_df[i].value_counts())


# In[ ]:


#subb = pd.DataFrame({
#       "Id": test_df["Id"],
#       "SalePrice": y_pred2
#   })
#subb.to_csv('house_pricev1.csv', index=False)


# In[ ]:


#remove utilities 


# In[ ]:





# In[ ]:



#sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

