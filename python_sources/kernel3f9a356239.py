#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[ ]:


os.chdir('../input/')

## With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.## DESCRIPTION OF VARIOUS ATTRIBUTES

MSSubClass: Identifies the type of dwelling involved in the sale.	

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

# In[ ]:


house=pd.read_csv('house.csv')


# In[ ]:


house.head()


# In[ ]:


house.dtypes


# In[ ]:


house.columns


# In[ ]:


house.shape


# In[ ]:


house.describe(include="all")


# In[ ]:


house.isnull().sum()


# In[ ]:


num_col=house.select_dtypes(exclude='object')
cat_col=house.select_dtypes(exclude=['int64','float64'])


# In[ ]:


num_col.isnull().sum()


# In[ ]:


num_col.dtypes


# In[ ]:


cat_col.dtypes


# In[ ]:


cat_col.isnull().sum()


# In[ ]:


num_col.describe(include="all")


# In[ ]:


# HEATMAP TO SEE MISSING VALUES IN NUMERICAL DATA
plt.figure(figsize=(15,5))
sns.heatmap(num_col.isnull(),yticklabels=0,cbar=True,cmap='viridis')

## The above heat mapshows that Lotfrontage,MasVnrArea and GarageYrBlt has NaN values.
# so lets analyse further to see whether we can drop these three features from our analysis.
# # Handling Missing Data  

# In[ ]:



data = house[['MasVnrArea','LotFrontage','GarageYrBlt','SalePrice']]
#correlation = data.corr(method='pearson')
plt.figure(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,cmap='ocean')

## since there is significant correlation between the these three variables n our variable of interest sales price we cannot remove them from our analysis cozof NaN values, so lets think of replacing the missing values.
##### KDE Plot described as Kernel Density Estimate is used for visualizing the Probability Density of a continuous variable.
# lets check the kde plots of Lotfrontage,MasVnrArea and GarageYrBlt
# In[ ]:


sns.kdeplot(house.MasVnrArea,Label='MasVnrArea',color='b')


# In[ ]:


house['MasVnrArea'].replace({np.nan:house.MasVnrArea.mean()},inplace=True)


# In[ ]:


sns.kdeplot(house. GarageYrBlt,Label=' GarageYrBlt',color='g')

# Looking at the Kdeplot for GarageYrBlt , we find that data in this column is not spread enough so we can use mean  or median of this column to fill its Missing Values. median would be more appropriate owing to the shape of the distribution.
# In[ ]:


house['GarageYrBlt'].replace({np.nan:house.GarageYrBlt.median()},inplace=True)


# In[ ]:


sns.kdeplot(house. LotFrontage
,Label='LotFrontage',color='r')

# the above plot seems to be mostly normal so we can replace the missing value with mean.
# In[ ]:


house['LotFrontage'].replace({np.nan:house.LotFrontage.mean()},inplace=True)


# In[ ]:


## HEATMAP SHOWING THE MISSING VALUES IN CATEGORICAL DATA


# In[ ]:


plt.figure(figsize=(15,5))
sns.heatmap(cat_col.isnull(),yticklabels=0,cbar=True,cmap='viridis')


# # Analysis of Numerical Data
 # DISTRIBUTION OF NUMERICAL VARIABLES  
# In[ ]:


#histogram
sns.distplot(house['SalePrice']);

## distribution of Saleprice is almost normal as per the above plot.
# In[ ]:


num_attributes=num_col.drop('SalePrice',axis=1)
fig = plt.figure(figsize=(12,18))
for i in range(len(num_attributes.columns)-1):
    fig.add_subplot(9,4,i+1)
    sns.distplot(num_attributes.iloc[:,i].dropna())
    plt.xlabel(num_attributes.columns[i])

plt.tight_layout()
plt.show()


# In[ ]:


num_col.skew() # the skewness of the plots of numerical attributes , which are observed above.


# #  Heatmap showing the correlation between the features.

# In[ ]:


plt.figure(figsize=(20,8))
sns.heatmap(num_col.corr(),annot=False,cmap='spring',square=True);


# In[ ]:


num_col.corr()['SalePrice'].sort_values(ascending=False)


## Here rather than focussing on the correlation between different features lets have a close look oon the correlation of different features with our feature of interest 'SalePrice'.

#its evident that the top correlated attributes with respect to our target are:
OverallQual  and GrLivArea  , closely followed by GarageCars and GarageArea.

## and the least correlated are    OverallCond,MSSubClass,EnclosedPorch,KitchenAbvGr  


# In[ ]:


fig, axs = plt.subplots(ncols=3,figsize=(15,4))
fig.suptitle("Negative Correlations",fontsize='20')

sns.lineplot(x="KitchenAbvGr",y='SalePrice',data=num_col,color='green',ax=axs[0])
sns.lineplot(x="MSSubClass",y="SalePrice",data=num_col,color='green',ax=axs[1])
sns.lineplot(x="OverallCond",y="SalePrice",color="green",data=num_col,ax=axs[2])


# In[ ]:


fig, axs = plt.subplots(ncols=3,figsize=(15,4))
fig.suptitle("Positive Correlations",fontsize='20')

sns.lineplot(x="OverallQual",y='SalePrice',data=num_col,color='red',ax=axs[0])
sns.lineplot(x="GarageArea",y="SalePrice",data=num_col,color='red',ax=axs[1])
sns.lineplot(x="GrLivArea",y="SalePrice",color="red",data=num_col,ax=axs[2])


# In[ ]:


# now lets look into the closely correlated variables in detail.
plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)


plt.subplot(4,2,1)
sns.swarmplot(y=num_col['SalePrice'],x=num_col['OverallQual'])

plt.subplot(4,2,2)
sns.swarmplot(x=num_col['GrLivArea'],y=num_col['SalePrice'])

plt.subplot(4,2,3)
sns.swarmplot(x=num_col['GarageCars'],y=num_col['SalePrice'])


plt.subplot(4,2,4)
sns.swarmplot(x=num_col['GarageArea'],y=num_col['SalePrice'])

## OverallQual
most people have shown a preference for a medium quality house in the quality range of 5-8 may be because their price range is also in  affordable range for most people.

##Garage Cars
Again most people prefer a medium range of garage in car capacity (2-3) at affordable price.

##Garage Area and GrLivArea is almost equally scattered over all ranges indicating that people are not quite worried about these variables, though there is a kinda linear rise in the  sale price among the different types.



# # Analysis Of Categorical Data

# In[ ]:


sns.swarmplot(y=num_col['SalePrice'],x=cat_col['Neighborhood'])
plt.xticks(rotation=70)


# In[ ]:


fig = plt.figure(figsize=(12.5,4))
sns.countplot(x='Neighborhood', data=cat_col,palette='spring')
plt.xticks(rotation=90)
plt.ylabel('Frequency')
plt.show()

## the plot shows that there is no much difference in preference of types of neighborhood.
##May be we can say compritively northridge heights and college creek are preferred n least preference to northparkvilla.
# In[ ]:


sns.swarmplot(y=num_col['SalePrice'],x=cat_col['Condition1'],palette='spring')

# The plot clearly shows that Normal is preferred over all other conditional proximities and the saleprice is also shows a wide range for these conditions ranging from less than 100000 to more than 700000. 
## People hardly prefer areas adjacent to North South Rail road and within 200' of East-West Rail road with  a low saleprice of between 100000 and 200000.
# In[ ]:


plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)
plt.xticks(rotation=70)

plt.subplot(4,2,1)
sns.swarmplot(y=num_col['SalePrice'],x=cat_col['RoofStyle'])
plt.xticks(rotation=70)

plt.subplot(4,2,2)
sns.swarmplot(x=cat_col['RoofMatl'],y=num_col['SalePrice'])
plt.xticks(rotation=70)

plt.subplot(4,2,3)
sns.swarmplot(x=cat_col['Exterior1st'],y=num_col['SalePrice'])
plt.xticks(rotation=70)

plt.subplot(4,2,4)
sns.swarmplot(x=cat_col['Exterior2nd'],y=num_col['SalePrice'])
plt.xticks(rotation=70)


## ROOFSTYLE
Gabble and Hip roofstyles are prefered over all other styles and their sales price mostly range between less than 200000 and 400000.

##ROOF MATERIAL
standard composite shingle roof material is prefered over all others(almost exclusively) and their price range between 100000 and 400000.

##EXTERIOR1st
vinyl siding is the most preffered exterior covering on the house followed by metal siding and wood siding.
salePrice is slightly higher for Vinyl siding when compared to metal n wood siding.
least preferred exterior covering are asphalt shingles,cinder block and imitation stucco whose saleprice range within 200000.

##EXTERIOR2nd
preferences of people for their  outer exterior covering on house  is almost same as that of the first exterior covering.

# In[ ]:


plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)

plt.subplot(5,2,1)
a1=sns.boxplot(x=cat_col['MasVnrType'],y=num_col['SalePrice']);

plt.subplot(5,2,2)
sns.boxplot(x=cat_col['MSZoning'],y=num_col['SalePrice']);

plt.subplot(5,2,3)

sns.boxplot(x=cat_col['Foundation'],y=num_col['SalePrice'],palette='rainbow');


plt.subplot(5,2,4)
sns.boxplot(x=cat_col['Street'],y=num_col['SalePrice'],palette='rainbow');

plt.subplot(5,2,5)
sns.boxplot(x=cat_col['LotShape'],y=num_col['SalePrice'],palette='rainbow');


plt.subplot(5,2,6)
sns.boxplot(x=cat_col['Utilities'],y=num_col['SalePrice'],palette='rainbow');

plt.subplot(5,2,7)
sns.boxplot(x=cat_col['Heating'],y=num_col['SalePrice'],palette='rainbow');

plt.subplot(5,2,8)
sns.boxplot(x=cat_col['BldgType'],y=num_col['SalePrice'],palette='rainbow');

plt.subplot(5,2,9)
sns.boxplot(x=cat_col['HouseStyle'],y=num_col['SalePrice'],palette='rainbow');

plt.subplot(5,2,10)
sns.boxplot(x=cat_col['KitchenQual'],y=num_col['SalePrice'],palette='rainbow');

##MasVnrType
As per the boxplot none type masonry veneer is the most preferred one followed by Brick face and the least preferred is Brick common type.
whereas price is higher for stone type may be because of which it is not preffered much.

##MSZoning
RL is preffered over all other zonal classification and it has affordable range between 100000 and 200000.
least prefferred is commercial zone inspite of having a lower saleprice .

##Foundation:
Most residents almost exclusively prefer poured concrete closely followed by cinder block above all other foundations inspite of a higher price for poured concrete. probably people want to be quite adamant about the strong foundation of the house upon which they build up their  dreams.

##Street
This clearly shows that Pave street has more Saleprices as compared to Grvl and very Interesting thing Most of the people(99.59%) prefer Pave Street as compared to Grvl and in Pave section also most of the Houses cost under 400,000. 
## Lotshape
almost all lotshapes are equally preffered and their saleprices are almost the same which makes this attribute a low preferred attribute for our analysis.
##Utilities
Houses with all public utilities are unanimously favoured by people and i should say that thee is no much difference in the price as well.

##Heating
Gas A and Gas W are preferred over other types of heating and they fall within a medium price range of 100000 to 200000.
##Dwelling type
most preferred dwelling type is single family dettached and there is no much variation in price among the different types.

##House style
Most preffered style is 2 Story followed by 1 story and 2.5story -second lvel finished.there is no much variation among the price so house style is not an attribute required for this analysis. 
##Kitcchen Qual
Seems like people are not ready to compromise in the quality of kitchen since excellent kitchen quality is preferred over all others inspite of the higherprice.









# In[ ]:


plt.figure(figsize=(13,15))
plt.subplots_adjust(hspace=0.5)
plt.xticks(rotation=70)

plt.subplot(4,2,1)
sns.swarmplot(y=num_col['SalePrice'],x=cat_col['MiscFeature'])
plt.xticks(rotation=70)

plt.subplot(4,2,2)
sns.swarmplot(x=cat_col['SaleType'],y=num_col['SalePrice'])
plt.xticks(rotation=70)

plt.subplot(4,2,3)
sns.swarmplot(x=cat_col['SaleCondition'],y=num_col['SalePrice'])
plt.xticks(rotation=70)

plt.subplot(4,2,4)
sns.swarmplot(x=cat_col['Fence'],y=num_col['SalePrice'])
plt.xticks(rotation=70)

Conclusion

## A medium range of overall quality is preferred which provides an affordable saleprice.
##From sale type , if the house is just constructed or even if conventional warranty deed , sale price will be high.
##from Kitchen Quality, a high quality kitchen is preferred which leads to a higher house price.
## paved street is preferred by most people which considerably increases our target saleprice.
##poured concrete foundation is preferred by people that again increases the price.
##Vinyl siding is preferred for exteriors which is quite costly thereby hiking the house price.
## House with all utilities are preferred though it dosent make much difference in the price range.
##Lotshape is something thats not specifically considered by people and it d not show much variation in price as well.
## there is a hike in price if sale condition is normal or partial.

# In[ ]:




