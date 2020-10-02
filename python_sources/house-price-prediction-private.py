inputFile = "../input/"

import pandas as pd

trainCSV = pd.read_csv(inputFile + "train.csv")
trainCSV = trainCSV.fillna(0)
# print(trainCSV)


trainCSV = trainCSV.drop(['Id', 'Street', 'Utilities', 'Condition2'], axis=1)

# --------------------------------------------MSZoning
colToBeReplaced = trainCSV['MSZoning']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "C (all)":
        columnReplacement.append(74528)
    elif eachItem == "FV":
        columnReplacement.append(214014)
    elif eachItem == "RH":
        columnReplacement.append(131558)
    elif eachItem == "RL":
        columnReplacement.append(191004)
    elif eachItem == "RM":
        columnReplacement.append(126316)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['MSZoning'], axis=1)
trainCSV['MSZoningReplacement'] = columnReplacement

# print(trainCSV)
# ---------------------------------------------LotFrontage
colToBeReplaced = trainCSV['LotFrontage']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "NA":
        columnReplacement.append(0)
    else:
        columnReplacement.append(eachItem)

trainCSV = trainCSV.drop(['LotFrontage'], axis=1)
trainCSV['LotFrontageReplacement'] = columnReplacement
# -------------------------------------------------Alley

colToBeReplaced = trainCSV['Alley']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(183452)
    elif eachItem == "Grvl":
        columnReplacement.append(122219)
    elif eachItem == "Pave":
        columnReplacement.append(168000)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Alley'], axis=1)
trainCSV['AlleyReplacement'] = columnReplacement

# ----------------------------------------LotShape

colToBeReplaced = trainCSV['LotShape']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "IR1":
        columnReplacement.append(206101)
    elif eachItem == "IR2":
        columnReplacement.append(239833)
    elif eachItem == "IR3":
        columnReplacement.append(216036)
    elif eachItem == "Reg":
        columnReplacement.append(164754)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['LotShape'], axis=1)
trainCSV['LotShapeReplacement'] = columnReplacement
# ----------------------------------------LandContour

colToBeReplaced = trainCSV['LandContour']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Bnk":
        columnReplacement.append(143104)
    elif eachItem == "HLS":
        columnReplacement.append(231533)
    elif eachItem == "Low":
        columnReplacement.append(203661)
    elif eachItem == "Lvl":
        columnReplacement.append(180183)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['LandContour'], axis=1)
trainCSV['LandContourReplacement'] = columnReplacement
# ----------------------------------------LotConfig

colToBeReplaced = trainCSV['LotConfig']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Corner":
        columnReplacement.append(181623)
    elif eachItem == "CulDSac":
        columnReplacement.append(223854)
    elif eachItem == "FR2":
        columnReplacement.append(177934)
    elif eachItem == "FR3":
        columnReplacement.append(208475)
    elif eachItem == "Inside":
        columnReplacement.append(176938)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['LotConfig'], axis=1)
trainCSV['LotConfigReplacement'] = columnReplacement
# -----------------------------------------LandSlope

colToBeReplaced = trainCSV['LandSlope']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Gtl":
        columnReplacement.append(179956)
    elif eachItem == "Mod":
        columnReplacement.append(196734)
    elif eachItem == "Sev":
        columnReplacement.append(204379)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['LandSlope'], axis=1)
trainCSV['LandSlopeReplacement'] = columnReplacement
# -----------------------------------------Neighborhood

colToBeReplaced = trainCSV['Neighborhood']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Blmngtn":
        columnReplacement.append(194870)
    elif eachItem == "Blueste":
        columnReplacement.append(137500)
    elif eachItem == "BrDale":
        columnReplacement.append(104493)
    elif eachItem == "BrkSide":
        columnReplacement.append(124834)
    elif eachItem == "ClearCr":
        columnReplacement.append(212565)
    elif eachItem == "CollgCr":
        columnReplacement.append(197965)
    elif eachItem == "Crawfor":
        columnReplacement.append(210624)
    elif eachItem == "Edwards":
        columnReplacement.append(128219)
    elif eachItem == "Gilbert":
        columnReplacement.append(192854)
    elif eachItem == "IDOTRR":
        columnReplacement.append(100123)
    elif eachItem == "MeadowV":
        columnReplacement.append(98576)
    elif eachItem == "Mitchel":
        columnReplacement.append(156270)
    elif eachItem == "NAmes":
        columnReplacement.append(145847)
    elif eachItem == "NPkVill":
        columnReplacement.append(142694)
    elif eachItem == "NWAmes":
        columnReplacement.append(189050)
    elif eachItem == "NoRidge":
        columnReplacement.append(335295)
    elif eachItem == "NridgHt":
        columnReplacement.append(316270)
    elif eachItem == "OldTown":
        columnReplacement.append(128225)
    elif eachItem == "SWISU":
        columnReplacement.append(142591)
    elif eachItem == "Sawyer":
        columnReplacement.append(136793)
    elif eachItem == "SawyerW":
        columnReplacement.append(186555)
    elif eachItem == "Somerst":
        columnReplacement.append(225379)
    elif eachItem == "StoneBr":
        columnReplacement.append(310499)
    elif eachItem == "Timber":
        columnReplacement.append(242247)
    elif eachItem == "Veenker":
        columnReplacement.append(238772)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Neighborhood'], axis=1)
trainCSV['NeighborhoodReplacement'] = columnReplacement
# -----------------------------------------Condition1


colToBeReplaced = trainCSV['Condition1']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Artery":
        columnReplacement.append(135091)
    elif eachItem == "Feedr":
        columnReplacement.append(142475)
    elif eachItem == "Norm":
        columnReplacement.append(184495)
    elif eachItem == "PosA":
        columnReplacement.append(225875)
    elif eachItem == "PosN":
        columnReplacement.append(215184)
    elif eachItem == "RRAe":
        columnReplacement.append(138400)
    elif eachItem == "RRAn":
        columnReplacement.append(184396)
    elif eachItem == "RRNe":
        columnReplacement.append(190750)
    elif eachItem == "RRNn":
        columnReplacement.append(212400)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Condition1'], axis=1)
trainCSV['Condition1Replacement'] = columnReplacement
# -----------------------------------------BldgType

colToBeReplaced = trainCSV['BldgType']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "1Fam":
        columnReplacement.append(185763)
    elif eachItem == "2fmCon":
        columnReplacement.append(128432)
    elif eachItem == "Duplex":
        columnReplacement.append(133541)
    elif eachItem == "Twnhs":
        columnReplacement.append(135911)
    elif eachItem == "TwnhsE":
        columnReplacement.append(181959)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['BldgType'], axis=1)
trainCSV['BldgTypeReplacement'] = columnReplacement

# -----------------------------------------HouseStyle


colToBeReplaced = trainCSV['HouseStyle']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "1.5Fin":
        columnReplacement.append(143116)
    elif eachItem == "1.5Unf":
        columnReplacement.append(110150)
    elif eachItem == "1Story":
        columnReplacement.append(175985)
    elif eachItem == "2.5Fin":
        columnReplacement.append(220000)
    elif eachItem == "2.5Unf":
        columnReplacement.append(157354)
    elif eachItem == "2Story":
        columnReplacement.append(210051)
    elif eachItem == "SFoyer":
        columnReplacement.append(135074)
    elif eachItem == "SLvl":
        columnReplacement.append(166703)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['HouseStyle'], axis=1)
trainCSV['HouseStyleReplacement'] = columnReplacement
# -----------------------------------------RoofStyle

colToBeReplaced = trainCSV['RoofStyle']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Flat":
        columnReplacement.append(194690)
    elif eachItem == "Gable":
        columnReplacement.append(171483)
    elif eachItem == "Gambrel":
        columnReplacement.append(148909)
    elif eachItem == "Hip":
        columnReplacement.append(218876)
    elif eachItem == "Mansard":
        columnReplacement.append(180568)
    elif eachItem == "Shed":
        columnReplacement.append(225000)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['RoofStyle'], axis=1)
trainCSV['RoofStyleReplacement'] = columnReplacement
# -----------------------------------------RoofMatl

colToBeReplaced = trainCSV['RoofMatl']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "ClyTile":
        columnReplacement.append(160000)
    elif eachItem == "CompShg":
        columnReplacement.append(179803)
    elif eachItem == "Membran":
        columnReplacement.append(241500)
    elif eachItem == "Metal":
        columnReplacement.append(180000)
    elif eachItem == "Roll":
        columnReplacement.append(137000)
    elif eachItem == "Tar&Grv":
        columnReplacement.append(185406)
    elif eachItem == "WdShake":
        columnReplacement.append(241400)
    elif eachItem == "WdShngl":
        columnReplacement.append(390250)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['RoofMatl'], axis=1)
trainCSV['RoofMatlReplacement'] = columnReplacement
# -----------------------------------------Exterior1st

colToBeReplaced = trainCSV['Exterior1st']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "AsbShng":
        columnReplacement.append(107385)
    elif eachItem == "AsphShn":
        columnReplacement.append(100000)
    elif eachItem == "BrkComm":
        columnReplacement.append(71000)
    elif eachItem == "BrkFace":
        columnReplacement.append(194573)
    elif eachItem == "CBlock":
        columnReplacement.append(105000)
    elif eachItem == "CemntBd":
        columnReplacement.append(231690)
    elif eachItem == "HdBoard":
        columnReplacement.append(163077)
    elif eachItem == "ImStucc":
        columnReplacement.append(262000)
    elif eachItem == "MetalSd":
        columnReplacement.append(149422)
    elif eachItem == "Plywood":
        columnReplacement.append(175942)
    elif eachItem == "Stone":
        columnReplacement.append(258500)
    elif eachItem == "Stucco":
        columnReplacement.append(162990)
    elif eachItem == "VinylSd":
        columnReplacement.append(213732)
    elif eachItem == "Wd Sdng":
        columnReplacement.append(149841)
    elif eachItem == "WdShing":
        columnReplacement.append(150655)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Exterior1st'], axis=1)
trainCSV['Exterior1stReplacement'] = columnReplacement
# -----------------------------------------Exterior2nd

colToBeReplaced = trainCSV['Exterior2nd']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "AsbShng":
        columnReplacement.append(114060)
    elif eachItem == "AsphShn":
        columnReplacement.append(138000)
    elif eachItem == "Brk Cmn":
        columnReplacement.append(126714)
    elif eachItem == "BrkFace":
        columnReplacement.append(195818)
    elif eachItem == "CBlock":
        columnReplacement.append(105000)
    elif eachItem == "CmentBd":
        columnReplacement.append(230093)
    elif eachItem == "HdBoard":
        columnReplacement.append(167661)
    elif eachItem == "ImStucc":
        columnReplacement.append(252070)
    elif eachItem == "MetalSd":
        columnReplacement.append(149803)
    elif eachItem == "Other":
        columnReplacement.append(319000)
    elif eachItem == "Plywood":
        columnReplacement.append(168112)
    elif eachItem == "Stone":
        columnReplacement.append(158224)
    elif eachItem == "Stucco":
        columnReplacement.append(155905)
    elif eachItem == "VinylSd":
        columnReplacement.append(214432)
    elif eachItem == "Wd Sdng":
        columnReplacement.append(148386)
    elif eachItem == "Wd Shng":
        columnReplacement.append(161328)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Exterior2nd'], axis=1)
trainCSV['Exterior2ndReplacement'] = columnReplacement
# -----------------------------------------MasVnrType

colToBeReplaced = trainCSV['MasVnrType']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(236484)
    elif eachItem == "BrkCmn":
        columnReplacement.append(146318)
    elif eachItem == "BrkFace":
        columnReplacement.append(204691)
    elif eachItem == "None":
        columnReplacement.append(156221)
    elif eachItem == "Stone":
        columnReplacement.append(265583)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['MasVnrType'], axis=1)
trainCSV['MasVnrTypeReplacement'] = columnReplacement

# -----------------------------------------ExterQual

colToBeReplaced = trainCSV['ExterQual']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Ex":
        columnReplacement.append(367360)
    elif eachItem == "Fa":
        columnReplacement.append(87985)
    elif eachItem == "Gd":
        columnReplacement.append(231633)
    elif eachItem == "TA":
        columnReplacement.append(144341)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['ExterQual'], axis=1)
trainCSV['ExterQualReplacement'] = columnReplacement
# -----------------------------------------ExterCond


colToBeReplaced = trainCSV['ExterCond']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Ex":
        columnReplacement.append(201333)
    elif eachItem == "Fa":
        columnReplacement.append(102595)
    elif eachItem == "Gd":
        columnReplacement.append(168897)
    elif eachItem == "Po":
        columnReplacement.append(76500)
    elif eachItem == "TA":
        columnReplacement.append(184034)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['ExterCond'], axis=1)
trainCSV['ExterCondReplacement'] = columnReplacement
# -----------------------------------------Foundation

colToBeReplaced = trainCSV['Foundation']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "BrkTil":
        columnReplacement.append(132291)
    elif eachItem == "CBlock":
        columnReplacement.append(149805)
    elif eachItem == "PConc":
        columnReplacement.append(225230)
    elif eachItem == "Slab":
        columnReplacement.append(107365)
    elif eachItem == "Stone":
        columnReplacement.append(165959)
    elif eachItem == "Wood":
        columnReplacement.append(185666)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Foundation'], axis=1)
trainCSV['FoundationReplacement'] = columnReplacement
# -----------------------------------------BsmtQual

colToBeReplaced = trainCSV['BsmtQual']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(105652)
    elif eachItem == "Ex":
        columnReplacement.append(327041)
    elif eachItem == "Fa":
        columnReplacement.append(115692)
    elif eachItem == "Gd":
        columnReplacement.append(202688)
    elif eachItem == "TA":
        columnReplacement.append(140759)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['BsmtQual'], axis=1)
trainCSV['BsmtQualReplacement'] = columnReplacement
# -----------------------------------------BsmtCond

colToBeReplaced = trainCSV['BsmtCond']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(105652)
    elif eachItem == "Fa":
        columnReplacement.append(121809)
    elif eachItem == "Gd":
        columnReplacement.append(213599)
    elif eachItem == "Po":
        columnReplacement.append(64000)
    elif eachItem == "TA":
        columnReplacement.append(183632)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['BsmtCond'], axis=1)
trainCSV['BsmtCondReplacement'] = columnReplacement
# -----------------------------------------BsmtExposure

colToBeReplaced = trainCSV['BsmtExposure']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(107938)
    elif eachItem == "Av":
        columnReplacement.append(206643)
    elif eachItem == "Gd":
        columnReplacement.append(257689)
    elif eachItem == "Mn":
        columnReplacement.append(192789)
    elif eachItem == "No":
        columnReplacement.append(165652)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['BsmtExposure'], axis=1)
trainCSV['BsmtExposureReplacement'] = columnReplacement
# -----------------------------------------BsmtFinType1

colToBeReplaced = trainCSV['BsmtFinType1']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(105652)
    elif eachItem == "ALQ":
        columnReplacement.append(161573)
    elif eachItem == "BLQ":
        columnReplacement.append(149493)
    elif eachItem == "GLQ":
        columnReplacement.append(235413)
    elif eachItem == "LwQ":
        columnReplacement.append(151852)
    elif eachItem == "Rec":
        columnReplacement.append(146889)
    elif eachItem == "Unf":
        columnReplacement.append(170670)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['BsmtFinType1'], axis=1)
trainCSV['BsmtFinType1Replacement'] = columnReplacement
# -----------------------------------------BsmtFinType2

colToBeReplaced = trainCSV['BsmtFinType2']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(110346)
    elif eachItem == "ALQ":
        columnReplacement.append(209942)
    elif eachItem == "BLQ":
        columnReplacement.append(151101)
    elif eachItem == "GLQ":
        columnReplacement.append(180982)
    elif eachItem == "LwQ":
        columnReplacement.append(164364)
    elif eachItem == "Rec":
        columnReplacement.append(164917)
    elif eachItem == "Unf":
        columnReplacement.append(184694)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['BsmtFinType2'], axis=1)
trainCSV['BsmtFinType2Replacement'] = columnReplacement
# -----------------------------------------Heating

colToBeReplaced = trainCSV['Heating']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Floor":
        columnReplacement.append(72500)
    elif eachItem == "GasA":
        columnReplacement.append(182021)
    elif eachItem == "GasW":
        columnReplacement.append(166632)
    elif eachItem == "Grav":
        columnReplacement.append(75271)
    elif eachItem == "OthW":
        columnReplacement.append(125750)
    elif eachItem == "Wall":
        columnReplacement.append(92100)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Heating'], axis=1)
trainCSV['HeatingReplacement'] = columnReplacement
# -----------------------------------------HeatingQC

colToBeReplaced = trainCSV['HeatingQC']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Ex":
        columnReplacement.append(214914)
    elif eachItem == "Fa":
        columnReplacement.append(123919)
    elif eachItem == "Gd":
        columnReplacement.append(156858)
    elif eachItem == "Po":
        columnReplacement.append(87000)
    elif eachItem == "TA":
        columnReplacement.append(142362)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['HeatingQC'], axis=1)
trainCSV['HeatingQCReplacement'] = columnReplacement
# -----------------------------------------CentralAir

colToBeReplaced = trainCSV['CentralAir']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "N":
        columnReplacement.append(105264)
    elif eachItem == "Y":
        columnReplacement.append(186186)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['CentralAir'], axis=1)
trainCSV['CentralAirReplacement'] = columnReplacement
# -----------------------------------------Electrical

colToBeReplaced = trainCSV['Electrical']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(167500)
    elif eachItem == "FuseA":
        columnReplacement.append(122196)
    elif eachItem == "FuseF":
        columnReplacement.append(107675)
    elif eachItem == "FuseP":
        columnReplacement.append(97333)
    elif eachItem == "Mix":
        columnReplacement.append(67000)
    elif eachItem == "SBrkr":
        columnReplacement.append(186825)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Electrical'], axis=1)
trainCSV['ElectricalReplacement'] = columnReplacement
# -----------------------------------------KitchenQual


colToBeReplaced = trainCSV['KitchenQual']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Ex":
        columnReplacement.append(328554)
    elif eachItem == "Fa":
        columnReplacement.append(105565)
    elif eachItem == "Gd":
        columnReplacement.append(212116)
    elif eachItem == "TA":
        columnReplacement.append(139962)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['KitchenQual'], axis=1)
trainCSV['KitchenQualReplacement'] = columnReplacement
# -----------------------------------------Functional

colToBeReplaced = trainCSV['Functional']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Maj1":
        columnReplacement.append(153948)
    elif eachItem == "Maj2":
        columnReplacement.append(85800)
    elif eachItem == "Min1":
        columnReplacement.append(146385)
    elif eachItem == "Min2":
        columnReplacement.append(144240)
    elif eachItem == "Mod":
        columnReplacement.append(168393)
    elif eachItem == "Sev":
        columnReplacement.append(129000)
    elif eachItem == "Typ":
        columnReplacement.append(183429)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Functional'], axis=1)
trainCSV['FunctionalReplacement'] = columnReplacement
# -----------------------------------------FireplaceQu

colToBeReplaced = trainCSV['FireplaceQu']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(141331)
    elif eachItem == "Ex":
        columnReplacement.append(337712)
    elif eachItem == "Fa":
        columnReplacement.append(167298)
    elif eachItem == "Gd":
        columnReplacement.append(226351)
    elif eachItem == "Po":
        columnReplacement.append(129764)
    elif eachItem == "TA":
        columnReplacement.append(205723)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['FireplaceQu'], axis=1)
trainCSV['FireplaceQuReplacement'] = columnReplacement
# -----------------------------------------GarageType

colToBeReplaced = trainCSV['GarageType']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(103317)
    elif eachItem == "2Types":
        columnReplacement.append(151283)
    elif eachItem == "Attchd":
        columnReplacement.append(202892)
    elif eachItem == "Basment":
        columnReplacement.append(160570)
    elif eachItem == "BuiltIn":
        columnReplacement.append(254751)
    elif eachItem == "CarPort":
        columnReplacement.append(109962)
    elif eachItem == "Detchd":
        columnReplacement.append(134091)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['GarageType'], axis=1)
trainCSV['GarageTypeReplacement'] = columnReplacement
# -----------------------------------------GarageYrBlt

colToBeReplaced = trainCSV['GarageYrBlt']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == 0:
        columnReplacement.append(0)
    else:
        columnReplacement.append(int(eachItem) - 1900)

trainCSV = trainCSV.drop(['GarageYrBlt'], axis=1)
trainCSV['GarageYrBltReplacement'] = columnReplacement

# ---------------------------------------------GarageFinish

colToBeReplaced = trainCSV['GarageFinish']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(103317)
    elif eachItem == "Fin":
        columnReplacement.append(240052)
    elif eachItem == "RFn":
        columnReplacement.append(202068)
    elif eachItem == "Unf":
        columnReplacement.append(142156)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['GarageFinish'], axis=1)
trainCSV['GarageFinishReplacement'] = columnReplacement
# ---------------------------------------------GarageQual

colToBeReplaced = trainCSV['GarageQual']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(103317)
    elif eachItem == "Ex":
        columnReplacement.append(241000)
    elif eachItem == "Fa":
        columnReplacement.append(123573)
    elif eachItem == "Gd":
        columnReplacement.append(215860)
    elif eachItem == "Po":
        columnReplacement.append(100166)
    elif eachItem == "TA":
        columnReplacement.append(187489)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['GarageQual'], axis=1)
trainCSV['GarageQualReplacement'] = columnReplacement
# ---------------------------------------------GarageCond

colToBeReplaced = trainCSV['GarageCond']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(103317)
    elif eachItem == "Ex":
        columnReplacement.append(124000)
    elif eachItem == "Fa":
        columnReplacement.append(114654)
    elif eachItem == "Gd":
        columnReplacement.append(179930)
    elif eachItem == "Po":
        columnReplacement.append(108500)
    elif eachItem == "TA":
        columnReplacement.append(187885)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['GarageCond'], axis=1)
trainCSV['GarageCondReplacement'] = columnReplacement
# ---------------------------------------------GarageCond

colToBeReplaced = trainCSV['PavedDrive']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "N":
        columnReplacement.append(115039)
    elif eachItem == "P":
        columnReplacement.append(132330)
    elif eachItem == "Y":
        columnReplacement.append(186433)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['PavedDrive'], axis=1)
trainCSV['PavedDriveReplacement'] = columnReplacement
# ---------------------------------------------PoolQC

colToBeReplaced = trainCSV['PoolQC']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(180404)
    elif eachItem == "Ex":
        columnReplacement.append(490000)
    elif eachItem == "Fa":
        columnReplacement.append(215500)
    elif eachItem == "Gd":
        columnReplacement.append(201990)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['PoolQC'], axis=1)
trainCSV['PoolQCReplacement'] = columnReplacement
# ---------------------------------------------PoolQC

colToBeReplaced = trainCSV['Fence']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(187596)
    elif eachItem == "GdPrv":
        columnReplacement.append(178927)
    elif eachItem == "GdWo":
        columnReplacement.append(140379)
    elif eachItem == "MnPrv":
        columnReplacement.append(148751)
    elif eachItem == "MnWw":
        columnReplacement.append(134286)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['Fence'], axis=1)
trainCSV['FenceReplacement'] = columnReplacement
# ---------------------------------------------MiscFeature

colToBeReplaced = trainCSV['MiscFeature']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(182046)
    elif eachItem == "Gar2":
        columnReplacement.append(170750)
    elif eachItem == "Othr":
        columnReplacement.append(94000)
    elif eachItem == "Shed":
        columnReplacement.append(151187)
    elif eachItem == "TenC":
        columnReplacement.append(250000)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['MiscFeature'], axis=1)
trainCSV['MiscFeatureReplacement'] = columnReplacement
# ---------------------------------------------SaleType

colToBeReplaced = trainCSV['SaleType']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "COD":
        columnReplacement.append(143973)
    elif eachItem == "CWD":
        columnReplacement.append(210600)
    elif eachItem == "Con":
        columnReplacement.append(269600)
    elif eachItem == "ConLD":
        columnReplacement.append(138780)
    elif eachItem == "ConLI":
        columnReplacement.append(200390)
    elif eachItem == "ConLw":
        columnReplacement.append(143700)
    elif eachItem == "New":
        columnReplacement.append(274945)
    elif eachItem == "Oth":
        columnReplacement.append(119850)
    elif eachItem == "WD":
        columnReplacement.append(173401)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['SaleType'], axis=1)
trainCSV['SaleTypeReplacement'] = columnReplacement
# ---------------------------------------------SaleCondition

colToBeReplaced = trainCSV['SaleCondition']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Abnorml":
        columnReplacement.append(146526)
    elif eachItem == "AdjLand":
        columnReplacement.append(104125)
    elif eachItem == "Alloca":
        columnReplacement.append(167377)
    elif eachItem == "Family":
        columnReplacement.append(149600)
    elif eachItem == "Normal":
        columnReplacement.append(175202)
    elif eachItem == "Partial":
        columnReplacement.append(272291)
    else:
        columnReplacement.append(0)
trainCSV = trainCSV.drop(['SaleCondition'], axis=1)
trainCSV['SaleConditionReplacement'] = columnReplacement

import numpy as np

labelNP = np.array(trainCSV['SalePrice'])
labelNP = np.reshape(labelNP, [-1, 1])
trainCSV = trainCSV.drop(['SalePrice'], axis=1)
featureNP = np.array(trainCSV)
print(featureNP.shape)

import tensorflow as tf

y = tf.placeholder(tf.float32, shape=[None, 1])

m1 = tf.Variable(np.abs(np.array((tf.truncated_normal(shape=[76, 100])))))
m2 = tf.Variable(np.abs(np.array((tf.truncated_normal(shape=[100, 64])))))
m3 = tf.Variable(np.abs(np.array((tf.truncated_normal(shape=[64, 16])))))
m4 = tf.Variable(np.abs(np.array((tf.truncated_normal(shape=[16, 1])))))

x = tf.placeholder(tf.float32, shape=[None, 76])

b1 = tf.Variable(np.abs((tf.truncated_normal(shape=[100]))))
b2 = tf.Variable(np.abs((tf.truncated_normal(shape=[64]))))
b3 = tf.Variable(np.abs((tf.truncated_normal(shape=[16]))))
b4 = tf.Variable(np.abs((tf.truncated_normal(shape=[1]))))

mx_b = tf.add(tf.matmul(x, m1), b1)
mx_b = tf.add(tf.matmul(mx_b, m2), b2)
mx_b = tf.add(tf.matmul(mx_b, m3), b3)
mx_b = tf.add(tf.matmul(mx_b, m4), b4)
mx_b = tf.abs(mx_b)

lr = tf.placeholder(tf.float32)

loss = tf.reduce_mean(mx_b - y)
trainingStep = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run([mx_b, m1, m2, m3, m4], feed_dict={x: featureNP}))

whileLoopFactor = 500
i = 0
while whileLoopFactor > 0.000001:
    i = i + 1
    print(sess.run([trainingStep, loss], feed_dict={x: featureNP, y: labelNP, lr: 0.01 - (1 / (100 * (i + 1)))}))
    whileLoopFactor = np.sum(
        np.array(sess.run([loss], feed_dict={x: featureNP, y: labelNP, lr: 0.01 - (1 / (100 * (i + 1)))})))
    # print(whileLoopFactor, i)

print(sess.run([mx_b, m1, m2, m3, m4], feed_dict={x: featureNP}))

# *--*-*-**-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*-*-*-*-*-*---*--*-*-*--*

testCSV = pd.read_csv(inputFile + "test.csv")
testCSV = testCSV.fillna(0)
# print(testCSV)
idRow = testCSV['Id']

testCSV = testCSV.drop(['Id', 'Street', 'Utilities', 'Condition2'], axis=1)

# --------------------------------------------MSZoning
colToBeReplaced = testCSV['MSZoning']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "C (all)":
        columnReplacement.append(74528)
    elif eachItem == "FV":
        columnReplacement.append(214014)
    elif eachItem == "RH":
        columnReplacement.append(131558)
    elif eachItem == "RL":
        columnReplacement.append(191004)
    elif eachItem == "RM":
        columnReplacement.append(126316)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['MSZoning'], axis=1)
testCSV['MSZoningReplacement'] = columnReplacement

# print(testCSV)
# ---------------------------------------------LotFrontage
colToBeReplaced = testCSV['LotFrontage']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "NA":
        columnReplacement.append(0)
    else:
        columnReplacement.append(eachItem)

testCSV = testCSV.drop(['LotFrontage'], axis=1)
testCSV['LotFrontageReplacement'] = columnReplacement
# -------------------------------------------------Alley

colToBeReplaced = testCSV['Alley']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(183452)
    elif eachItem == "Grvl":
        columnReplacement.append(122219)
    elif eachItem == "Pave":
        columnReplacement.append(168000)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Alley'], axis=1)
testCSV['AlleyReplacement'] = columnReplacement

# ----------------------------------------LotShape

colToBeReplaced = testCSV['LotShape']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "IR1":
        columnReplacement.append(206101)
    elif eachItem == "IR2":
        columnReplacement.append(239833)
    elif eachItem == "IR3":
        columnReplacement.append(216036)
    elif eachItem == "Reg":
        columnReplacement.append(164754)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['LotShape'], axis=1)
testCSV['LotShapeReplacement'] = columnReplacement
# ----------------------------------------LandContour

colToBeReplaced = testCSV['LandContour']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Bnk":
        columnReplacement.append(143104)
    elif eachItem == "HLS":
        columnReplacement.append(231533)
    elif eachItem == "Low":
        columnReplacement.append(203661)
    elif eachItem == "Lvl":
        columnReplacement.append(180183)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['LandContour'], axis=1)
testCSV['LandContourReplacement'] = columnReplacement
# ----------------------------------------LotConfig

colToBeReplaced = testCSV['LotConfig']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Corner":
        columnReplacement.append(181623)
    elif eachItem == "CulDSac":
        columnReplacement.append(223854)
    elif eachItem == "FR2":
        columnReplacement.append(177934)
    elif eachItem == "FR3":
        columnReplacement.append(208475)
    elif eachItem == "Inside":
        columnReplacement.append(176938)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['LotConfig'], axis=1)
testCSV['LotConfigReplacement'] = columnReplacement
# -----------------------------------------LandSlope

colToBeReplaced = testCSV['LandSlope']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Gtl":
        columnReplacement.append(179956)
    elif eachItem == "Mod":
        columnReplacement.append(196734)
    elif eachItem == "Sev":
        columnReplacement.append(204379)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['LandSlope'], axis=1)
testCSV['LandSlopeReplacement'] = columnReplacement
# -----------------------------------------Neighborhood

colToBeReplaced = testCSV['Neighborhood']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Blmngtn":
        columnReplacement.append(194870)
    elif eachItem == "Blueste":
        columnReplacement.append(137500)
    elif eachItem == "BrDale":
        columnReplacement.append(104493)
    elif eachItem == "BrkSide":
        columnReplacement.append(124834)
    elif eachItem == "ClearCr":
        columnReplacement.append(212565)
    elif eachItem == "CollgCr":
        columnReplacement.append(197965)
    elif eachItem == "Crawfor":
        columnReplacement.append(210624)
    elif eachItem == "Edwards":
        columnReplacement.append(128219)
    elif eachItem == "Gilbert":
        columnReplacement.append(192854)
    elif eachItem == "IDOTRR":
        columnReplacement.append(100123)
    elif eachItem == "MeadowV":
        columnReplacement.append(98576)
    elif eachItem == "Mitchel":
        columnReplacement.append(156270)
    elif eachItem == "NAmes":
        columnReplacement.append(145847)
    elif eachItem == "NPkVill":
        columnReplacement.append(142694)
    elif eachItem == "NWAmes":
        columnReplacement.append(189050)
    elif eachItem == "NoRidge":
        columnReplacement.append(335295)
    elif eachItem == "NridgHt":
        columnReplacement.append(316270)
    elif eachItem == "OldTown":
        columnReplacement.append(128225)
    elif eachItem == "SWISU":
        columnReplacement.append(142591)
    elif eachItem == "Sawyer":
        columnReplacement.append(136793)
    elif eachItem == "SawyerW":
        columnReplacement.append(186555)
    elif eachItem == "Somerst":
        columnReplacement.append(225379)
    elif eachItem == "StoneBr":
        columnReplacement.append(310499)
    elif eachItem == "Timber":
        columnReplacement.append(242247)
    elif eachItem == "Veenker":
        columnReplacement.append(238772)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Neighborhood'], axis=1)
testCSV['NeighborhoodReplacement'] = columnReplacement
# -----------------------------------------Condition1


colToBeReplaced = testCSV['Condition1']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Artery":
        columnReplacement.append(135091)
    elif eachItem == "Feedr":
        columnReplacement.append(142475)
    elif eachItem == "Norm":
        columnReplacement.append(184495)
    elif eachItem == "PosA":
        columnReplacement.append(225875)
    elif eachItem == "PosN":
        columnReplacement.append(215184)
    elif eachItem == "RRAe":
        columnReplacement.append(138400)
    elif eachItem == "RRAn":
        columnReplacement.append(184396)
    elif eachItem == "RRNe":
        columnReplacement.append(190750)
    elif eachItem == "RRNn":
        columnReplacement.append(212400)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Condition1'], axis=1)
testCSV['Condition1Replacement'] = columnReplacement
# -----------------------------------------BldgType

colToBeReplaced = testCSV['BldgType']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "1Fam":
        columnReplacement.append(185763)
    elif eachItem == "2fmCon":
        columnReplacement.append(128432)
    elif eachItem == "Duplex":
        columnReplacement.append(133541)
    elif eachItem == "Twnhs":
        columnReplacement.append(135911)
    elif eachItem == "TwnhsE":
        columnReplacement.append(181959)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['BldgType'], axis=1)
testCSV['BldgTypeReplacement'] = columnReplacement

# -----------------------------------------HouseStyle


colToBeReplaced = testCSV['HouseStyle']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "1.5Fin":
        columnReplacement.append(143116)
    elif eachItem == "1.5Unf":
        columnReplacement.append(110150)
    elif eachItem == "1Story":
        columnReplacement.append(175985)
    elif eachItem == "2.5Fin":
        columnReplacement.append(220000)
    elif eachItem == "2.5Unf":
        columnReplacement.append(157354)
    elif eachItem == "2Story":
        columnReplacement.append(210051)
    elif eachItem == "SFoyer":
        columnReplacement.append(135074)
    elif eachItem == "SLvl":
        columnReplacement.append(166703)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['HouseStyle'], axis=1)
testCSV['HouseStyleReplacement'] = columnReplacement
# -----------------------------------------RoofStyle

colToBeReplaced = testCSV['RoofStyle']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Flat":
        columnReplacement.append(194690)
    elif eachItem == "Gable":
        columnReplacement.append(171483)
    elif eachItem == "Gambrel":
        columnReplacement.append(148909)
    elif eachItem == "Hip":
        columnReplacement.append(218876)
    elif eachItem == "Mansard":
        columnReplacement.append(180568)
    elif eachItem == "Shed":
        columnReplacement.append(225000)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['RoofStyle'], axis=1)
testCSV['RoofStyleReplacement'] = columnReplacement
# -----------------------------------------RoofMatl

colToBeReplaced = testCSV['RoofMatl']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "ClyTile":
        columnReplacement.append(160000)
    elif eachItem == "CompShg":
        columnReplacement.append(179803)
    elif eachItem == "Membran":
        columnReplacement.append(241500)
    elif eachItem == "Metal":
        columnReplacement.append(180000)
    elif eachItem == "Roll":
        columnReplacement.append(137000)
    elif eachItem == "Tar&Grv":
        columnReplacement.append(185406)
    elif eachItem == "WdShake":
        columnReplacement.append(241400)
    elif eachItem == "WdShngl":
        columnReplacement.append(390250)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['RoofMatl'], axis=1)
testCSV['RoofMatlReplacement'] = columnReplacement
# -----------------------------------------Exterior1st

colToBeReplaced = testCSV['Exterior1st']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "AsbShng":
        columnReplacement.append(107385)
    elif eachItem == "AsphShn":
        columnReplacement.append(100000)
    elif eachItem == "BrkComm":
        columnReplacement.append(71000)
    elif eachItem == "BrkFace":
        columnReplacement.append(194573)
    elif eachItem == "CBlock":
        columnReplacement.append(105000)
    elif eachItem == "CemntBd":
        columnReplacement.append(231690)
    elif eachItem == "HdBoard":
        columnReplacement.append(163077)
    elif eachItem == "ImStucc":
        columnReplacement.append(262000)
    elif eachItem == "MetalSd":
        columnReplacement.append(149422)
    elif eachItem == "Plywood":
        columnReplacement.append(175942)
    elif eachItem == "Stone":
        columnReplacement.append(258500)
    elif eachItem == "Stucco":
        columnReplacement.append(162990)
    elif eachItem == "VinylSd":
        columnReplacement.append(213732)
    elif eachItem == "Wd Sdng":
        columnReplacement.append(149841)
    elif eachItem == "WdShing":
        columnReplacement.append(150655)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Exterior1st'], axis=1)
testCSV['Exterior1stReplacement'] = columnReplacement
# -----------------------------------------Exterior2nd

colToBeReplaced = testCSV['Exterior2nd']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "AsbShng":
        columnReplacement.append(114060)
    elif eachItem == "AsphShn":
        columnReplacement.append(138000)
    elif eachItem == "Brk Cmn":
        columnReplacement.append(126714)
    elif eachItem == "BrkFace":
        columnReplacement.append(195818)
    elif eachItem == "CBlock":
        columnReplacement.append(105000)
    elif eachItem == "CmentBd":
        columnReplacement.append(230093)
    elif eachItem == "HdBoard":
        columnReplacement.append(167661)
    elif eachItem == "ImStucc":
        columnReplacement.append(252070)
    elif eachItem == "MetalSd":
        columnReplacement.append(149803)
    elif eachItem == "Other":
        columnReplacement.append(319000)
    elif eachItem == "Plywood":
        columnReplacement.append(168112)
    elif eachItem == "Stone":
        columnReplacement.append(158224)
    elif eachItem == "Stucco":
        columnReplacement.append(155905)
    elif eachItem == "VinylSd":
        columnReplacement.append(214432)
    elif eachItem == "Wd Sdng":
        columnReplacement.append(148386)
    elif eachItem == "Wd Shng":
        columnReplacement.append(161328)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Exterior2nd'], axis=1)
testCSV['Exterior2ndReplacement'] = columnReplacement
# -----------------------------------------MasVnrType

colToBeReplaced = testCSV['MasVnrType']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(236484)
    elif eachItem == "BrkCmn":
        columnReplacement.append(146318)
    elif eachItem == "BrkFace":
        columnReplacement.append(204691)
    elif eachItem == "None":
        columnReplacement.append(156221)
    elif eachItem == "Stone":
        columnReplacement.append(265583)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['MasVnrType'], axis=1)
testCSV['MasVnrTypeReplacement'] = columnReplacement

# -----------------------------------------ExterQual

colToBeReplaced = testCSV['ExterQual']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Ex":
        columnReplacement.append(367360)
    elif eachItem == "Fa":
        columnReplacement.append(87985)
    elif eachItem == "Gd":
        columnReplacement.append(231633)
    elif eachItem == "TA":
        columnReplacement.append(144341)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['ExterQual'], axis=1)
testCSV['ExterQualReplacement'] = columnReplacement
# -----------------------------------------ExterCond


colToBeReplaced = testCSV['ExterCond']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Ex":
        columnReplacement.append(201333)
    elif eachItem == "Fa":
        columnReplacement.append(102595)
    elif eachItem == "Gd":
        columnReplacement.append(168897)
    elif eachItem == "Po":
        columnReplacement.append(76500)
    elif eachItem == "TA":
        columnReplacement.append(184034)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['ExterCond'], axis=1)
testCSV['ExterCondReplacement'] = columnReplacement
# -----------------------------------------Foundation

colToBeReplaced = testCSV['Foundation']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "BrkTil":
        columnReplacement.append(132291)
    elif eachItem == "CBlock":
        columnReplacement.append(149805)
    elif eachItem == "PConc":
        columnReplacement.append(225230)
    elif eachItem == "Slab":
        columnReplacement.append(107365)
    elif eachItem == "Stone":
        columnReplacement.append(165959)
    elif eachItem == "Wood":
        columnReplacement.append(185666)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Foundation'], axis=1)
testCSV['FoundationReplacement'] = columnReplacement
# -----------------------------------------BsmtQual

colToBeReplaced = testCSV['BsmtQual']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(105652)
    elif eachItem == "Ex":
        columnReplacement.append(327041)
    elif eachItem == "Fa":
        columnReplacement.append(115692)
    elif eachItem == "Gd":
        columnReplacement.append(202688)
    elif eachItem == "TA":
        columnReplacement.append(140759)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['BsmtQual'], axis=1)
testCSV['BsmtQualReplacement'] = columnReplacement
# -----------------------------------------BsmtCond

colToBeReplaced = testCSV['BsmtCond']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(105652)
    elif eachItem == "Fa":
        columnReplacement.append(121809)
    elif eachItem == "Gd":
        columnReplacement.append(213599)
    elif eachItem == "Po":
        columnReplacement.append(64000)
    elif eachItem == "TA":
        columnReplacement.append(183632)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['BsmtCond'], axis=1)
testCSV['BsmtCondReplacement'] = columnReplacement
# -----------------------------------------BsmtExposure

colToBeReplaced = testCSV['BsmtExposure']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(107938)
    elif eachItem == "Av":
        columnReplacement.append(206643)
    elif eachItem == "Gd":
        columnReplacement.append(257689)
    elif eachItem == "Mn":
        columnReplacement.append(192789)
    elif eachItem == "No":
        columnReplacement.append(165652)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['BsmtExposure'], axis=1)
testCSV['BsmtExposureReplacement'] = columnReplacement
# -----------------------------------------BsmtFinType1

colToBeReplaced = testCSV['BsmtFinType1']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(105652)
    elif eachItem == "ALQ":
        columnReplacement.append(161573)
    elif eachItem == "BLQ":
        columnReplacement.append(149493)
    elif eachItem == "GLQ":
        columnReplacement.append(235413)
    elif eachItem == "LwQ":
        columnReplacement.append(151852)
    elif eachItem == "Rec":
        columnReplacement.append(146889)
    elif eachItem == "Unf":
        columnReplacement.append(170670)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['BsmtFinType1'], axis=1)
testCSV['BsmtFinType1Replacement'] = columnReplacement
# -----------------------------------------BsmtFinType2

colToBeReplaced = testCSV['BsmtFinType2']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(110346)
    elif eachItem == "ALQ":
        columnReplacement.append(209942)
    elif eachItem == "BLQ":
        columnReplacement.append(151101)
    elif eachItem == "GLQ":
        columnReplacement.append(180982)
    elif eachItem == "LwQ":
        columnReplacement.append(164364)
    elif eachItem == "Rec":
        columnReplacement.append(164917)
    elif eachItem == "Unf":
        columnReplacement.append(184694)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['BsmtFinType2'], axis=1)
testCSV['BsmtFinType2Replacement'] = columnReplacement
# -----------------------------------------Heating

colToBeReplaced = testCSV['Heating']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Floor":
        columnReplacement.append(72500)
    elif eachItem == "GasA":
        columnReplacement.append(182021)
    elif eachItem == "GasW":
        columnReplacement.append(166632)
    elif eachItem == "Grav":
        columnReplacement.append(75271)
    elif eachItem == "OthW":
        columnReplacement.append(125750)
    elif eachItem == "Wall":
        columnReplacement.append(92100)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Heating'], axis=1)
testCSV['HeatingReplacement'] = columnReplacement
# -----------------------------------------HeatingQC

colToBeReplaced = testCSV['HeatingQC']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Ex":
        columnReplacement.append(214914)
    elif eachItem == "Fa":
        columnReplacement.append(123919)
    elif eachItem == "Gd":
        columnReplacement.append(156858)
    elif eachItem == "Po":
        columnReplacement.append(87000)
    elif eachItem == "TA":
        columnReplacement.append(142362)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['HeatingQC'], axis=1)
testCSV['HeatingQCReplacement'] = columnReplacement
# -----------------------------------------CentralAir

colToBeReplaced = testCSV['CentralAir']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "N":
        columnReplacement.append(105264)
    elif eachItem == "Y":
        columnReplacement.append(186186)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['CentralAir'], axis=1)
testCSV['CentralAirReplacement'] = columnReplacement
# -----------------------------------------Electrical

colToBeReplaced = testCSV['Electrical']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(167500)
    elif eachItem == "FuseA":
        columnReplacement.append(122196)
    elif eachItem == "FuseF":
        columnReplacement.append(107675)
    elif eachItem == "FuseP":
        columnReplacement.append(97333)
    elif eachItem == "Mix":
        columnReplacement.append(67000)
    elif eachItem == "SBrkr":
        columnReplacement.append(186825)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Electrical'], axis=1)
testCSV['ElectricalReplacement'] = columnReplacement
# -----------------------------------------KitchenQual


colToBeReplaced = testCSV['KitchenQual']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Ex":
        columnReplacement.append(328554)
    elif eachItem == "Fa":
        columnReplacement.append(105565)
    elif eachItem == "Gd":
        columnReplacement.append(212116)
    elif eachItem == "TA":
        columnReplacement.append(139962)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['KitchenQual'], axis=1)
testCSV['KitchenQualReplacement'] = columnReplacement
# -----------------------------------------Functional

colToBeReplaced = testCSV['Functional']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Maj1":
        columnReplacement.append(153948)
    elif eachItem == "Maj2":
        columnReplacement.append(85800)
    elif eachItem == "Min1":
        columnReplacement.append(146385)
    elif eachItem == "Min2":
        columnReplacement.append(144240)
    elif eachItem == "Mod":
        columnReplacement.append(168393)
    elif eachItem == "Sev":
        columnReplacement.append(129000)
    elif eachItem == "Typ":
        columnReplacement.append(183429)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Functional'], axis=1)
testCSV['FunctionalReplacement'] = columnReplacement
# -----------------------------------------FireplaceQu

colToBeReplaced = testCSV['FireplaceQu']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(141331)
    elif eachItem == "Ex":
        columnReplacement.append(337712)
    elif eachItem == "Fa":
        columnReplacement.append(167298)
    elif eachItem == "Gd":
        columnReplacement.append(226351)
    elif eachItem == "Po":
        columnReplacement.append(129764)
    elif eachItem == "TA":
        columnReplacement.append(205723)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['FireplaceQu'], axis=1)
testCSV['FireplaceQuReplacement'] = columnReplacement
# -----------------------------------------GarageType

colToBeReplaced = testCSV['GarageType']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(103317)
    elif eachItem == "2Types":
        columnReplacement.append(151283)
    elif eachItem == "Attchd":
        columnReplacement.append(202892)
    elif eachItem == "Basment":
        columnReplacement.append(160570)
    elif eachItem == "BuiltIn":
        columnReplacement.append(254751)
    elif eachItem == "CarPort":
        columnReplacement.append(109962)
    elif eachItem == "Detchd":
        columnReplacement.append(134091)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['GarageType'], axis=1)
testCSV['GarageTypeReplacement'] = columnReplacement
# -----------------------------------------GarageYrBlt

colToBeReplaced = testCSV['GarageYrBlt']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == 0:
        columnReplacement.append(0)
    else:
        columnReplacement.append(int(eachItem) - 1900)

testCSV = testCSV.drop(['GarageYrBlt'], axis=1)
testCSV['GarageYrBltReplacement'] = columnReplacement

# ---------------------------------------------GarageFinish

colToBeReplaced = testCSV['GarageFinish']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(103317)
    elif eachItem == "Fin":
        columnReplacement.append(240052)
    elif eachItem == "RFn":
        columnReplacement.append(202068)
    elif eachItem == "Unf":
        columnReplacement.append(142156)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['GarageFinish'], axis=1)
testCSV['GarageFinishReplacement'] = columnReplacement
# ---------------------------------------------GarageQual

colToBeReplaced = testCSV['GarageQual']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(103317)
    elif eachItem == "Ex":
        columnReplacement.append(241000)
    elif eachItem == "Fa":
        columnReplacement.append(123573)
    elif eachItem == "Gd":
        columnReplacement.append(215860)
    elif eachItem == "Po":
        columnReplacement.append(100166)
    elif eachItem == "TA":
        columnReplacement.append(187489)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['GarageQual'], axis=1)
testCSV['GarageQualReplacement'] = columnReplacement
# ---------------------------------------------GarageCond

colToBeReplaced = testCSV['GarageCond']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(103317)
    elif eachItem == "Ex":
        columnReplacement.append(124000)
    elif eachItem == "Fa":
        columnReplacement.append(114654)
    elif eachItem == "Gd":
        columnReplacement.append(179930)
    elif eachItem == "Po":
        columnReplacement.append(108500)
    elif eachItem == "TA":
        columnReplacement.append(187885)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['GarageCond'], axis=1)
testCSV['GarageCondReplacement'] = columnReplacement
# ---------------------------------------------GarageCond

colToBeReplaced = testCSV['PavedDrive']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "N":
        columnReplacement.append(115039)
    elif eachItem == "P":
        columnReplacement.append(132330)
    elif eachItem == "Y":
        columnReplacement.append(186433)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['PavedDrive'], axis=1)
testCSV['PavedDriveReplacement'] = columnReplacement
# ---------------------------------------------PoolQC

colToBeReplaced = testCSV['PoolQC']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(180404)
    elif eachItem == "Ex":
        columnReplacement.append(490000)
    elif eachItem == "Fa":
        columnReplacement.append(215500)
    elif eachItem == "Gd":
        columnReplacement.append(201990)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['PoolQC'], axis=1)
testCSV['PoolQCReplacement'] = columnReplacement
# ---------------------------------------------PoolQC

colToBeReplaced = testCSV['Fence']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(187596)
    elif eachItem == "GdPrv":
        columnReplacement.append(178927)
    elif eachItem == "GdWo":
        columnReplacement.append(140379)
    elif eachItem == "MnPrv":
        columnReplacement.append(148751)
    elif eachItem == "MnWw":
        columnReplacement.append(134286)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['Fence'], axis=1)
testCSV['FenceReplacement'] = columnReplacement
# ---------------------------------------------MiscFeature

colToBeReplaced = testCSV['MiscFeature']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "0":
        columnReplacement.append(182046)
    elif eachItem == "Gar2":
        columnReplacement.append(170750)
    elif eachItem == "Othr":
        columnReplacement.append(94000)
    elif eachItem == "Shed":
        columnReplacement.append(151187)
    elif eachItem == "TenC":
        columnReplacement.append(250000)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['MiscFeature'], axis=1)
testCSV['MiscFeatureReplacement'] = columnReplacement
# ---------------------------------------------SaleType

colToBeReplaced = testCSV['SaleType']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "COD":
        columnReplacement.append(143973)
    elif eachItem == "CWD":
        columnReplacement.append(210600)
    elif eachItem == "Con":
        columnReplacement.append(269600)
    elif eachItem == "ConLD":
        columnReplacement.append(138780)
    elif eachItem == "ConLI":
        columnReplacement.append(200390)
    elif eachItem == "ConLw":
        columnReplacement.append(143700)
    elif eachItem == "New":
        columnReplacement.append(274945)
    elif eachItem == "Oth":
        columnReplacement.append(119850)
    elif eachItem == "WD":
        columnReplacement.append(173401)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['SaleType'], axis=1)
testCSV['SaleTypeReplacement'] = columnReplacement
# ---------------------------------------------SaleCondition

colToBeReplaced = testCSV['SaleCondition']
columnReplacement = []
for eachItem in colToBeReplaced:
    if eachItem == "Abnorml":
        columnReplacement.append(146526)
    elif eachItem == "AdjLand":
        columnReplacement.append(104125)
    elif eachItem == "Alloca":
        columnReplacement.append(167377)
    elif eachItem == "Family":
        columnReplacement.append(149600)
    elif eachItem == "Normal":
        columnReplacement.append(175202)
    elif eachItem == "Partial":
        columnReplacement.append(272291)
    else:
        columnReplacement.append(0)
testCSV = testCSV.drop(['SaleCondition'], axis=1)
testCSV['SaleConditionReplacement'] = columnReplacement

featureNP = np.array(testCSV)
output = (sess.run(mx_b, feed_dict={x: featureNP}))

outputNP = np.array(output)
outputNP = outputNP.flatten()

submissionPD = pd.DataFrame()
submissionPD['Id'] = idRow
submissionPD['SalePrice'] = outputNP

submissionPD.to_csv('submissionPD.csv', index=False)
