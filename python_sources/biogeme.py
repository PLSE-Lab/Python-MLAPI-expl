#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# In[ ]:





# In[ ]:


pip install biogeme


# In[ ]:


import biogeme

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.models import piecewise
import biogeme.loglikelihood as ll
import biogeme.optimization as opt


# In[ ]:


df = pd.read_csv("../input/thesisdata4/thesisdata4.dat",sep='\t')
database = db.Database("thesisdata4",df)
from headers import *

male = DefineVariable('male',gender == 1,database)
age_65_more = DefineVariable('age_65_more',age >= Numeric(65),database)
age_45_65 = DefineVariable('age_45_65',Numeric(45)<=age< Numeric(65),database)
age_20_45 = DefineVariable('age_20_45',Numeric(20)<=age< Numeric(45),database)
HHcar = DefineVariable('HHcar',hhcar / hhsize,database)
govermentaljob = DefineVariable('govermentaljob',prigov == 1,database)
high_education = DefineVariable('high_education',edu >= 4,database)


#===========================

coef_intercept = Beta('coef_intercept',0.0,None,None,0)
coef_age_65_more = Beta('coef_age_65_more',0.0,None,None,0)
coef_age_45_65 = Beta('coef_age_45_65',0.0,None,None,0)
coef_age_20_45 = Beta('coef_age_20_45',0.0,None,None,0)
coef_male = Beta('coef_male',0.0,None,None,0)
coef_HHcar = Beta('coef_HHcar',0.0,None,None,0)
coef_govermentaljob = Beta('coef_govermentaljob',0.0,None,None,0)
coef_high_education = Beta('coef_high_education',0.0,None,None,0)

#===============================

HIERARCHY = coef_intercept +coef_age_65_more * age_65_more + coef_age_45_65 * age_45_65 + coef_age_20_45 * age_20_45 +            coef_male * male + coef_HHcar * HHcar + coef_govermentaljob * govermentaljob +            coef_high_education * high_education
EGALITARIANISM = coef_intercept +coef_age_65_more * age_65_more + coef_age_45_65 * age_45_65 + coef_age_20_45 * age_20_45 +            coef_male * male + coef_HHcar * HHcar + coef_govermentaljob * govermentaljob +            coef_high_education * high_education
INDIVIDUALISM = coef_intercept + coef_age_65_more * age_65_more + coef_age_45_65 * age_45_65 + coef_age_20_45 * age_20_45 +            coef_male * male + coef_HHcar * HHcar + coef_govermentaljob * govermentaljob +            coef_high_education * high_education

#=============================

sigma_s = Beta('sigma_s',1,None,None,1)
#=========================

INTER_hierchy1 = Beta('INTER_hierchy1',0,None,None,1)
INTER_hierchy2 = Beta('INTER_hierchy2',0,None,None,0)
INTER_hierchy4 = Beta('INTER_hierchy4',0,None,None,0)
INTER_hierchy5 = Beta('INTER_hierchy5',0,None,None,0)
INTER_egaism1 = Beta('INTER_egaism1',0,None,None,1)
INTER_egaism2 = Beta('INTER_egaism2',0,None,None,0)
INTER_egaism3 = Beta('INTER_egaism3',0,None,None,0)
INTER_indism1 = Beta('INTER_indism1',0,None,None,1)
INTER_indism2 = Beta('INTER_indism2',0,None,None,0)
INTER_indism4 = Beta('INTER_indism4',0,None,None,0)

#========================================
B_hierchy1_F1 = Beta('B_hierchy1_F1',-1,None,None,1)
B_hierchy2_F1 = Beta('B_hierchy2_F1',-1,None,None,0)
B_hierchy4_F1 = Beta('B_hierchy4_F1',1,None,None,0)
B_hierchy5_F1 = Beta('B_hierchy5_F1',1,None,None,0)
B_egaism1_F1 = Beta('B_egaism1_F1',-1,None,None,1)
B_egaism2_F1 = Beta('B_egaism2_F1',-1,None,None,0)
B_egaism3_F1 = Beta('B_egaism3_F1',1,None,None,0)
B_indism1_F1 = Beta('B_indism1_F1',-1,None,None,1)
B_indism2_F1 = Beta('B_indism2_F1',-1,None,None,0)
B_indism4_F1 = Beta('B_indism4_F1',1,None,None,0)

#================================

MODEL_hierchy1 = INTER_hierchy1 + B_hierchy1_F1 * HIERARCHY
MODEL_hierchy2 = INTER_hierchy2 + B_hierchy2_F1 * HIERARCHY
MODEL_hierchy4 = INTER_hierchy4 + B_hierchy4_F1 * HIERARCHY
MODEL_hierchy5 = INTER_hierchy5 + B_hierchy5_F1 * HIERARCHY
MODEL_egaism1 = INTER_egaism1 + B_egaism1_F1 * EGALITARIANISM
MODEL_egaism2 = INTER_egaism2 + B_egaism2_F1 * EGALITARIANISM
MODEL_egaism3 = INTER_egaism3 + B_egaism3_F1 * EGALITARIANISM
MODEL_indism1 = INTER_indism1 + B_indism1_F1 * INDIVIDUALISM
MODEL_indism2 = INTER_indism2 + B_indism2_F1 * INDIVIDUALISM
MODEL_indism4 = INTER_indism4 + B_indism4_F1 * INDIVIDUALISM

#=========================

SIGMA_STAR_hierchy1 = exp(Beta('SIGMA_STAR_hierchy1',1,None,None,0))
SIGMA_STAR_hierchy2 = exp(Beta('SIGMA_STAR_hierchy2',1,None,None,0))
SIGMA_STAR_hierchy4 = exp(Beta('SIGMA_STAR_hierchy4',1,None,None,0))
SIGMA_STAR_hierchy5 = exp(Beta('SIGMA_STAR_hierchy5',1,None,None,0))
SIGMA_STAR_egaism1 = exp(Beta('SIGMA_STAR_egaism1',1,None,None,0))
SIGMA_STAR_egaism2 = exp(Beta('SIGMA_STAR_egaism2',1,None,None,0))
SIGMA_STAR_egaism3 = exp(Beta('SIGMA_STAR_egaism3',1,None,None,0))
SIGMA_STAR_indism1 = exp(Beta('SIGMA_STAR_indism1',1,None,None,0))
SIGMA_STAR_indism2 = exp(Beta('SIGMA_STAR_indism2',1,None,None,0))
SIGMA_STAR_indism4 = exp(Beta('SIGMA_STAR_indism4',1,None,None,0))

#=================

F = {}
F['egaism1'] = Elem({0:0,  1:ll.loglikelihoodregression(egaism1,MODEL_egaism1,SIGMA_STAR_egaism1)},  (egaism1 > 0)*(egaism1 < 6))
F['egaism2'] = Elem({0:0,  1:ll.loglikelihoodregression(egaism2,MODEL_egaism2,SIGMA_STAR_egaism2)},  (egaism2 > 0)*(egaism2 < 6))
F['egaism3'] = Elem({0:0,  1:ll.loglikelihoodregression(egaism3,MODEL_egaism3,SIGMA_STAR_egaism3)},  (egaism3 > 0)*(egaism3 < 6))
F['hierchy1'] = Elem({0:0,  1:ll.loglikelihoodregression(hierchy1,MODEL_hierchy1,SIGMA_STAR_hierchy1)},  (hierchy1 > 0)*(hierchy1 < 6))
F['hierchy2'] = Elem({0:0,  1:ll.loglikelihoodregression(hierchy2,MODEL_hierchy2,SIGMA_STAR_hierchy2)},  (hierchy2 > 0)*(hierchy2 < 6))
F['hierchy4'] = Elem({0:0,  1:ll.loglikelihoodregression(hierchy4,MODEL_hierchy4,SIGMA_STAR_hierchy4)},  (hierchy4 > 0)*(hierchy4 < 6))
F['hierchy5'] = Elem({0:0,  1:ll.loglikelihoodregression(hierchy5,MODEL_hierchy5,SIGMA_STAR_hierchy5)},  (hierchy5 > 0)*(hierchy5 < 6))
F['indism1'] = Elem({0:0,  1:ll.loglikelihoodregression(indism1,MODEL_indism1,SIGMA_STAR_indism1)},  (indism1 > 0)*(indism1 < 6))
F['indism2'] = Elem({0:0,  1:ll.loglikelihoodregression(indism2,MODEL_indism2,SIGMA_STAR_indism2)},  (indism2 > 0)*(indism2 < 6))
F['indism4'] = Elem({0:0,  1:ll.loglikelihoodregression(indism4,MODEL_indism4,SIGMA_STAR_indism4)},  (indism4 > 0)*(indism4 < 6))

#===================

loglike = bioMultSum(F)

#=================

import biogeme.messaging as msg
logger = msg.bioMessage()
#logger.setSilent()
#logger.setWarning()
logger.setGeneral()
#logger.setDetailed()

#====================

biogeme  = bio.BIOGEME(database,loglike)
biogeme.modelName = "10LatentRegression"

#=====================

results = biogeme.estimate(algorithm=opt.newtonTrustRegionForBiogeme)

#=====================


print(f"Estimated betas: {len(results.data.betaValues)}")
print(f"final log likelihood: {results.data.logLike:.3f}")
print(f"Output file: {results.data.htmlFileName}")
results.writeLaTeX()
print(f"LaTeX file: {results.data.latexFileName}")

#======================

