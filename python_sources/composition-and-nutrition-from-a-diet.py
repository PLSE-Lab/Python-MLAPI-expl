#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nutrition_code as nut # nutrition module

SOURCE_DIR = '../input/nutrition/'
nut.load(source=SOURCE_DIR)


# This kernel is complimentary to a series of articles from the Food Wall project :
# 
# 1. [nutritional requirements to feed Singapore](https://www.footprintzero.com.sg/post/nutritional-requirements-to-feed-singapore)
# 2. [nutrition content and chemical composition of national diets](https://www.footprintzero.com.sg/post/nutrition-content-and-chemical-composition-of-national-diets)
# 
# The [Food Wall](https://www.footprintzero.com.sg/food-wall) is a modular greenhouse design for high food self-sufficiency in a tropical city using Singapore as a case study.  The population diet determines both the output requirement for a self-sufficient agriculture system and is the basis for input specification for an organic waste digestion system based on urban sewage and food waste. 
# 
# To start, how much variation is in diets over time and by country?  By combining information from two datasets - FAO [food balances](http://www.fao.org/faostat/en/#data/FBS) and [USDA nutrition data](https://ndb.nal.usda.gov/ndb/) it is possible to explore and gain insights into these trends.  
# 
# The USDA food data is an inventory of food items and their nutritional content in terms of calories, macronutrients - fiber, fat, carbs, protein and micronutrients - Ca, K, P, etc.. and major Vitamins.     

# In[ ]:


usda = nut.tables['nutrition']
usda[[x for x in usda.columns if not x in ['nutrition id','crop group']]].head()


# FAO database includes food balances by country and by year.  A food balance is an estimation technique at a national level that attempts to quantify the average supply of food to individuals in the population.  FAO has published material describing the food balance estimation methodology and assumptions to consider.  The table in this dataset *nutrition* is the summary of the food balance dataset by food item code, by year and by country.  
# 
# These tables were created manually from the website without knowledge of the Kaggle dataset, and as a followup exercise it is possible to integrate with the Kaggle dataset tables.

# In[ ]:


foodbal = nut.tables['diets_global']
years = list(foodbal.year.unique())
countries = list(foodbal.country.unique())
foodbal.head()


# a quick inspection using a pivot report already reveals a pattern by country and by year only considering for the sum of the kg per person per annum

# In[ ]:


pd.pivot_table(foodbal,values='amount kg_pa',columns='year',index='country',aggfunc=np.sum)


# to explore deeper into the nutritional changes requires mapping the food item id between USDA and FAO. This was done manually in a table *food_item* by inspecting the food descriptions one-by-one

# In[ ]:


food_item = nut.tables['food_item']
food_item[[x for x in food_item.columns if not x == 'yield']].head()


# The custom Diet class operates on these three tables to present national food balances in terms of their nutritional content

# In[ ]:


china_modern = nut.Diet.from_country(2013,'China')
china_modern.nutrition()


# The main source of variation in the country-year dataset is in the energy portion of the food - refined grains (carbs) and animal food (fats), whereas the other food groups have less varaiation and in some cases reduction - an inverse to the trends in energy foods. 

# In[ ]:


diet_index = [(x,y) for x in years for y in countries]
diets = [nut.Diet.from_country(x[0],x[1]) for x in diet_index]
food_groups = [d.food_groups() for d in diets]
yr = pd.DataFrame({'year':[x[0] for x in diet_index]})
cntry =pd.DataFrame( {'country':[x[1] for x in diet_index]})
energy = pd.DataFrame({'energy':[fg['01 cereal refined']+fg['05 animal'] for fg in food_groups]})
nut_rich = pd.DataFrame({'nutrient rich':[sum(fg)-fg['01 cereal refined']-fg['05 animal'] 
                                  for fg in food_groups]})
nut_avg = np.mean(nut_rich)

fgrcds = pd.concat([yr,cntry,energy/nut_avg['nutrient rich'],nut_rich/nut_avg['nutrient rich']],axis=1)
fgrcds.set_index(['year','country'],inplace=True)
fgrcds.loc[(1965,)]


# Define a normalizing constant of the average level of nutrition rich food groups [wholegrain, vegetables, fruits] *nut_avg* and divide the energy and nutrition food groups by this normalizing constant to see the level of each of these food groups on a relative basis.  
# 
# From the data, in 1965 there are clear differences between countries - with US, France and Italy consuming higher kg of food in general, and in energy based food groups (cereal, animal) compared to India and China, and Thailand and Japan falling inbetween these two groups. 
# 
# By applying a similar approach using nutrition groups [fats, protein, carbs, fiber] and grouping fat_pr = [fats,proteins] and carb_fib = [carbs,fiber] the same pattern appears of a general higher consumption, and particularly higher consumption rates for fat and protein which occur in higher concentrations in animal based food groups. 

# In[ ]:


nut_basis = [d.nutrition() for d in diets]
fat_pr = pd.DataFrame({'fat_pr':[nt['fats']+nt['protein'] for nt in nut_basis]})
carb_fib = pd.DataFrame({'carb_fib':[nt['carbs']+nt['fiber'] for nt in nut_basis]})

cf_avg = np.mean(carb_fib)

ntrcds = pd.concat([yr,cntry,fat_pr/cf_avg.carb_fib,carb_fib/cf_avg.carb_fib],axis=1)
ntrcds.set_index(['year','country'],inplace=True)
ntrcds.loc[(1965,)]


# In[ ]:


fgrcds.loc[(2013,)]/fgrcds.loc[(1965,)]-1


# In[ ]:


ntrcds.loc[(2013,)]/ntrcds.loc[(1965,)]-1


# Across these countries there is an observed increase in fat-proteins in range +20 to + 60% and China with an increase of +170% from 1965 to 2013.  This increase is most pronounced when compared to the moderate changes in the carb-fiber group -17 to +20%.  Similar to the cross-section inspection for 1965, the change from 1965 to 2013 shows increase in refined cereals and animal food groups at a faster rate compared to nutrient rich groups.

# For the design basis diet for the Food Wall greenhouse, a weighted blend of these four countries is selected to model the Singapore diet.
# 
# Population level food balance for Singapore is available by national reporting using similar but different categories of food groups from USDA and FAO.  The [Singapore National Nutrition Survey](https://www.hpb.gov.sg/workplace/workplace-programmes/useful-information-for-organisations/national-reports-and-surveys) presents a detailed analysis of national food balance and insights into nutritional trends by demographic and age.  As a first pass, the high level food groups are mapped and the Singapore diet is estimated as a composite hybrid blend of four diets from the FAO country database - China, India, Thailand and Japan.  The selection of weights and countries for the blend is based on a best match to high-level food groups.  This match is imperfect since Singapore's reported high level of refined grain of 255 kg/yr and low level of animal 175 kg/yr cannot be match by any combination of countries, so a compromise is made in the weight selection.
# 
# It is possible to perform a similar food item mapping from USDA to this report at a food item detail level and this is left as a follow-up exercise.  

# In[ ]:


sgp = nut.sgp_diet()
sgp.food_groups()


# In[ ]:


sgp.nutrition()


# For the design of a compost system, the chemical and elemental composition information is needed, specifically the C:N ratio which is an operational parameter in both anaerobic and aerobic digestion systems.  The elemental composition is estimated for a diet using some general hueristics of the average chemical properties of plant and animal cells.  
# 
# The input to a digester is the resulting diet blend after subtracting the portion consumed by the humans.  For food waste only system, the human anerobic digestion is not considered, whereas for a food waste + sewage system, the human digestion is also modeled as a process operation on the source diet.  The consideration of these process steps are left for a follow-up analysis and the analysis in this section is estimation of properties for the source diet that is fed to a population. 

# In[ ]:


cho_rcds = nut.tables['cho']
cho_pvt = pd.pivot_table(cho_rcds,columns='nutrition group',index='element',values='wt')
cho_pvt.loc[['C','H','O','N','S']][['carbs','mono unsaturated','polyunsaturated',
        'sat fats','protein','fiber']]


# **Carbohydrates**
# 
# Carbohydrates (carbs) primary function is energy storage and are composed exclusively of CHO and derived from base sugar molecules isomers of glucose C6H12O6.  They also have a secondary function as building block for more complex proteins and tissue.  Carbs fall into two main groups - simple sugars - (glucose, sucrose, fructose, maltose) and complex carbohydrates - (amylose, amylopectin, glycogen) which are long chain combinations of simple sugars connected by glycosidic linkages and molecular formula C6H10O5.  The ratio of simple sugars to complex carbohydrates is source-dependent.  For this first pass analysis, a simple hueristic is applied of 5/95 blend of simple sugar/polysaccharide for the "carb" nutritional category.  This results in a blend of C(44%) H(6%) and O(50%).
# 
# **Fats**
# 
# Fats are triglycerides - triplet structures of fatty acids.  Fats and fatty acids - also known as lipids - have a similar function as carbohydrates in energy storage, and also a functional role as an organic barrier in cell membranes.  For humans and animals one example of fatty acids non-energy function is in the blood brain barrier.  Fats can be synthesized by animals so dietary fat is not a strict requirement, however many diet studies have found a moderate level of unsaturated fats to be beneficial and less obvious is the nutritional benefit of saturated fats which are a risk to cardiovascular health.  Unsaturated fatty acids are characterized by their carbon number and the degree of saturation.  For example C18:1 - oleic acid - has 18 carbon atoms and one unsaturated bond.  Similar to amino acids, each food source has a specific compositional mix of fatty acids.  The structure of animal based fatty acids is diffrent from those from plant origin.  On average, the most frequently occuring plant based unsaturated fatty acids have chain length of 18 and 1-3 unsaturated bonds.  Saturated fatty acids from plant origin most common carbon number is 16 - palmitic and 18 - stearic aicd.  The CHO content of fats is estimated from these representative molecules with an average formula of C(77%) H(12%) O(11%) mostly the same for saturated and unsaturated as the 2 hydrogen atom difference is small for such large molecules.
# 
# **Proteins**
# 
# Proteins are composed of amino acids and contain mostly CHO with a significant level of N,S.  Proteins often also include trace inorganics as catalysts such as Fe in Hemoglobin in animals to carry oxygen in the blood and Mg in RubisCO for photosynthesis in chloroplasts of plant cells.  Nitrogen content is based on a range of amine R-NH2, R-NH, R-NH3+ functional groups from the base amino acids and varies with the type of protein in the range 13-19%, with a typical value around 16%.  Sulfur is also based on a range of functional groups and content similarly varies in a range of 0.9 - 1.7% and a typical value is selected at 1.1%.  The CHO content of proteins is estimated as the same as carbohydrates with an overall composition of C(47%) H(7%) O(29%) N(16%) S(1.1%).
# 
# **Fiber**
# 
# Of the four main groups - fibers are the most complex and difficult to characterize.  Fibers origin mostly from plant cell walls and are large complex macromolecules with a carbohydrated based polymer macrostructure with attached diverse range of functional groups unique to the particular tissue. Two broad classes of fibers are - carb-based 60% C6H12O6 - (Hemicellulose, cellulose) and complex 40% C31H34O11 - (Pectin, Lignin).  Complex fibers not only contain CHONS from sugar and amino acid monomer base, but also contain significant quantities of ash - inorganic metals important for plant nutrition - so that complete digestion of fibers in an aerobic compost process is important for release of micronutrient inorganic minerals.  A simplifying assumption for this exercise which may be technically incorrect is that fibers only contain CHO.  The justification for this assumption is based on a reference source that claimed that reporting of protein in food is derived from total nitrogen measurement, and therefore may include NS bound in the fiber so that there is no need to separately account for fiber NS and protein NS.  By applying a simple hueristic of 60/40 carb-based/complex the resulting composition is C(49%) H(7%) O(44%)  
# 
# **Inorganic minerals**
# 
# Plant nutrition is based on soluble ions - macronutrients (N, P, K, S, Mg, Ca) and micronutrients (Fe, Mn, B, Cu, Mo, Zn, Co).  Organic based ions (N, S, P) must be oxidized to their anion form NO2-, NO3-, PO4(3-), SO4(2-) to be available.  Nitrogen can form as an anion in oxidized state or a cation Ammonium NH4+ in a reduced state. Other anions not directly involved in plant nutrition are free Cl- and Si(4-).  The organic N, S of a food based organic waste stream is estimated from protein content.  Ca from animal source can either be in free ions in solution, or bound in structural organs - bones, shell as CaCO3.  For CaCO3 the material must first be acid digested to Ca2+.  For plant based sources Ca2+ and K+ are already in the form of free ions in solution.  The USDA database also contains direct estimates of P, K, Fe levels for range of food groups, which coud be added for a more accurate elemental analysis.  This first pass analysis looks at Ca and K using the direct USDA lookup, and applies a placeholder estimate for P, Mg levels with an arbitrary function of K = P and Mg = 0.5Ca + 0.5K.         

# In[ ]:


sgp.elements()


# Finally using the elemental CHONS breakdown, the C:N ratio can be determined for a given diet.

# In[ ]:


sgp.cn_ratio()

