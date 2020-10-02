#!/usr/bin/env python
# coding: utf-8

# This notebook is a part of the solution for DSG: EIE competition. The solution splited into 4 parts. Here is the list of notebook in correct order. The part of solution you are currently reading is highlighted in bold.
# 
# [**1. Introduction to the solution of DSG: EIE**](https://www.kaggle.com/niyamatalmass/1-introduction-to-the-solution-of-dsg-eie)
# 
# [2. Sub-region and State wise E.F and Evaluation](https://www.kaggle.com/niyamatalmass/2-sub-region-and-state-wise-e-f-and-evaluation)
# 
# [3. Individual Power plant E.F and Evaluation](https://www.kaggle.com/niyamatalmass/3-individual-power-plant-e-f-and-evaluation)
# 
# [4. Final thoughts, recommendation](https://www.kaggle.com/niyamatalmass/4-final-thoughts-recommendation)
# ***
# <br/>

# <h1 align="center"><font color="#5831bc" face="Comic Sans MS">Introduction to the solution of DSG: Environmental Insights Explorer</font></h1> 

# # <font color="#5831bc" face="Comic Sans MS">Notebook Overview</font> 
# This notebook explains the theory behind the solutions of Data science for good: EIE. This competition is organized by The Environmental Insights Explorer team at Google. They are keen to gather insights on ways to improve calculations of global emissions factors for sub-national regions. The goal of this competition is to show that it is possible to calculate the emission factor using remote sensing data ( satellite ). 
# 
# After researching and experimenting deeply, I have found a very promising and significant methodology to calculate the emission factor from satellite data using Google Earth Engine. This notebook demonstrates the theories and methodologies of my solutions. It will give an overall view of my methodology. The notebook is organized in different sections for readability. I highly suggest reading this notebook first, before proceeding to the actual implementation of my methodology because it will help to understand how the implementation is made and overall view of my method.

# # <font color="#5831bc" face="Comic Sans MS" size=40>Methodology Overview</font>
# 
# **Goal**: *Develop a methodology to calculate an average historical emissions factor of electricity generated for a sub-national region, using remote sensing data and techniques*. From the official documentation of EPA, we have found the formula for our emission factor. It is ```E.F = Total Emission/total electricity generation```. From this, we can easily say that we have to only calculate total emission from remote sensing data and total electricity generation is available in our local dataset. So we have to calculate total emissions from remote ( satellite ) sensing data.
# 
# 
# **Resources**: We reduce our problem that we have to calculate the total emission amount in satellite data and divide that by electricity generation, we will get the emission factor. To do all our research, we like to use Sentinel-5p no2 offline satellite images stored in GEE. This new version of satellite images captures high-resolution images of tropospheric no2, which is very useful for our research. Secondly, we will use two EPA emission dataset for getting total electricity generation and evaluation of our model. One is yearly emission data 2018 and new power plant wise emission data 2019. 
# 
# 
# **Challenges**: We have a lot of challenges we have to solve for calculating total emissions from satellite images. There are a lot of factors that affect no2 in the atmosphere. 
# 
#     Removing noises: There are a lot of factors that go into for no2 in the atmosphere. For example, one of the major sources of no2 is transportation, power plant and also forest burning and thousands of other factors. But we need no2 generated from the power plant. Sentinel-5p satellite captures all no2 in the atmosphere. So our one of the hardest challenges is to remove noises.
#     
#     Weather on no2: Weather has a very diverse effect on no2 on atmosphere and power plant. Heavy wind tends to move no2 from its sources very quickly and it removes spatial patterns. In high humidity, power plants tend to reduce no2 emission. There are a lot of other weather factors that affect no2 in the atmosphere.
#     
#     Evaluation: Another important challenge is our model evaluation. If we can't test our methodology performance then we will very uncertain about the model performance. So we have to evaluate our model. But there is a lot of data ambiguity for comparing satellite images data to ground-based data. 
#     
# We have to build a method for calculating atmosphere no2 from power generation in a way that overcomes these challenges. 
# 

# # <font color="#5831bc" face="Comic Sans MS">Methodology</font>
# We will explain our methodology by giving an example. For better understanding, we start with a single sub-region, Peurto Rico. Our goal is to calculate yearly total no2 emission from the power plant in Peurto Rico. After a lot of research, we have found an analytic method that gives very good results. 
# 
# * **Select all power plant and its neighbouring area as AOI**: Get points of every power plants in Peurto Rico. Draw a circle around that point. Because the power plant has a very diverse area requirement, we calculate the circle using relative radius. The radius of the circle is calculated using power plant total output in GWH and a multiplication factor. This way not all power plant circle will be the same. The large power plant will have a large circle and power plant that generate more energy will get a bigger circle. 
# 
# * **Masking with AOI**: After calculating our AOIs, we will mask with Sentinel-5p no2 data to exclude pixels that don't fall in our AOIs. This way we have already reduced most of the noises from other sources. But not all? There are road, industry and weather inside each circle. So we have to remove the noise inside the circles. 
# 
# * **Apply weighted reductions**: After making a yearly composite of the Peurto Rico no2 images using ee.Reducer.sum(), we will now reduce (sum) all the circles(AOIs) of Peurto Rico to get yearly total no2. But we will not do a general reduction. We will use a weighted reduction. These weights are calculated for each power plant. The weight is applied in a way that pixels away from power plant centroid will get less weight and pixels near the centroid of the circle will get higher weight. For example, let's see a hypothetical example. Below are six points, as we move from each point we get less and less weight that's why the near point is whiter compared to others. *This photo doesn't mean power plant and white mean more weights.*
#   
#   ![Just example](https://i.imgur.com/Wv92X0n.png)  
# 
# 
#    By using these weights, we have able to reduces a lot of different noises in our calculation. For example, wind moves no2 very quickly and heavy wind can introduce new no2 from neighbouring sources. And transportation, industry and other sources inside the circles are generally situated far from centroid because centroid is our power plant. So as we move from the centroid,  pixels will get less weight. This way we have removes our most noises from calculations. 

# # <font color="#5831bc" face="Comic Sans MS">Conclusion</font> 
# We just discussed the overall idea behind my methodology. It gives an overall idea of my solutions. But the solutions are divided into 4 parts. Each part contains detailed information on each problem and its solutions. It also gives the necessary recommendations. You are welcome for reading my notebooks. See you in the next part of the solutions!
