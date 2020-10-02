#!/usr/bin/env python
# coding: utf-8

# This notebook is a part of the solution for DSG: EIE competition. The solution splited into 4 parts. Here is the list of notebook in correct order. The part of solution you are currently reading is highlighted in bold.
# 
# [1. Introduction to the solution of DSG: EIE](https://www.kaggle.com/niyamatalmass/1-introduction-to-the-solution-of-dsg-eie)
# 
# [2. Sub-region and State wise E.F and Evaluation](https://www.kaggle.com/niyamatalmass/2-sub-region-and-state-wise-e-f-and-evaluation)
# 
# [3. Individual Power plant E.F and Evaluation](https://www.kaggle.com/niyamatalmass/3-individual-power-plant-e-f-and-evaluation)
# 
# [**4. Final thoughts, recommendation**](https://www.kaggle.com/niyamatalmass/4-final-thoughts-recommendation)
# ***
# <br/>

# <h1 align="center"><font color="#5831bc" face="Comic Sans MS">Final thoughts, recommendations</font></h1> 

# # Performance of the methodology
# 
# * Because the methodology uses only satellite imagery for calculating emission, this will make the calculation more frequently. 
# * The method and implementation fully based on google earth engine platform, which makes the solutions fully fast, scalable.
# * Very good at reducing noise like emission from other sources. 
# * Because of weighted reductions, this methodology is very effective reducing weather effect like wind speed, participation etc. 
# * Easily applicable to any geospatial area. We have already calculated the emission factor for each state and power plants. 
# * Easily applicable to any time frame. In our notebook, we calculated yearly, ozone season emission factor. But it is really changing the dates in calculation function and calculate the emission factor for any time series. 
# * Useful for marginal emission factor. We already calculated emission factor for individual power plant. With just a few tweaks we can calculate the marginal emission factor. 

# # Recommendation
# * This methodology for calculating emission factor works great. But still, we can improve the model. 
# * Using cumulative cost mapping we can reduce noise even further. 
# * Combining this method with the ground-based model can build a hybrid model which will more robust. 
# * If we can apply machine learning, this will make emission factor much much more accurate. 

# # Conclusion
# Congratulations! Thank you. We have finally come to an end. Thanks for reading. I hope that this method helps the EIE team to calculate the emission factor more robust and accurate. Also, I hope we together learn something new. Thanks! 
