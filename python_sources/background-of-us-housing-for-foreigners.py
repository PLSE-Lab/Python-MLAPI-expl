#!/usr/bin/env python
# coding: utf-8

# ## Understand Background
# 
# I'm a Taiwanese and very unfamiliar with US real estates. I believe many data scientists all around the world would run into the same problem as I did. Without knowing the terminology of real estates, it is impossible for us to build any usable model for house price prediction.
# 
# ### Land Lot
# 
# After reading wiki page ["Land Lot"](https://en.wikipedia.org/wiki/Land_lot),   we will have basic idea about **Lot**:
# 
# 1. Lot is a basic unit of owned real estate. (column: *LotArea*)
# 2. Lot's shape are various. (column: *LotShape* )
# 3. The feet line that connected to street or road is called **Frontage**. (column: *LotFrontage*)
# 4. The terrian is a crucial characteristic of a lot. (columns: *LandSlope*, *LandContour*, *LandSlope*)
# 5. The area of the lot, however, is measured as if the land is flat.
# 
# Some advanced description about Lot (column: *LotConfig*) are:
# 1. Inside Lot: Only 1 frontage. see [Reference](https://www.allbusiness.com/barrons_dictionary/dictionary-inside-lot-4952919-1.html)
# 2. Corner Lot: A lot is located at the intersection of two roads. see [Reference](https://www.allbusiness.com/barrons_dictionary/dictionary-inside-lot-4952919-1.html)
# 3. Cul-de-sac: At the end of road, also called dead end. see [Wiki](https://en.wikipedia.org/wiki/Dead_end_(street))
# 4. FR2, FR3 means a lot has 2 or 3 frontages.
# 
# ### House
# 
# After studying what Lot is, it's time to study the thing grown upon on it.  
# 
# #### Dwelling Types
# 
# Dwelling Types means the original purpose when the house is built for. For instance, for a family or multiple families.
# 
# [**Single-Famaily detached**](https://en.wikipedia.org/wiki/Single-family_detached_home) is a free and standalone building. In legal term, it is designed and used for a single unit of dwelling. Its opposite is [Multi-family Residential](https://en.wikipedia.org/wiki/Multi-family_residential), an apartment is one of this kind.
# 
# **Two-Family Conversion** is a subset of Multi-family Residential building which serves 2 unit of dwelling but it is transformed from Single-Family detached.
# 
# [**Duplex**](https://en.wikipedia.org/wiki/Duplex_(building)) is a Two familiy residential building either there are stacked or attached together.
# 
# [**Townhouse**](https://en.wikipedia.org/wiki/Townhouse) is a housing style usually located in cities. The buildings array along streets and each building shares common walls with others attached to it.
# 
# ![Townhouse example](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Gda%C5%84sk_kamienice_przy_D%C5%82ugim_Targu.jpg/375px-Gda%C5%84sk_kamienice_przy_D%C5%82ugim_Targu.jpg)
# 
# There are two types of townhouse, one is called **inside unit** and the other is named **end unit**. Inside unit is located between other townhouses and end unit, on the other hand,  is locateed at the corner on intersections or the end of streets and shares only one common wall with another townhouse next to it.
# 
# 
# #### House Styles
# 
# House Styles defines what kinds of building architectures the house is. 
# 
# The word Storey or Story (American English) means a floor can be used common functionalities. (ex: Living, working, storage, etc.) Thus, 1 Story means a 1 floor house and 2 Story means 2 floor house.  
# 
# There are storeys number as 1.5, 2.5 representing that the top floor is less than full height to ceiling. 
# 
# Split-level is a style of house with staggered floors inside. The staggered floor splits the original floor into two or more storeys and generates more usable space on the orginal floor. The common split styles are Sitesplit and Backsplit.
# 
# ![Split Level  Example 1](https://upload.wikimedia.org/wikipedia/en/thumb/8/88/Traditional_Side_Split_Level_Home.jpg/330px-Traditional_Side_Split_Level_Home.jpg)
# <p style="text-align: center;">A Sitesplit example. </p>
# 
# ![Split Level Example 2](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Backsplit_splitlevel_house_in_Southeast_PA.JPG/330px-Backsplit_splitlevel_house_in_Southeast_PA.JPG)
# <p style="text-align: center;">A Backsplit example.</p>
# 
# A Split foyer is a house style with a floor in the foyer that leads to the main hall of the house (or the garage).  
# 
# ![Split Foyer Example](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQopfkGA0VQ-a0P0mEzZK_PWrnsWwaLgZAmuLruHBKOEhYVjpHfAQ)
# <p style="text-align: center;">A Split Foyer Example (outside)</p>
# 
# ![Spilt Foyer Example 2](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRt0mbb2D2JrlP7mgbcjKO7R_5hn50kni98BsCHarmrdOfehqB6)
# <p style="text-align: center;">A Split Foyer Example (Inside)</p>
# 
# #### Roof - Styles
# 
# [Flat Roof](https://en.wikipedia.org/wiki/Flat_roof) is one of the most common types of roof. Strictly to define, the angle between horizontal surface and its pitch is apporixmate less than 10 degree. Flat is suitable in desert climate and can provide more living space on top of building for residents. 
# 
# ![Flat Roof Example](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Ashdod_2005%2C_rooftop_view_p2.JPG/330px-Ashdod_2005%2C_rooftop_view_p2.JPG)
# <p style="text-align: center;">A Flat Roof Example</p>
# 
# [Gable Roof](https://en.wikipedia.org/wiki/Gable_roof) is one of the most common types of roof shape in those part of world with cold temparate. Gable Roof is consisted of two opposite slopes connected and the highest feet line becomes the roof ridge.
# 
# ![Gable Roof Example](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Hopfen_Kirche.JPG/330px-Hopfen_Kirche.JPG)
# <p style="text-align: center;">A building with Gable Roof</p>
# 
# [Gambrel Roof](https://en.wikipedia.org/wiki/Gambrel) is an advanced Gable roof. It has two slopes on each side and the shape of roof is symmetric.
# 
# ![Gable Roof Example](https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/LightningVolt_Barn.jpg/180px-LightningVolt_Barn.jpg)
# <p style="text-align: center;">A Gambrel Roof (Barn)</p>
# 
# [Hip Roof](https://en.wikipedia.org/wiki/Hip_roof) has four sides slope downward to the walls. Unlike gable roof which has a vertial wall reaching to a pitch,  all walls are attached to four sides directly. A hip floor might have a ridge or a pitch only. The latter one is named **pyramid roof**.  Hip roof is much hard to construct than gable roof and it has better resistance on strong winds because it has no large surface that catch winds.
# 
# ![Hip Roof Example](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Houses_at_Khammam_city_03.JPG/375px-Houses_at_Khammam_city_03.JPG)
# <p style="text-align: center;">A Hip Roof Example</p>
# 
# [Mansard Roof](https://en.wikipedia.org/wiki/Membrane_roofing) are mixed with grambrel roofs and hip roofs. It is four-sided and each side is designed according to gambrel's roof style. The sloping sides are not attached each other at a fleet line or a pitch. Insteand, they are connected with vertical surface on top. The lower slopes are steeper than the upper ones and are used to placed dormer windows. The steep roofs also create more usable spaces. 
# 
# ![Mansard Roof Exampe](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Wik_22_Szczecin_Pa%C5%82ac_Sejmu_Stan%C3%B3w_Pomorskich.jpg/330px-Wik_22_Szczecin_Pa%C5%82ac_Sejmu_Stan%C3%B3w_Pomorskich.jpg) <p style="text-align: center;">A Mansard Roof Example</p>
# 
# [Shed Roof](https://en.wikipedia.org/wiki/Mono-pitched_roof) has a significant appearance that each slop does not attach to another. 
# 
# ![Shed Roof](https://upload.wikimedia.org/wikipedia/en/thumb/e/e2/Modern-skillion-roofs.jpg/330px-Modern-skillion-roofs.jpg)<p style="text-align: center;">A Shed Roof Example</p>
# 
# #### Roof - Materials (TODO)
# 
# 
# 
# [Shingle](https://en.wikipedia.org/wiki/Roof_shingle) is a roof is covered by elements which overlapping with each other. Like gable roof, it has two sloping surfaces and its ridge is covered with cap.
# 
# ![Shingle Roof](https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Zakopane-schronisko-2.jpg/465px-Zakopane-schronisko-2.jpg) <p style="text-align: center;">A Roof Shingle Example</p>
# 
# [Membrane Roof](https://en.wikipedia.org/wiki/Membrane_roofing) is a roof made of waterproof materials and is often used in commercial buildings. 
# 
# 
# 
# 

# In[ ]:


# TODO: Study Neighborhood.

# ### Neighborhood

# The Neighborhoods listed in dataset are in [Ames, Iowa](https://en.wikipedia.org/wiki/Ames,_Iowa).

# According to the information from website https://www.realtor.com/, we  
# 1. Bloomington Heights
# 2. 


# 

# 

# In[ ]:




