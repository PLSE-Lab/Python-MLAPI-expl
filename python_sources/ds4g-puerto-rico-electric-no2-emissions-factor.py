#!/usr/bin/env python
# coding: utf-8

# # Summary
# <p>  I get an NO2 emissions factor of 7.4 kt/GWh for Puerto Rico for the data supplied to us.  This assumes an NO2 lifetime
# (tau) of 6 hours, independent of wind speed.  There are admittedly a lot of loose ends
# with my analysis, but that's my story and I'm sticking to it.
# </p>
# <p>
# The code I used to get this is here:
# </p>
# <p>
#     https://www.kaggle.com/vgates/ds4g-emissions-factor-wind-rotation-code
#  </p>
#  <p>
#     It runs fine on my PC but errors out on Kaggle and I'm out of time to figure out why.  I'd be happy to work on that if the organizers have any interest.
#     </p>
# <p>
#      This challenge has some challenges: The satellite photos don't directly give NO2 _emissions_; they're snapshots of NO2 *concentration* at a moment in time.  Power plants are not the only thing emitting NO2; in fact power plants are often a fairly small contribution (but this may not be true in Puerto Rico, where per capita energy consumption is much lower than the continental US).  The spatial resolution of the NO2 data is a big improvement on previous instruments, but the pixels are still too large to separate sources which are close together if there is only a single picture.  Because the pixels shift on successful flybys, spatial resolution can be increased by pixel averaging.
# </p>
# <p>
#       Puerto Rico sits in trade winds which are relatively constant in direction and magnitude, and (apart from hurricanes), the wind speeds are generally fairly small.  The relative constancy of wind direction means that one can probably get a pretty good picture of a plume without rotating pictures with wind direction.  On the other hand, if one source is upwind of another, their plumes may overlap.
# </p>
# <p>
#        For a model to generalize to other locations, it must be able to deal with plumes which may go every which way, instead of mostly East to West as in PR.  Inspired by these 2 papers:
# </p>
# <p>
# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/grl.50267
# </p>
# <p>
# https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2015GL063148
# </p>
# <p>
# I decided to try rotating the satellite pictures, about each power plant location, to put the wind direction right-to-left for all pictures, and summing them.  Having all the plumes lined up should better quantify the concentration in the plume.
# </p>
# <p>
#    I did this for only a subset of the GPPD list provided; I chose just the 9 plants with the highest capacity.
#    In the combined rotated pictures, emissions from other sources form circles.  Here is a combined picture where the
# center of rotation is the Palo Seco plant.  It may not be obvious at the magnification shown here, but there at least
# 2 other circles visible; one smallish one near Palo Seco (you can locate Palo Seco because it's the center of the circle),which may be the San Juan CC plant.  There's a much larger circle which is almost tangent to the bottom of the picture; this may be one of the southern coast power plants, possiblly Aguirre.
# 
# 
# 

# In[ ]:


from IPython.display import display, Image
display(Image(filename='../input/sumpicpalosecooiljpg/sumPicPaloSecoOil.jpg'))


# <p>
# Here's a photo with Aguirre as the center of rotation; there does appear to be a plume near that location (you may need to view the photo larger than it will show up in a kernel).
#    As you can see, NO2 concentrations can be inflated by the circular smearing out of other plumes onto the power plant in question.
#  <p>
#  </p>
#    I tried fitting an exponential to the tail of a region near each power plant, to get the NO2 lifetime in the atmosphere.  This is known to vary with wind speed, but the wind speeds are in too narrow a range to divide particularly finely.  The values are all over the map, but 4 of the 9 plants give about 4 hours.  The NO2 lifetime is needed to convert concentrations to emissions.  For the number at the top, I used the value of 6 hours from the Fioletov paper above.

# In[ ]:


from IPython.display import display, Image
display(Image(filename='../input/sumpicaguirrejpg/sumPicAguirre.jpg'))


# 
# # Puerto Rico has some features which may not to apply to other regions:
# 
# * As noted, it sits in trade winds which are relatively constant in direction and magnitude, and (apart from hurricanes), the wind
# speeds are generally fairly small.  The relative constancy of wind direction means that one can probably get a pretty
# good picture of a plume without rotating pictures with wind direction.  On the other hand, if one source is upwind of
# another, their plumes may overlap.
# * It's an island.  Upwind of it, there are only a few small islands and lot of open ocean, unlike a landlocked subnational unit which may 
#   have power plants and population centers upwind.
# * It doesn't seem to have a lot of fossil-fuel-intensive heavy industry, which would produce its own NO2 plumes.  There
#   are a couple of cement plants, but that's all I've managed to find.  There is no on-island oil refining.
# * A number of its power plants are not near its largest urban centers
# * It gets a much larger percentage of its electricity from oil than the rest of the US (65% vs 0.3%)
# * Many of its biggest plants burn heavy No. 6 fuel oil, which is often high in sulphur. This may lead to different
# emissions than other oil plants.  In particular, it can give high SO2 emissions:
# https://en.wikipedia.org/wiki/Heavy_fuel_oil    
# https://slate.com/business/2014/05/puerto-rico-is-burning-oil-to-generate-electricity-its-completely-insane.html 
# * Its residents use one third as much energy than residents of the 50 U.S. States, so the NO2 emissions of urban centers
#   may be smaller relative to power generation.
# https://www.eia.gov/state/analysis.php?sid=RQ
# * It has mountains in the middle of the island, which may distort wind flow and hence emissions plumes
# * At least one of its plants (the AES coal plant) have NOx mitigation equipment installed.    

# #Loose Ends
# Loose Ends:
# * My code uses the U,V components of wind as-is.  I haven't tried warping the wind vector to the new coordinate system.  My guess is it's a small effect.
# * I haven't examined the effect of rainfall
# * I haven't binned the data by wind speed. PR wind speeds are mostly pretty low; not clear if binning would help.  There is some evidence that the plumes are longer with more wind in this very interesting kernel:
# 
# https://www.kaggle.com/jyesawtellrickson/visualising-wind-effects
# 
# * My code does not attempt to subtract a background, but this could be done using the space around the plume.  This is probably important, not just because of true ambient NO2 from non-power-plant sources, but also for the smear left by big emitters as the pictures are rotated.
# 
# * In the absense of background subtraction, one could attempt to subtract a contribution from motor vehicles, based on, say,Puerto Rico gasoline consumption data (which does seem to exist on the internet).
# 
# * There is some evidence that the total NO2 in a file goes down as the cloud_fraction goes up.  This could be corrected for:
# 

# Here's the output from the model which I used, along with the total electrical consumption of 16121 GWh, to calculate the emissions factor.

# In[ ]:





# In[ ]:


import pandas as pd
model_output = pd.read_csv('../input/model-outputcsv/model_output.csv')
model_output


# In[ ]:


from IPython.display import display, Image
display(Image(filename='../input/cloudfractionregplotpng/cloudfractionregplot.png'))


# <p>(and here's an illustration of what I mean about the slow, steady-direction winds)</p>

# In[ ]:


from IPython.display import display, Image
display(Image(filename='../input/windhistpng/windhist.png'))

