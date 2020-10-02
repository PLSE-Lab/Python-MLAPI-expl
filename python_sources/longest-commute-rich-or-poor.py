'''
I'm very curious about the relationship between income and lifestyle choices such as daily
commute time to work. This is my first stab at exploring it, comments and suggestions welcome!
'''

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

########### read in the data #################################################
# read data into a dataframe
population_a = '../input/pums//ss13pusa.csv'
population_b = '../input/pums//ss13pusb.csv'
popa = pd.read_csv(population_a,usecols=['SERIALNO','JWMNP','JWTR','PINCP'])
popb = pd.read_csv(population_b,usecols=['SERIALNO','JWMNP','JWTR','PINCP'])
pop = pd.DataFrame(pd.concat([popa,popb],axis = 0))
pop = pop.dropna()

# subsetting cars, since an overwhelming majority of Americans (90%) drive to work
pop_car = pop[pop['JWTR'] == 1]
pop_noncar = pop[pop['JWTR'] != 1]

########### set the color scheme #############################################
# add the Tableu 20 scheme to create visually pleasant graphs  
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

# scale the RGB values to a format matplotlib accepts  
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)    

# set the default color cycle to tableau20
mpl.rcParams['axes.color_cycle'] = tableau20


########### set plot area parameters #########################################
# set plot size
plt.figure(figsize=(12, 18))    

# add a 2x1 grid, first subplot for car commuters
ax = plt.subplot(211)    

# remove the plot frame lines to bring clarity  
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
   
# set data range limit   
plt.ylim(0, 180)    
plt.xlim(0, 1400000) 

# add vertical tick lines to guide the reader's eye
for x in range(0, 1400000, 200000):    
    plt.plot([x] * len(range(0, 180)), range(0, 180), "--", lw=0.5, color="black", alpha=0.3) 
 
# add title and footnote
plt.text(650000, 190, "Daily commute time to work for car drivers, 2013 US census data", 
         fontsize=17, ha="center") 

# set labels
ax.set_xlabel('Annual income(2013 $)')
ax.set_ylabel('Travel time to work')

# plotting the actual data
plt.scatter(pop_car['PINCP'],pop_car['JWMNP'],c=tableau20[0], edgecolors='none')


# add the second subplot on the 2x1 grid, for non-car commuters
bx = plt.subplot(212)    

# Remove the plot frame lines to bring clarity  
bx.spines["top"].set_visible(False)    
bx.spines["bottom"].set_visible(False)    
bx.spines["right"].set_visible(False)    
bx.spines["left"].set_visible(False)   

# set data range limit   
plt.ylim(0, 180)    
plt.xlim(0, 1400000) 

# add vertical tick lines to guide the reader's eye   
for x in range(0, 1400000, 200000):    
    plt.plot([x] * len(range(0, 180)), range(0, 180), "--", lw=0.5, color="black", alpha=0.3) 
 
# add title and footnote
plt.text(650000, 190, "Daily commute time to work for non-car drivers, 2013 US census data", 
         fontsize=17, ha="center") 

# set labels
bx.set_xlabel('Annual income(2013 $)')
bx.set_ylabel('Travel time to work')

# plotting the actual data
plt.scatter(pop_noncar['PINCP'],pop_noncar['JWMNP'],c=pop_noncar['JWTR'].astype(np.int), edgecolors='none')

# save the plots
plt.savefig('Income_v_CommuteTime',dpi=200)