#!/usr/bin/env python
# coding: utf-8

# # Key objectives
# 
# The objectives are given to us in the problem statement, and it transpires that the main challenges are not primarily in the prediction or detection space, but rather in the <i>cleaning and automation space</i> space:<br>
# 
# <blockquote>Our biggest challenge is <b>automating</b> the combination of police data, census-level data, and other socioeconomic factors. Shapefiles are unusual and messy - which makes it difficult to, for instance, generate maps of police behavior with precinct boundary layers mixed with census layers. Police incident data are also very difficult to <b>normalize and standardize</b> across departments since there are no federal standards for data collection.<br>
# 
# <b>Performance</b> - How well does the solution combine shapefiles and census data? <b>How much manual effort is needed?</b><br>
# 
# <b>Accuracy</b> - Does the solution provide reliable and <b>accurate analysis</b>? How well does it match census-level demographics to police deployment areas?<br>
# 
# <b>Approachability</b> - The best solutions should use best coding practices and have useful comments.<br></blockquote>
# 
# There are a multiplicity of issues in this messy data! So I decided to prioritize two of the main barriers to getting going:<br>
# 
# <blockquote>
# <p style="color:MediumSeaGreen;"><b>
# 1. How do we even know what data we have, and what broad issues will we face with it?<br>
# 2. How do we standardize co-ordinate reference systems (CRS) across census-level data and police-level data?</b>
#     </p></blockquote>  
#    

# I'm going to skip the exploratory data analysis phase in this notebook as this has already been very well covered by (among others) [Chris Crawford](https://www.kaggle.com/crawford/another-world-famous-starter-kernel-by-chris") and [Darren Sholes](https://www.kaggle.com/dsholes/confused-start-here), and so I'm just going to dive into "solution mode"... Let's get going...!

# ## About projections and co-ordinate reference systems
# 
# If, like me, you are going "co-ordinate reference systems? huh?" I can highly recommend this quick 4-page intro which demystifies it all: http://www.ee.co.za/wp-content/uploads/legacy/Projections%20and%20coordinate.pdf

# ## Getting set up

# In[ ]:


# Import required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import geopandas as gpd
import geopy as gpy
from geopandas import GeoDataFrame
from shapely.geometry import Point
# Set Google Maps API key for use later on - this is required to run the find_crs() function below
# Register for free at https://cloud.google.com/maps-platform/
read_in = pd.read_csv("../input/api-key/key.csv")
api_key = read_in["Key"][0]


# ## External "helper" data sources
# 
# 1. <b>texas_tracts</b>: The American Community Survey (ACS) data is broken down into "census tracts" but the corresponding shapefiles are not supplied. Fortunately [Darren Sholes](https://www.kaggle.com/dsholes/confused-start-here) supplied us with a link: https://www.census.gov/geo/maps-data/data/cbf/cbf_tracts.html (this is one of many sets of shapefiles available on www.census.gov). You guys in the US don't realise how lucky you are to have such a rich, publicly available, nicely formatted set of census data!
# 2. <b>FIPS</b>: The census tracts use what are known as FIPS codes to break down the US into states and counties. These are available for download from https://www.census.gov/geo/reference/codes/cou.html and will allow us to read which state and county data is included in any given ACS file.
# 3. <b>usps_df</b>: It will also be useful to convert the US state abbreviations to full state names (especially if you are not from the US and are not already familiar with them!) - this information is available from https://en.wikipedia.org/wiki/List_of_U.S._state_abbreviations
# 4. <b>projection_df</b>: And then finally a list of coordinate reference systems / spatial reference systems was obtained from  http://www.spatialreference.org/. This is used to determine which one is "right" for any given set of data later on. My compiled list includes 823 "likely candidates" but do note that for "production purposes" it would need to be expanded as it's not exhaustive and has been compiled for purposes of demonstrating the proposed method.
# 
# You can download the "custom" files that I compiled for this purpose at https://github.com/shotleft/CPE-Challenge-Kaggle.

# In[ ]:


# Read in the Texas census tract file - a separate census tract zip file is required for each state
# In this example I'll be sticking to Texas!
texas_tracts = gpd.read_file("../input/census-tracts-texas-censusgov/cb_2017_48_tract_500k/cb_2017_48_tract_500k.shp")

# Read in FIPS County Codes for lookup reference
FIPS = pd.read_csv("../input/cpe-helper-data/FIPS_County_Codes.csv", dtype = "object")

# Read in the US Postal Service State Codes
usps_df = pd.read_csv("../input/cpe-helper-data/USPS_State_Codes.csv")

# Read in the coordinate reference systems compiled via spatial reference.
# Note, I selected 823 of the "most likely to be used" - this list could be expanded for thoroughness!
projection_df = pd.read_csv("../input/cpe-helper-data/CRS_References.csv")


# ## Unpacking the need for knowing what we have
# 
# Does this sound like rather a trivial problem to be investigating? I thought so too, until I unzipped the data and got a look at the folder structure! It took a long while just to drill down into all the folders, try to work out what files we have and don't have and why - and when I started reading the data into Python with a view to exploring further I just ran into more problems! We only have 6 counties for this challenge, which is still manageable, but imagine once we extend this project to all 52 states with all the counties each contains! Here is a quick glimpse into the folder structure:
# 
# ![](https://shotlefttodatascience.files.wordpress.com/2018/10/folder_img2.jpg)
# 
# Some of the key issues I uncovered were:<br>
# 
# - The departments are given as numbers? What do those numbers actually represent?
# - In some cases we are given a "use of force" incident file, and in others not which affects what we'll be able to analyse and plot, it would be nice to know without opening every folder to look!
# - After [some reading](https://en.wikipedia.org/wiki/Shapefile) it becomes apparent that the following shapefile types are <i>required</i> if we are to make sense of given shapefile data: "shp", "shx", "dbf", "prj" - but are they present in all cases?
# - After examining the ACS data provided I also discovered that in some cases the wrong region has been provided in a category, for example the race-sex-age data supplied for Travis County actually contains data for Hill County, eeish!
# 
# Fortunately, what IS consistent is the directory structure, directory naming conventions and placement of files within directories so we can use this to create a function to read and summarize "what we have" as a starting point...

# ### Assembling our starter data structures
# 
# Knowing that we have a consistent folder structure allows us to read in which folders and files we have.

# In[ ]:


# Read in initial list of departments available (the top level of our directory structure)
dept_list = [d for d in next(os.walk("../input/data-science-for-good/cpe-data"))[1] if not d.startswith((".", "_"))]

# Read in directories per department (the next level down)
dept_dir_dict = {}
for dept in dept_list:
    depts = [d for d in next(os.walk(os.path.join("../input/data-science-for-good/cpe-data", dept)))[1] if not d.startswith((".", "_"))]
    dept_dir_dict.update({dept:depts})
    
# Read in sub-directories per directory (one more level down)
dept_subdir_dict = {}
for dept in dept_dir_dict:
    for subdir in dept_dir_dict[dept]:
        subsubdir = [d for d in next(os.walk(os.path.join("../input/data-science-for-good/cpe-data", dept, subdir)))[1] if not d.startswith((".", "_"))]
        dept_subdir_dict.update({subdir:subsubdir})
        
# For each department we expect ACS data - get a list of ACS directories
sub_dir_acs = []
for i in range(len(dept_list)):
    sub = [s for s in dept_dir_dict[dept_list[i]] if "ACS" in s][0]
    sub_dir_acs.append(sub)
    
# Within each ACS directory we expect 5 sub-folders for each category of data - get a list of ACS sub-directories
sub_dir_acs_det = []
for i in range(len(sub_dir_acs)):
    subsub = [s for s in dept_subdir_dict[sub_dir_acs[i]]]
    sub_dir_acs_det.append(subsub)
for i in range(len(sub_dir_acs_det)):
    sub_dir_acs_det[i].sort()
    
# Create a dictionary that can be used to reference which type of ACS data we want to retrieve
acs_dict = {"education" : 0, "education25" : 1, 
           "housing" : 2, "poverty" : 3, "rsa" : 4}

# And create a dictionary to go back - from dictionary number to ACS descriptions
inv_acs_dict = {v: k for k, v in acs_dict.items()}
    
# For each department we expect shapefile data - get a list of shapefile directories
sub_dir_shp = []
for i in range(len(dept_list)):
    sub = [s for s in dept_dir_dict[dept_list[i]] if "hape" in s][0]
    sub_dir_shp.append(sub)


# ### Create functions to read in required data
# 
# Sooner or later we'll need to be reading in the actual data so that we can plot it and analyse it - so I've written some functions to do this for ease of use down the line... Each is documented via docstring below:

# In[ ]:


def read_uoffile(dept):
    """This function reads in UOF data for the requested department.
    Returns a Pandas dataframe after reading in the relevant .csv file."""
    path = os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)])
    file = [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", 
                                               dept_list[int(dept)])) if "UOF" in f][0]
    full_path = os.path.join(path, file)
    
    df = pd.read_csv(full_path)
    return df

def read_shapefile(dept):
    """This function reads in police shapefile data for the requested department.
    Returns a GeoPandas dataframe after reading in the relevant "shp", "shx", "dbf" and "prj" files."""
    path = os.path.join("../input/data-science-for-good/cpe-data", dept_list[int(dept)], 
                        sub_dir_shp[int(dept)])
    file = [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", 
                                               dept_list[int(dept)], 
                                               sub_dir_shp[int(dept)])) if ".shp" in f][0]
    full_path = os.path.join(path, file)
    
    gdf = gpd.read_file(full_path)
    return gdf

def read_acsfile_key(dept,category):
    """This function reads in ACS data for the requested department and category.
    Returns a FIPS key (where digits 0-1 = State, digits 2-4 = County)."""
    path = os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)], 
                        sub_dir_acs[int(dept)], 
                        sub_dir_acs_det[int(dept)][int(category)])
    file = [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)], 
                        sub_dir_acs[int(dept)], 
                        sub_dir_acs_det[int(dept)][int(category)])) if "ann" in f][0]
    full_path = os.path.join(path, file)
    
    df = pd.read_csv(full_path).head()
    FIPS_info = df.iloc[1, 1]
    return FIPS_info

def read_acsfile(dept,category):
    """This function reads in ACS data for the requested department and category.
    Returns a Pandas dataframe after reading in the relevant .csv file."""
    path = os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)], 
                        sub_dir_acs[int(dept)], 
                        sub_dir_acs_det[int(dept)][int(category)])
    file = [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", 
                        dept_list[int(dept)], 
                        sub_dir_acs[int(dept)], 
                        sub_dir_acs_det[int(dept)][int(category)])) if "ann" in f][0]
    full_path = os.path.join(path, file)
    
    df = pd.read_csv(full_path)
    return df


# ### Create functions to check the data
# And then we create a set of functions to check the data - again each is documented via docstring below:

# In[ ]:


def check_shapefiles():
    """This function checks availability of required mandatory shapefiles by department.
    Returns a list of lists containing shapefile extensions."""
    mandatory_files = ["shp", "shx", "dbf", "prj"]
    shapefile_check = []

    for i in range(len(dept_list)):
        row_check = []
        for file in mandatory_files:
            try:
                if [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", dept_list[i], sub_dir_shp[i])) if file in f][0]:
                    row_check.append(file)
            except:
                pass
        shapefile_check.append(row_check)
    return shapefile_check

def check_uoffiles():
    """This function checks availability of required use of force files by department.
    Returns a list UOF if available or None if not available."""
    uof_var = "UOF"
    uoffile_check = []

    for i in range(len(dept_list)):
        try:
            if [f for f in os.listdir(os.path.join("../input/data-science-for-good/cpe-data", dept_list[i])) if uof_var in f][0]:
                uoffile_check.append(uof_var)
        except:
            uoffile_check.append("None")
    return uoffile_check

def check_acsfiles():
    """This function retrieves the FIPS codes per ACS file by department.
    Returns a list of lists containing FIPS data.
    """
    FIPS_grid = []
    for j in range(5):
        cat_check = []
        for i in range(len(dept_list)):
            FIPS_info = read_acsfile_key(i,j)
            cat_check.append(FIPS_info)
        FIPS_grid.append(cat_check)
    return FIPS_grid

def color_exceptions(val):
    """Checks for defined exception values and highlights them in red."""
    color = 'red' if (val == "None") or (val == False) or (val == ['shp', 'shx', 'dbf']) else 'black'
    return 'color: %s' % color

def assemble_overview():
    """This function assembles an overview of the data provided and highlights any key issues,
    including 1) missing *.prj files 2) missing use of force files 3) inconsistent ACS data"""
    
    # Let's run the check_shapefiles and get a list of shapefiles available per department
    dept_shapes = check_shapefiles()

    # Let's run the check_uoffiles and get a list of uoffiles available per department
    dept_uofs = check_uoffiles()

    # Let's run the check_acsfiles and get a list of acsfiles available per department
    cat_acs = check_acsfiles()
    
    overview = pd.DataFrame({"dept": dept_list, 
                             "uofs": dept_uofs, 
                             "shapes": dept_shapes, 
                             inv_acs_dict[0] : cat_acs[0],
                             inv_acs_dict[1] : cat_acs[1],
                             inv_acs_dict[2] : cat_acs[2],
                             inv_acs_dict[3] : cat_acs[3],
                             inv_acs_dict[4] : cat_acs[4]})
    overview["state"] = overview["education"].str[0:2]
    overview["education"] = overview["education"].str[0:5]
    overview["education25"] = overview["education25"].str[0:5]
    overview["housing"] = overview["housing"].str[0:5]
    overview["poverty"] = overview["poverty"].str[0:5]
    overview["rsa"] = overview["rsa"].str[0:5]

    # Get county text
    county = []
    for i in overview.index:
        county_val = FIPS.loc[((FIPS["state_code"] == overview.loc[i, "education"][0:2]) & (FIPS["fips_county"] == overview.loc[i, "education"][2:5])), "description"].values[0]
        county.append(county_val)
    overview["county"] = county

    # Get state text keys
    state = []
    for i in overview.index:
        state_val = FIPS.loc[(FIPS["state_code"] == overview.loc[i, "state"]), "state"].values[0]
        state.append(state_val)
    overview["state"] = state

    # Re-assemble the df with fields in required order
    overview = overview[['dept', 'state', 'county', 'uofs', 'shapes', 'education', 'education25', 'housing',
           'poverty', 'rsa']]

    # Add a column reflecting if there are any issues with the ACS data
    acs_check = []
    for i in overview.index:
        eval = overview.loc[i, "education"] == overview.loc[i, "education25"] == overview.loc[i, "housing"] == overview.loc[i, "poverty"] == overview.loc[i, "rsa"]
        acs_check.append(eval)
    overview["acs_ok"] = acs_check

    # And finally, let's color any exceptions found and present our overview
    data_overview = overview.style.applymap(color_exceptions)
    
    return data_overview


# ### The result - what do we have?
# 
# So the main aim is to end up with a single neat Pandas df, which gives us an overview of our files, and so of the key issues at a glance:
# 
# - What county / state combinations does each Dept file represent?
# - Has a use of force (uof) file been provided or not?
# - In the provided shapefile data are any of the 4 key files missing?
# - In the ACS data were the files provided consistently for the same county?

# In[ ]:


overview = assemble_overview()
overview


# In the above output we see immediately:
# 
# 1. We have only been given use of force data for 3 of our 6 counties: Travis, Mecklenburg and Dallas.
# 2. The shapefile data from Travis is missing a "prj" file so we're going to need to identify the correct coordinate reference system before we can even thinking of analysing or plotting that data further.
# 3. And for both Travis and Worcester we have troublesome ACS data - notice that for Travis all entries are "48453", except for rsa (race-sex-age) where we've been given the data for FIPS county "48217". Similarly for Worcester the rsa data is for a different county.
# 
# Once we have data available for all 52 states, this quick analysis will give us a good idea of where we can dive in and start working with the data, and where we need to focus on further data preparation efforts before we begin.<br>
# 
# Dallas looks reasonably hassle-free and complete, so let's do some basic plotting using Dept_37-00049 as an example, just to get a feel for the process...<br>
# 
# (Note that when reading in data I'll be using the row indices to refer to requested departments and the columns education through rsa (0 through 4) will refer to the requested ACS data)

# ## Basic plotting - Dallas
# <b>Objective</b>: We have 3 separate datasets that we want to overlay: American Community Survey data, police shapefile data, and use of force data. The ACS data is consistently plotted using CRS = epsg:4269, so for my purposes I'm going to use that as my base standard - everything else must transformed to conform to this standard!

# ### Create functions required to prepare for plotting
# Our data needs some preparation before it can be plotted - again each is documented via docstring below:

# In[ ]:


# Note that it is not currently possible to run this function on Kaggle currently due 
# to the limitation around rtree https://github.com/Kaggle/docker-python/issues/108
def make_geodf_uof_acs(uoffile, long_col, lat_col, acsfile):
    """Converts the specified uoffile with longitude and latitude to a GeoPandas dataframe,
    and ensures that only UOF data within the supplied ACS file boundaries is displayed."""
    geometry = [Point(xy) for xy in zip(uoffile[long_col].astype("float"), 
                                    uoffile[lat_col].astype("float"))]
    uoffile = GeoDataFrame(uoffile, crs = "epsg:4269", geometry = geometry)
    uoffile = gpd.sjoin(uoffile, acsfile, how="inner", op='intersects')
    return uoffile

def make_geodf_uof(df, x_col, y_col):
    """Converts a pandas df to a GeoPandas df, using the specified x and y data to create 'geometry'."""
    geometry = [Point(xy) for xy in zip(df[x_col].astype("float"), 
                                    df[y_col].astype("float"))]
    df = GeoDataFrame(df, geometry = geometry)
    return df

def check_crs(shapefile):
    """The ACS data uses epsg:4269 as a standard, so our aim is to standardize all map co-ordinates on epsg:4269
    so that no matter where the data comes from it can be analysed and plotted within the same projection.
    Returns an evaluation of the relevant shapefile with recommendation where required."""
    target = 'epsg:4269'
    if shapefile.crs == {}:
        print("no initial projection - use find_crs() to find the correct projection") # see later in notebook
    elif shapefile.crs == target:
        print("no problem - ready for mapping")
    else:
        print("requires conversion - use conv_crs() to fix")
        
def conv_crs(shapefile):
    """Converts the specified shapefile from one CRS to our standard epsg:4269"""
    shapefile = shapefile.to_crs(epsg='4269')
    return shapefile


# ### Preparing our plot data
# The ACS data is quite nice and consistent - the same can't be said for the police data - this is another whole area of challenge! For now I'll manually do some identification and cleaning of these files just so we can check if our overlay objective is being achieved as expected.

# #### First the ACS data...

# In[ ]:


# Read in the ACS file for Dallas (3), Poverty (3) as per overview table above
dallas_acs = read_acsfile(3, 3)

# Rename the GEO.id column to AFFGEOID so it matches to the corresponding tract file column name
dallas_acs.rename(columns = {"GEO.id": "AFFGEOID"}, inplace = True)

# Merge the ACS file for Dallas, Poverty with the Texas tracts data
dallas_acs = dallas_acs.merge(texas_tracts, on = "AFFGEOID")

# And then convert the resulting df to a Gdf
dallas_acs = GeoDataFrame(dallas_acs, crs = "epsg:4269", geometry = dallas_acs["geometry"])

# And then let's look at the resulting output
fig, ax = plt.subplots(figsize = (12, 12))
# Plot the ACS data
ax = dallas_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
# Add a title
fig.suptitle('Dallax Texas, Census Tracts', x = 0.5, y = 0.89)
plt.show()


# #### And then the police file shape data...

# In[ ]:


# Let's read in the police shapefile data given
dallas_shapes = read_shapefile(3)
dallas_shapes.head()


# In[ ]:


# Let's check how the projection of our police shapefile lines up against our agreed standard for ACS
check_crs(dallas_shapes)


# In[ ]:


# If we print the crs values we can see they are completely different
print(dallas_acs.crs, "vs", dallas_shapes.crs)


# In[ ]:


# And we can see visually that it's problematic by trying to plot the ACS data and the police shapefile data together - 
# we get a plot with nothing in it!
fig, ax = plt.subplots(figsize = (10, 10))
ax = dallas_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
dallas_shapes.plot(ax = ax, color = "none", edgecolor = "b", linewidth = 1.5)
fig.suptitle('Dallas Texas, Census Tracts with Police Districts', x = 0.5, y = 0.88)
plt.show()


# In[ ]:


# That's OK though because remember we do have an initial CRS to work with so 
# let's use our function to convert to our standard epsg:4269
dallas_shapes = conv_crs(dallas_shapes)


# In[ ]:


# And if we plot ACS data and police shapefile data they are now lining up nicely
fig, ax = plt.subplots(figsize = (12, 12))
# Plot the ACS data
ax = dallas_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
# Add the shapefile data
dallas_shapes.plot(ax = ax, color = "none", edgecolor = "b", linewidth = 1.5)
# Add a title
fig.suptitle('Dallas Texas, Census Tracts with Police Districts', x = 0.5, y = 0.88)
plt.show()


# #### And then the use of force data...

# In[ ]:


# Now let's read in our UOF data for Dallas - in this case we're lucky as the incident co-ordinates are given in 
# latitude and longitude already...
dallas_uof = read_uoffile(3)
dallas_uof.columns


# In[ ]:


# We need to do a little cleaning of non-null, non-numeric and data type values before proceeding
dallas_uof.dropna(subset = ["LOCATION_LATITUDE"], inplace=True)
dallas_uof.dropna(subset = ["SUBJECT_RACE"], inplace=True)
dallas_uof.drop([0], inplace = True)
dallas_acs["HC02_EST_VC01"] = dallas_acs["HC02_EST_VC01"].astype("float")


# In[ ]:


# And now let's convert our use of force data to a GeoPandas dataframe
dallas_uof = make_geodf_uof(dallas_uof, "LOCATION_LONGITUDE", "LOCATION_LATITUDE")


# In[ ]:


# It would be nice to add labels to our data as a finishing touch so we'll get "representative points"
# for each shape which we'll use to plot our labels later on
dallas_shapes['coords'] = dallas_shapes['geometry'].apply(lambda x: x.representative_point().coords[:])
dallas_shapes['coords'] = [coords[0] for coords in dallas_shapes['coords']]


# In[ ]:


# Let's get some basic statistics for HC02_EST_VC01 ("Below poverty level; Estimate; Population for whom 
# poverty status is determined") - we'll use this to highight rich vs poor levels on our map
dallas_acs["HC02_EST_VC01"].describe()


# #### And finally our plot
# I'm quite convinced there are many ways to automate this as well (this would be my next challenge!) but for now let's get the job done :). It would also be nice to use the spatial join functionality mentioned above to ensure that only incidents within the census tract boundaries are plotted if desired (hopefully Kaggle will soon advise that rtree can be installed on the platform!).

# In[ ]:


fig, ax = plt.subplots(figsize = (12, 12))

# Plot the ACS data by poverty level
ax = dallas_acs.plot(ax = ax, color = "c", edgecolor = "darkgrey", linewidth = 0.5)
dallas_acs[dallas_acs["HC02_EST_VC01"] >= dallas_acs["HC02_EST_VC01"].describe()["50%"]].plot(ax = ax, color = "mediumturquoise", edgecolor = "darkgrey", linewidth = 0.5)
dallas_acs[dallas_acs["HC02_EST_VC01"] >= dallas_acs["HC02_EST_VC01"].describe()["75%"]].plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)

# Add the shapefile data
dallas_shapes.plot(ax = ax, color = "none", edgecolor = "b", linewidth = 1.5)

# Add the use of force data
dallas_uof.plot(ax = ax, color = "orangered", alpha = 0.5, markersize = 10)

# Some labels would also be nice!
texts = []
for i in dallas_shapes.index:
    text_item = [dallas_shapes["coords"][i][0] + 0.015, 
                  dallas_shapes["coords"][i][1] + 0.015, 
                  dallas_shapes["Name"][i]]
    texts.append(text_item)
    plt.text(texts[i][0], texts[i][1], texts[i][2], color = "black")

# And finally there are a lot of colours so a key will be useful   
low_line = matplotlib.lines.Line2D([], [], color='c',markersize=120, label='poverty - low')
med_line = matplotlib.lines.Line2D([], [], color='mediumturquoise',markersize=120, label='poverty - medium')
high_line = matplotlib.lines.Line2D([], [], color='paleturquoise', markersize=120, label='poverty - high')
police_line = matplotlib.lines.Line2D([], [], color='b', markersize=120, label='police precincts')
uof_line = matplotlib.lines.Line2D([], [], color='orangered', markersize=120, label='use of force')
handles = [low_line, med_line, high_line, police_line, uof_line]
labels = [h.get_label() for h in handles] 
ax.legend(handles=handles, labels=labels, fontsize = 10, loc='lower right', shadow = True)

# Add a title
fig.suptitle('Dallas Texas, Use of Force by Poverty Levels and Police District', x = 0.5, y = 0.88)
plt.show()


# ## Sorting out Travis!
# <b>Objective</b>: Travis was the department where no *prj file was available so we don't know the current projection and we need to find the right coordinate reference system before we can proceed with plotting the police-supplied data onto the census data.

# #### First the ACS data...

# In[ ]:


# Read in the ACS file for Travis(4), Poverty (3) as per overview table above
travis_acs = read_acsfile(4, 3)

# Rename the GEO.id column to AFFGEOID so it matches to the corresponding tract file column name
travis_acs.rename(columns = {"GEO.id": "AFFGEOID"}, inplace = True)

# Merge the ACS file for Dallas, Poverty with the Texas tracts data
travis_acs = travis_acs.merge(texas_tracts, on = "AFFGEOID")

# And then convert the resulting df to a Gdf
travis_acs = GeoDataFrame(travis_acs, crs = "epsg:4269", geometry = travis_acs["geometry"])

# And then let's look at the resulting output

fig, ax = plt.subplots(figsize = (12, 12))
ax = travis_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
fig.suptitle('Travis Texas', x = 0.5, y = 0.84)
plt.show()


# #### And then the police file shape data...

# In[ ]:


# Let's read in the police shapefile data given
travis_shapes = read_shapefile(4)
travis_shapes.head()


# In[ ]:


# Let's check how the projection of our police shapefile lines up against our agreed standard for ACS
check_crs(travis_shapes)


# In[ ]:


# If we print the crs values we can see that there is simply NO crs data available for Travis ({})
print(travis_acs.crs, "and", travis_shapes.crs)


# #### So let's turn to the use of force data...
# Because <i>this</i> data comes with addresses which are the key to helping us determine the right CRS to use...

# In[ ]:


# Now let's read in our UOF data for Travis - we seem to have 2 Y co-ordinates(!) as well as longitude and latitude -
# yikes!
travis_uof = read_uoffile(4)
travis_uof.columns


# In[ ]:


# A closer look at the data reveals that latitude and longitude are seldom given so there are 
# too many null values to be useful. The 2 "Y co-ordinates" are actually X and Y, they've just been
# mis-labelled by whatever process they went through previously. We also observe that our X and Y are
# certainly not latitude or longitude so we're going to have to find a suitable projection,
# AND we are given physical address data in this file, and this is going to provide us with the means
# to determine the right projection (LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION)
travis_uof.head(5)


# In[ ]:


# Let's re-name our columns for ease of use
travis_uof.rename(columns = {"Y_COORDINATE" : "X_COORDINATE", "Y_COORDINATE.1" : "Y_COORDINATE"}, inplace = True)


# In[ ]:


# And then do a little cleaning of non-null, non-numeric values
travis_uof.dropna(subset = ["X_COORDINATE"], inplace=True)
travis_uof.drop([0], inplace = True)
travis_uof.drop(list(travis_uof[travis_uof["X_COORDINATE"] == "-"].index), inplace = True)
travis_uof.drop([2094], inplace = True)
travis_acs["HC02_EST_VC01"] = travis_acs["HC02_EST_VC01"].astype("float")
travis_uof.head(5)


# ### Create a function to "detect" the best fit CRS
# Whatever inconsistencies there are in the police data, the one thing that DOES seem to be consistently supplied is the physical address of the incident. This is what will enable us to determine the correct CRS to use for the data - the docstring below explains the principle behind this function. 

# In[ ]:


def find_crs(rand_geodf_source):
    """Selects 3 random addresses from the specified GeoPandas df and retrieves latitude and longitude for them
    via Google maps API (this is then transformed to our standard epsg:4269. Selects a list of 'likely' projections
    - just based on State for now for demonstration purposes - and then tests what happens when we convert from that 
    CRS to our standard CRS. The CRS with the least difference in distance between our data and Google is deemed 
    the closest match and can be used for conversion."""
    # Now let's get corresponding locations for these addresses from Google
    rand_google = gpd.tools.geocode(rand_addresses.values, provider="googlev3", api_key = api_key)
    # And then convert to the standard projection we've decided upon
    rand_google.to_crs(epsg='4269')
    
    # Let's create a df where we'll store our evaluation data
    rand_eval = pd.DataFrame(rand_geodf_source["address"])
    
    # Get a list of projections to try
    projection_tries = projection_df[projection_df["PROJ Description"].str.contains(usps_df.loc[usps_df["USPS"] == rand_geodf_source["state"][0], "State"].values[0])]
    projection_tries.reset_index(drop = True, inplace = True)
        
    for i in range(len(projection_tries)):
        # First we make a copy of our source data to work on
        rand_geodf = rand_geodf_source.copy()

        # Let's now set the crs to the first one we want to try
        rand_geodf.crs = {'init' : projection_tries["PROJ"][i]}
        # And then convert to our standard
        rand_geodf = rand_geodf.to_crs(epsg='4269')

        # And let's store the outcomes of our first test
        rand_eval[projection_tries["PROJ"][i]] = rand_geodf.distance(rand_google)
    
    # Find the mean of the values for each column
    rand_eval.loc['avg'] = rand_eval.mean()
    
    # Find the best fit
    answer = rand_eval.loc['avg'].dropna().sort_values().index[0]
    
    return answer, rand_eval


# In[ ]:


# Now we can randomly pick 3 addresses we'll use to validate on
rand_addresskeys = list(np.random.randint(1,len(travis_uof), 3))

# We need the full address for best geo-coding results
travis_uof["FULL_ADDRESS"] = travis_uof["LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION"] + ", " + travis_uof["LOCATION_CITY"] + ", " + travis_uof["LOCATION_STATE"]

# Now let's assemble the 3 series we'll use to create our test df
rand_addresses = travis_uof.loc[rand_addresskeys, "FULL_ADDRESS"]
rand_state = travis_uof.loc[rand_addresskeys, "LOCATION_STATE"]
rand_x = travis_uof.loc[rand_addresskeys, "X_COORDINATE"]
rand_y = travis_uof.loc[rand_addresskeys, "Y_COORDINATE"]

# And finally create the test df
rand_address_table = pd.DataFrame({"address" : rand_addresses.values, 
                                   "state" : rand_state,
                                   "X_COORDINATE": rand_x.values, 
                                   "Y_COORDINATE": rand_y.values})

# Convert our rand_address_table to a rand_geodf (a GeoPandas df)
rand_geodf_source = make_geodf_uof(rand_address_table, "X_COORDINATE", "Y_COORDINATE")
rand_geodf_source.reset_index(drop = True, inplace = True)


# ### The result - what do we have?
# 
# - We now have the <b>answer</b> to "which co-ordinate reference system was used to compile this data?" 
# - We can also examine the contents of <b>rand_eval</b> to see which CRS's were our top contenders

# In[ ]:


# Run the find_crs function and then display our final answer
answer, rand_eval = find_crs(rand_geodf_source)
answer


# In[ ]:


# And also have a look at the top 5 options - notice that there are in fact 4 different projections
# that would minimize the difference between our 'ground truth' co-ordinates obtained from Google
# and our new projections based on our chosen CRS
rand_eval.loc["avg"].dropna().sort_values().head()


# In[ ]:


# Let's now convert our Travis dataframe to a GeoPandas dataframe
travis_uof = make_geodf_uof(travis_uof, "X_COORDINATE", "Y_COORDINATE")


# In[ ]:


# And then specify the CRS we identified as best fit
travis_uof.crs = {'init' : answer}
travis_shapes.crs = {'init' : answer}


# In[ ]:


# After which we can convert to our standard
travis_uof = travis_uof.to_crs(epsg='4269')
travis_shapes = travis_shapes.to_crs(epsg='4269')


# In[ ]:


# It would be nice to add labels to our data as a finishing touch so we'll get "representative points"
# for each shape which we'll use to plot our labels later on
travis_shapes['coords'] = travis_shapes['geometry'].apply(lambda x: x.representative_point().coords[:])
travis_shapes['coords'] = [coords[0] for coords in travis_shapes['coords']]


# In[ ]:


fig, ax = plt.subplots(figsize = (12, 12))
# Plot the ACS data
ax = travis_acs.plot(ax = ax, color = "paleturquoise", edgecolor = "darkgrey", linewidth = 0.5)
# Add the shapefile data
travis_shapes.plot(ax = ax, column = "DISTRICT", cmap = "viridis", vmin = 1.8)
# Add the use of force data
travis_uof.plot(ax = ax, color = "orangered", alpha = 0.2, markersize = 10) 
# Provide a legend
uof_line = matplotlib.lines.Line2D([], [], color='orangered', markersize=120, label='use of force')
handles = [uof_line]
labels = [h.get_label() for h in handles] 
ax.legend(handles=handles, labels=labels, fontsize = 10, loc='lower right', shadow = True)
# Add a title
fig.suptitle('Travis Texas, Use of Force by Police District', x = 0.5, y = 0.82)
plt.show()


# ### Conclusion

# This is just a "proof of concept" type notebook, but with some additional effort to automate the above process for all departments and include robust error handling I think it might lay a good foundation! But let me know what you think - this is my first kernel submission on Kaggle, so be gentle :)... https://shotlefttodatascience.com/

# In[ ]:




