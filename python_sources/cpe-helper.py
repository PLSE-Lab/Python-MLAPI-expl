#!/usr/bin/env python
# coding: utf-8

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# # CPE helper
# 
# Hello folks! This is the main kernel and where I will present my answer. I've been working a lot the last two months to provide an answer that would be really good for CPE and the things they do.

# # Summary
# 
# TL;DR: I've created a file processor that has two main functions: generate population statistics for each police precinct and assist with the standardization of the police data.
# 
# *Take a look at the code if possible. I believe it is well documented and one of the best aspects of the solution:*
# 
# - [example function (areal interpolation)](https://github.com/araraonline/kag-cpe/blob/master/cpe_help/util/interpolation.py)
# - [project on github](https://github.com/araraonline/kag-cpe)
# 
# ![](http://policingequity.org/wp-content/themes/bkt/images/center-for-policing-equity.png)
# 
# Center for Policing Equity (CPE) is a research center that is focused on justice and racial equity. To put in a another light, they produce important research that helps law enforcement agencies and communities to forge a way towards mutual trust and public safety.
# 
# Now, CPE faces a sort of trouble, and that is the huge amount and huge variety in the data that they receive. Basically, they have information on:
# 
# - Location of police precincts
# - Use of force incidents
# - Officer-involved shootings
# - Vehicle stops
# - Etc
# 
# Another issue is that they need to aggregate data found at the police level and data found at the census level. I believe this was the main thing that was asked in this challenge and it is where my answer comes in...
# 
# I came up with a tool that automates a big part of the process. First, it retrieves demographic characteristics that are to be used in the analysis. Second, it makes those characteristics available at the police precinct level.
# 
# These are the main functions. But also, there is a full python package that gives support to it, providing a nice framework for CPE, be it if they decide to automate more of the process or if they just want a quick set of tools to assist with the daily work.

# # Demonstration
# 
# *results here came from a local run (not on the Kaggle kernel)*

# In[ ]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# In[ ]:


import pathlib
import sys

import pandas

get_ipython().run_line_magic('matplotlib', 'inline')


# import package
ROOT_DIR = '../input/kaggle-cpe/cpe/cpe/'
ROOT_DIR = str(pathlib.Path(ROOT_DIR).resolve())
sys.path.append(ROOT_DIR)

from cpe_help import Department, util


dept = Department.sample()


# ## ACS Data

# ### City level

# In[ ]:


path = dept.output_dir / 'acs' / 'city.geojson'
df = util.io.load_geojson(path)
df


# In[ ]:


df.plot();


# ### Census tract level

# In[ ]:


path = dept.output_dir / 'acs' / 'census_tracts.geojson'
df = util.io.load_geojson(path)
df.head()


# In[ ]:


df.plot();


# ### Block group level
# 
# <https://en.wikipedia.org/wiki/Census_block_group>

# In[ ]:


path = dept.output_dir / 'acs' / 'block_groups.geojson'
df = util.io.load_geojson(path)
df.head()


# In[ ]:


df.plot();


# ### Police precincts level

# In[ ]:


path = dept.output_dir / 'acs' / 'police_precincts.geojson'
df = util.io.load_geojson(path)
df.head()


# In[ ]:


df.plot();


# ## Individual department files
# 
# Individual department files can also be processed. For example, the use of force file for the Austin Police Department had coordinate variables in a non-usual CRS (coordinate reference system). The processing step transformed those coordinates into the latitude and longitude values.

# In[ ]:


dept = Department('37-00027')


# Original file:

# In[ ]:


df = dept.files['uof'].load_raw()
df.head()


# Processed file (note that this table has a `geometry` column with points in it and that the `INCIDENT_DATE` was standardized:

# In[ ]:


df = dept.files['uof'].load_processed()
df.head()


# In[ ]:


df = df[df['LOCATION_GEOCODED'].astype(bool)]
df.plot(markersize=5);


# ## Python package
# 
# The answer in itself is a python package. So, if you are using python, you can access department methods directly.

# In[ ]:


from cpe_help import Department

dept = Department.sample()
dept


# In[ ]:


dept.full_name


# In[ ]:


df = dept.load_police_precincts()
df.head()


# In[ ]:


df.plot();


# ## Output directory
# 
# If you want to retrieve the outputs manually, they are stored in a specific directory for each department.
# 
# For example:

# In[ ]:


for file in util.file.list_files(dept.output_dir):
    print(file.relative_to(util.path.BASE_DIR))


# # How to use it?

# ## Preparation
# 
# Before running the project, you will first need to do some preparation steps...

# ### Clone the repository
# 
# The project itself is being hosted at github. If you have git, you can clone it by running the command:
# 
# ```bash
# git clone https://github.com/araraonline/kag-cpe
# ```
# 
# If you do not have git, you can download the repository directly from the project page:
# 
# 1. Go to <https://github.com/araraonline/kag-cpe>
# 2. Click "Clone or download" and then ""Download ZIP"
# 3. Once you download the zip file, extract it to a directory of your choice

# ### Install conda
# 
# Conda is a package and environment manager that is used to install the dependencies for the project. You can retrieve it by installing Anaconda or Miniconda. Miniconda is a lightweight version of Anaconda, I usually opt for it!
# 
# - [Download Anaconda](https://www.anaconda.com/download/)
# - [Download Miniconda](https://conda.io/miniconda.html)

# ### Create the conda environment
# 
# To run the project, you will need a custom conda environment. You can create one by:
# 
# 1. Start the Anaconda prompt (Windows) or terminal (Linux/MacOS)
# 2. Move into the project root directory (`cd` into it)
# 3. Update conda using the following command
# 
#    ```
#    conda update conda
#    ```
# 
# 4. Create the environment
# 
#    ```
#    conda env create -f environment.yml
#    ```
# 
# The last step may take a while (don't worry if it looks stuck at the `Solving...` phase, it is just calculating things). Once it finishes, take a look at the final lines of output and, it's all done!

# ## Configuration
# 
# It is also possible to configure some things.
# 
# Most of the configuration is present in the `cpe_help.conf` file. It is in a simple format that resembles windows INI files.
# 
# Some parameters:
# 
# 
# ### Census year
# 
# This is the year Census data will be retrieved from. For example, if it is configured to 2015, the machine will retrieve ACS 5-year estimates from 2011 to 2015 and TIGER shapefiles from 2015. As I'm writing, there's data available up to 2017 ([release](https://www.census.gov/programs-surveys/acs/news/data-releases/2017/release.html)).
#   
# *Do not change the year without checking ACS variables first*. The name of the variable on the ACS endpoint [may vary from year to year](https://www.census.gov/programs-surveys/acs/technical-documentation/table-and-geography-changes.html). This can lead the machine to download erroneous values.
# 
# ### Census key
# 
# This is the key that will be used in the requests to the Census API. Make sure to **[request a new key](https://api.census.gov/data/key_signup.html)** and set it up in the configuration file.
# 
# ### ACS Variables
# 
# Represents variables that will be requested from the Census API (ACS 5-year esimates).
#   
# The values to left of ` = ` correspond to the variable names in the Census Data API. The values to the right correspond to how we want to save this variable locally.
#   
# You can look at this [little guide](https://www.kaggle.com/center-for-policing-equity/data-science-for-good/discussion/70489) if you think of adding new variables.
#   
# Also, make sure that the variables are available at the block group level (some aren't, even if they are part of a detailed table).
# 
# ### Date format
# 
# Sets the standard date and time format for output ([reference](http://strftime.org/)).
# 
# ### User-Agent
# 
# Determines the [User-Agent string](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent) sent alongside requests. It should contain identification and possibly a way to contact CPE.
# 
# Some pointers on choosing the User-Agent string: <https://webmasters.stackexchange.com/a/6305>

# ## Running
# 
# This is the easiest step, but also takes a long time on the first run.
# 
# You must first activate your conda environment:
# 
# - Windows: `activate cpe-kaggle`
# - Linux/MacOS: `source activate cpe-kaggle`
# 
# Then, make sure you are in the package main directory and run the preparation script (it will create the necessary directories for departments/etc):
# 
# ```
# doit -f prepare_kaggle.py
# ```
# 
# After that, you can start the script for the main pipeline:
# 
# ```
# doit
# ```
# 
# *(when no file is specified, the default file executed is the dodo.py)*
# 
# After this, you will see a long list of tasks being executed. (message me if you've got any errors). Their names are pretty indicative. You can see what they are actually doing by reading the dodo.py source code or get a simple description by running the command `doit list`.
# 
# Take note that some tasks will take a long time to run. This is due to them being CPU-intensive or downloading a big amount of data.

# ## Customizing departments
# 
# Each department can have a specific class linked to it (which defaults to the Department class, if not specified). To create custom behavior for a specific department (say, department 37-00027):
# 
# 1. Create a `department3700027.py` file at the `cpe_help/departments` directory.
# 2. Inside the file, declare a subclass of Department named Department3700027. For example:

# In[ ]:


from cpe_help import Department

class Department3700027(Department):
    # code for the department goes here
    # ...
    pass


# 3. Override/create any of the methods you want
# 
#    For example, if you want to set a specific CRS when loading the input shapefile, you can override the `load_external_shapefile` method:

# In[ ]:


from cpe_help import Department, util

class Department3700027(Department):

    # NAD 1983 StatePlane Texas Central FIPS 4203 Feet
    CRS = util.crs.from_esri(102739)

    def load_external_shapefile(self):
        # set up CRS when loading police boundaries
        df = super().load_external_shapefile()
        df.crs = self.CRS
        return df


# And it's done!
# 
# *you can also add department-specific files for precessing, check the [DepartmentFile](#DepartmentFile) section below*

# # How does it work?
# 
# In this section, I will get a little more technical and show the thing works.
# 
# I will also try to provide some examples of how the pieces are assembled together.

# ## Pipeline
# 
# First of all, this machine is a pipeline. That is, a series of steps that can be taken in a determined order. To assemble these steps together, I used the [doit](http://pydoit.org/) package. This package  is just like Make, but a bit more tasty and with python bindings, easing the job a lot.
# 
# So, doit abstracts pipelines as a series of tasks,... Each task has some dependencies and some targets. For example, our dependencies are the inputs files and our targets are the outputs. There's also a recipe for each step, and, amazingly, each step will only be calculated if it needs to be.

# ## Main Classes

# ### Department
# 
# In the hope of allowing ease customization, I took an object-oriented approach, setting a department as a class. Each class has methods for load, saving and processing files of the department.
# 
# If we want to load department 11-00091 class, just call the Department constructor:

# In[ ]:


dept = Department('11-00091')
dept


# This will return the generic class department. However, if a specific class has been added for the department in question, it will be instantiated instead:

# In[ ]:


dept = Department('37-00027')
dept


# This contain exclusive handling that involves the loading of shapefiles and the processing of a UOF file.

# ### DepartmentFile
# 
# Departments may have some files that are specific to them. Each department file must be processed in a specific manner, and you can do so with a DepartmentFile.
# 
# For example, to create a new DepartmentFile:

# In[ ]:


import pandas

from cpe_help import DepartmentFile, util


# real UOF example for department 37-00027 (Austin, Texas)

class UOF(DepartmentFile):

    def __init__(self, department):
        self.department = department

    @property
    def raw_path(self):
        directory = self.department.tabular_input_dir
        return directory / '37-00027_UOF-P_2014-2016_prepped.csv'

    @property
    def processed_path(self):
        directory = self.department.other_output_dir
        return directory / 'uof.geojson'

    def load_raw(self):
        return pandas.read_csv(
            self.raw_path,
            low_memory=False,
            skiprows=[1],
        )

    def load_processed(self):
        return util.io.load_geojson(self.processed_path)

    def process(self):
        # loads raw file, processes it and saves output to self.processed_path
        pass


# This represents a specific department file for a specific department. We can plug it in the department simply by modifying its `files` attribute of the department. Iike so:

# In[ ]:


import collections


class Department3700027(Department):

    @property
    def files(self):
        return collections.OrderedDict([
            ('uof', UOF(self)),
        ])


# DepartmentFile's will be automatically retrieved and processed when you run the pipeline. Also, you can access their contents by calling one of their methods:

# In[ ]:


dept = Department('37-00027')
df = dept.files['uof'].load_processed()
df.head()


# ## Other things
# 
# Here are some aspects that are not crucial to the understanding of the program itself, but may be useful if you want to modify something some of the code later on.

# ### Census API
# 
# One of the requests of this challenge involved using data from the ACS to generate statistics for each police precincts. I noticed that this data was being retrieved manually, and at census tract level. So, I went one step ahead and created an automation tools that would retrieve the variables you want and interpolate those values into the police precinct areas.
# 
# The variables themselves can be changed using the configuration files and I think this shall be a very useful tool for CPE.

# ### TIGER shapefiles
# 
# We are initially provided boundaries for each police district in a department. With these precincts, we can localize where the department is (city/state/county).
# 
# This localization is essential because we want to retrieve demographic values only for the needed areas.
# 
# To do so, we use a Census service called TIGER (Topologically Integrated Geographic Encoding and Referencing).
# 
# You can read more about TIGER [here](https://www.census.gov/geo/maps-data/data/tiger.html).

# ### Areal interpolation
# 
# Areal interpolation is the name of the method for using information from a set of polygons into a set of another polygons. This is what Chris provided in an initial example.
# 
# The type of areal interpolation used here is *weighted areal interpolation*. There's more information about it and other methods in the [Limitations](#Limitations) section.

# # Last considerations

# ## Limitations

# ### UOF files
# 
# I didn't have enough time to actually deal with the preprocessing of the provided UOF files (and other files also, like OIS, vehicle stops, etc). This is something that I would visit with much care, thinking, how much standardization can we do before we lose the original shape of the data.
# 
# ### Geocoding
# 
# This is linked with the above. Lots of department files had spatial data encoded in the form of an address. They can be transformed into geographic coordinates by [geocoding](https://en.wikipedia.org/wiki/Geocoding), but this was left out of scope for my solution.
# 
# As you have already worked with Google, you may try to cooperate with them on this, since Google has one of the best geocoding services out there. You may also roll your own solution.
# 
# ### Areal interpolation
# 
# To transform the statistics retrieved from the Census API into statistics for police departments, one would need to use [areal interpolation][1].
# 
# The method used here is *weighted* areal interpolation. It is one of the simplest methods available, so, it may not come with the biggest accuracy.
# 
# There has been lots of research on this area... Lam provides a [review][2] of the areal interpolation methods that were available by 1983.
# 
# The method that I would use if resources were available, is dasymetric mapping. It makes use of ancillary variables to improve the accuracy of predictions over the target regions.
# 
# Mennis (2015) provides an [excellent example][3] of using zoning data to increase the accuracy of urban population analysis.
# 
# One problem with the areal interpolation thing is that there are not enough tools available, and the ones that are available are coupled into ArcGIS. There is, however python package available to interact with ArcGIS ([arcpy][4]). So, maybe that's the way to go.
# 
# 
# [1]: http://desktop.arcgis.com/en/arcmap/latest/extensions/geostatistical-analyst/what-is-areal-interpolation.htm
# [2]: https://pdfs.semanticscholar.org/1d0e/c81b45f4cef124d9369cc8d8f4883f6a8c22.pdf
# [3]: https://www.huduser.gov/portal/periodicals/cityscpe/vol17num1/ch9.pdf
# [4]: http://desktop.arcgis.com/en/arcmap/latest/analyze/arcpy/what-is-arcpy-.htm
# 
# ### Margin of Error
# 
# Another limitation of the machine is the lack of margin of error for predicted variables. I believe this margin of error could be calculated using [Variance Replicate Tables](https://www.census.gov/programs-surveys/acs/data/variance-tables.html), but, to be frank, my knowledge in the area is not enough. Also, there's the issue of integrating this with areal interpolation (that may generate an error of its own kind) and the large amount of complexity that would need to be added into the codebase.
# 
# ### Year
# 
# I don't believe this is a serious impedance, but, it's more of something to be considered. Currently, you can only specify one year in the configuration file and then:
# 
# - TIGER will retrieve boundaries for that year
# - ACS will retrieve 5-years estimates ending in that year

# ## Some suggestions
# 
# ### Call Rscript from python
# 
# I've noticed you using R in your work. While this answer is presented as python, it may be actually easy to integrate R into it, for example, if you wanna generate some plots with ggplot.
# 
# The way to do it would be to create a R script that does the function you want, and then call this R script from python. Like so:
# 
# ```
# import subprocess
# 
# subprocess.run([
# 	'Rscript',
# 	'foo.R',
# 	'arg1',
# 	'arg2',
# ])
# ```
# 
# You can also try the [rpy2](https://rpy2.readthedocs.io/) package.
# 
# Ref: [Rscript function | R Documentation](https://www.rdocumentation.org/packages/utils/versions/3.5.1/topics/Rscript)
# 
# ### Ask for missing data
# 
# When there's missing data, like in the shapefile where there were missing points, I think the best thing you can do is ask the police for the missing data. We can try anything in the code, but, anything we try, it won't be better than the actual data.
# 
# ### Date and time format
# 
# The date and time columns in the data are present in the most varied formats. I recommend using a standard, like [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).
# 
# ## A little note about levels of automation
# 
# There's always a doubt of how deep this processing can go. In an ideal place, all departments would give CPE all the data theiy need and this data would come clean and standardized... In a situation like this, it would be pretty simple to generate an script to, say, for each department, generate some analysis and reports...
# 
# The data that is coming from departments, however, is not a sea of roses. It comes in the most varied formats with various levels of detail. This makes it hard to create an automation method that is valid for all files. Essentially, we'd be method for each one of them. In this same light, I believe CPE did an awesome job at coming up with hand-tailored analysis for each department.
# 
# But also, it makes me question where could automation be useful? I believe that, in this case, it could be useful to keep two versions of each file. One version, the raw, would be manually used by CPE to provide customized analysis that can be so very flexible and beautiful. The other version, standardized, could be used to feed an automation pipeline and then generate figures, reports, maps, etc.
# 
# The `DepartmentFile` class was based on this line of thought.
# 
# ## Trouble?
# 
# If you have any trouble for the installation or running steps, please, leave a comment below! I will try to help the best I can.
# 
# ## Thanks
# 
# I would like to thank everyone involved in this challenge. Two people that were actually special are Kat and Chris. Kat is the representative for CPE and I think she is helping organize this competition. Chris is on the Kaggle side and he's the main organizer here. Both were really helpful whenever needed, and provided crucial information for the challenge! Also, they are really nice :)
# 
# Also sending my regards to Shivam Bansal and Jose Berengueres that are being great companions through the end of this competition.
# 
# Besides people here, I want to thank my friend Zin that's helping me with internet issues and Windows issues right now...
# 
# Now I gotta go and upload my code!
# 
# Bye bye!
