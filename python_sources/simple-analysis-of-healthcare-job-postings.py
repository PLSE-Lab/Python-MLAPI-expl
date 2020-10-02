#!/usr/bin/env python
# coding: utf-8

# <u><h1>Simple analysis of healthcare job openings</h1></u><br>
# ***Yogesh Virkar***  -- *11/15/2018*

# <h1>Part 1: Convert XML format to pandas dataframe</h1>
# Feel free to skip this part if you are not interested in knowing how parsing the xml file works. You can simply copy over the classes for your own analysis and checkout the example usage in Part 1.3 inorder to convert the xml file to a pandas dataframe. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# <h2>Part 1.1: As the first step, we write classes that define all the tags associated with the XML tree for the given input .xml file. The XMLTagsUpperLevel defines all the tags at the higher level in the XML tree and the XMLTagsLowerLevel defines all the tags in the lower level (at the record definition level) of the XML tree. Subclassing XMLTagsLowerLevel from Enum python class allows us to iterate over all the tags defined for a record. Coding the parser becomes quite convenient as a result. </h2>

# In[ ]:


# XML model
from enum import Enum


class XMLTagsUpperLevel:
    """
    This class defines the XML tag constants at the higher level of XML tree. The tag <record> is found below the tag
    <page> in the tree hierarchy.
    """
    PAGE = "page"
    RECORD = "record"


class XMLTagsLowerLevel(Enum):
    """
    This class defines all the XML tag constants that are one level below the <record> tag. This is defined as an
    enumerated type for ease of iterating over all tags.
    """
    UNIQUE_ID = "uniq_id"
    CRAWL_TIMESTEMP = "crawl_timestamp"
    URL = "url"
    JOB_TITLE = "job_title"
    CATEGORY = "category"
    COMPANY_NAME = "company_name"
    POST_DATE = "post_date"
    JOB_DESCRIPTION = "job_description"
    JOB_TYPE = "job_type"
    SALARY_OFFERED = "salary_offered"
    JOB_BOARD = "job_board"
    GEO = "geo"
    LOCATION = "location"


# <h2>Part 1.2: The following code defines the XML parser required to parse the xml file and store the data into a pandas dataframe. </h2>

# In[ ]:


# XML parser 
import xml.etree.ElementTree as ET

class XMLParser:
    BASE_PATH = "../input"
    
    
    def __init__(self, file_path=BASE_PATH + "/emedcareers_eu_job_board_common_setup_pc_jobspikr_deduped_n_merged_20180822_071129200803658.xml"):
        """
        Initializes the XMLParser class instance.
        :param file_path: Path to input xml file containing all the jobs data.
        """
        self.file_path = file_path


    def xml_to_pandas_df(self):
        """
        Using the standard xml python library, we parse the data xml file and convert the xml data to a pandas
        data frame.
        :return: A pandas data frame instance containing all the jobs data.
        """
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        jobs_data = dict()
        for tag in XMLTagsLowerLevel:
            jobs_data[tag.value] = []

        for i, page in enumerate(root.findall(XMLTagsUpperLevel.PAGE)):
            for j, record in enumerate(page.findall(XMLTagsUpperLevel.RECORD)):
                for tag in XMLTagsLowerLevel:
                    temp = record.find(tag.value)
                    if temp is not None:
                        jobs_data[tag.value].append(temp.text)
                    else:
                        jobs_data[tag.value].append("")

        return pd.DataFrame(data=jobs_data)


# <h2>Part 1.3: The following code demonstrates the usage of the XML parser. The final data is obtained in the jobs_df pandas dataframe variable. </h2>

# In[ ]:


base_path = "../input"
file_path = base_path + "/" + os.listdir("../input")[0]
parser = XMLParser(file_path)
jobs_df = parser.xml_to_pandas_df()
print("----------------------------")
print("Dataframe columns:")
print("----------------------------")
print(jobs_df.columns)
print("----------------------------")
print("Sample data:")
print("----------------------------")
print(jobs_df.head())
print("----------------------------")
print("Dataframe shape:")
print("----------------------------")
print(jobs_df.shape)


# <h1>Part 2: Analysis</h1>
# <h2>Part 2.1: Top-k frequency distributions</h2>

# In[ ]:


# Import libraries needed for plotting 
import matplotlib.pyplot as plt
import seaborn as sns


# ***Define helper function to manipulate dataframe: For the input dataframe "df", group the rows by the column defined by the parameter "by" and return the number of rows per value in the by column in a separate column titled COLUMN_COUNTS***

# In[ ]:


COLUMN_COUNTS = "counts"


def groupby_and_get_number_of_rows(df, by):
    """
    For the input dataframe df, group the rows by the column defined by the parameter by and return the number of rows
    per value in the by column in a separate column titled COLUMN_COUNTS. 
    :param df: input pandas dataframe
    :param by: input column name by which we group the rows in df
    :return: A dataframe with first column as by and second column as COLUMN_COUNTS
    """
    return df.groupby([by]).size().reset_index(name=COLUMN_COUNTS)


# ***Define another helper function to plot frequency result***

# In[ ]:


__X_LABEL = "frequency"
__Y_LABEL_1_STR = "{} (top {} out of {})"
__Y_LABEL_2_STR = "{}"


def plot_top_k_frequency_distribution_for_df_property(column_count_df, df_property, plot_properties, top_k):
    x_labels = column_count_df[df_property]
    n_x_labels = len(x_labels)
    if top_k > n_x_labels:
        top_k = n_x_labels
    plt.figure(figsize=(plot_properties.width, plot_properties.height))
    # _, ax = plt.subplots()
    sns.set(font_scale=plot_properties.font_scale)
    h1 = sns.barplot(y=df_property, x=COLUMN_COUNTS, data=column_count_df.head(n=top_k),
                     color=plot_properties.bar_plot_color, alpha=plot_properties.alpha)
    h1.set_yticklabels(list(x_labels[:top_k]))
    h1.set_xticks(plot_properties.xticks)
    y_label = __Y_LABEL_2_STR.format(df_property) if top_k == n_x_labels else __Y_LABEL_1_STR.format(df_property, top_k,
                                                                                                     n_x_labels)
    plt.xlabel(__X_LABEL)
    plt.ylabel(y_label)
    plt.show()


# ***A class that defines the default plot properties ***

# In[ ]:


class DefaultPlotProperties:
    WIDTH = 60
    HEIGHT = 40
    FONT_SCALE = 3
    BAR_PLOT_COLOR = [0, 0.5, 0.7]
    ALPHA = 0.8
    XTICK_ROTATION = 90
    XTICKS = range(0, 1000, 100)


# ***A class that defines the plot properties ***

# In[ ]:


class PlotProperties:
    def __init__(self,
                 width=DefaultPlotProperties.WIDTH,
                 height=DefaultPlotProperties.HEIGHT,
                 font_scale=DefaultPlotProperties.FONT_SCALE,
                 bar_plot_color=DefaultPlotProperties.BAR_PLOT_COLOR,
                 alpha=DefaultPlotProperties.ALPHA,
                 xtick_rotation=DefaultPlotProperties.XTICK_ROTATION,
                 xticks=DefaultPlotProperties.XTICKS):
        self.width = width
        self.height = height
        self.font_scale = font_scale
        self.bar_plot_color = bar_plot_color
        self.alpha = alpha
        self.xtick_rotation = xtick_rotation
        self.xticks = xticks


# ***Though most folk do not use much object-oriented principles when writing Python code, I have found it somewhat useful to write modular code for the ease of code reuse and  maintainability. So the above is just my attempt at modular design to hopefully make anlaysis easier. ***

# <h3>Part 2.1.1: Highest number of job posting by a particular company (plotting top 30)</h3>

# In[ ]:


num_jobs_per_company_df = groupby_and_get_number_of_rows(jobs_df, by=XMLTagsLowerLevel.COMPANY_NAME.value)
num_jobs_per_company_df.sort_values([COLUMN_COUNTS], ascending=False, inplace=True)
print(num_jobs_per_company_df.head())
plot_properties = PlotProperties(xticks=range(0, 5000, 500))
plot_top_k_frequency_distribution_for_df_property(num_jobs_per_company_df, XMLTagsLowerLevel.COMPANY_NAME.value, plot_properties, top_k=30)


# **<h3>Part 2.1.2: Location with highest job openings (plotting top 30)</h3>**

# In[ ]:


num_jobs_per_location_df = groupby_and_get_number_of_rows(jobs_df, by=XMLTagsLowerLevel.LOCATION.value)
num_jobs_per_location_df.sort_values([COLUMN_COUNTS], ascending=False, inplace=True)
print(num_jobs_per_location_df.head())
plot_properties = PlotProperties(bar_plot_color=[0.7, 0, 0], xticks=range(0, 8000, 500))
plot_top_k_frequency_distribution_for_df_property(num_jobs_per_location_df, XMLTagsLowerLevel.LOCATION.value, plot_properties, top_k=30)


# ***I am trying to figure out why some of the company names are weird such as "0203 8757550" seen in the top 30 companies for the number of job postings in 2.1.1 and why "Science" appears as a location in part 2.1.2. More will be added soon... ***

# In[ ]:




