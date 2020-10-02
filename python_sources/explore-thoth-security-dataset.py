#!/usr/bin/env python
# coding: utf-8

# # Context

# Thoth Security Dataset is part of a series of datasets related to observations regarding software stacks (e.g. dependency tree, installability, performance, security, health) as part of Project Thoth. All these datasets can be found also here where they are described and explored to facilitate their use. All these observations are created with different components which are part of Project Thoth and stored in Thoth Knowledge Graph which is used by Thoth Adviser to provide advises on software stacks depending on User requirements.

# # Goal

# The goal is to provide datasets widely available and useful for data scientists. Thoth Team within the office of the CTO at Red Hat has collected datasets that can be made open source within the IT domain for training Machine Learning models.

# # Content

# Thoth Security Dataset contains two folders containing outputs from two Thoth Security Indicators (SI) Analyzers:
# 
# 1. [SI-bandit](https://github.com/thoth-station/si-bandit) is an analyzer for security indicators based on [bandit](https://pypi.org/project/bandit/) Python package,
#     a tool designed to find common security issues in Python code. This Python package has different [classes of tests](https://readthedocs.org/projects/bandit/downloads/pdf/latest/):
# 
#     - B1xx misc tests
#     - B2xx application/framework misconfiguration
#     - B3xx blacklists (calls)
#     - B4xx blacklists (imports)
#     - B5xx cryptography
#     - B6xx injection
#     - B7xx XSS
# 
#     Each test in a group has two assigned parameters:
# 
#     - level of SEVERITY.
#     - level of CONFIDENCE.
# 
#     that are manually assigned.
# 
# 2. [SI-cloc](https://github.com/thoth-station/si-cloc) is an analyzer for security indicators based on [cloc](https://github.com/AlDanial/cloc) RPM package
#     that counts blank lines, comment lines, and physical lines of source code in many programming languages.
#     It's important to take into account some of the known [limitations](https://github.com/AlDanial/cloc#limitations-) for this package:
# 
#     - Lines containing both source code and comments are counted as lines of code.
#     - Python docstrings can serve several purposes. They may contain documentation, comment out blocks of code,
#     or they can be regular strings (when they appear on the right hand side of an assignment or as a function argument).
#     cloc is unable to infer the meaning of docstrings by context; by default, cloc treats all docstrings as comments.
#     The switch ``--docstring-as--code`` treats all docstrings as code.
#     - Language definition files read with ``--read-lang-def`` or ``--force-lang-def`` must be plain ASCII text files.

# # How you can use the Data

# You can download and use this data for free for your own purpose, all we ask is three things
# 
# - you cite Thoth Team as the source if you use the data;
# - you accept that you are solely responsible for how you use the data;
# - you do not sell this data to anyone, it is free!

# ## Install packages

# In[ ]:


get_ipython().system('pip install thoth-lab==0.2.3')


# ## Import packages

# In[ ]:


import json
import pandas as pd

from pathlib import Path
from thoth.lab.security import SecurityIndicators
security_indicators = SecurityIndicators()


# # Explore Thoth Solver Dataset

# ## Aggregate Data

# In[ ]:


security_indicator_bandit_repo_path = Path('/kaggle/input/thoth-security-dataset-v10/security/si-bandit/')
security_indicator_bandit_reports = []

for security_indicator_bandit_path in security_indicator_bandit_repo_path.iterdir():

    with open(security_indicator_bandit_path, 'r') as si_bandit_json:
        si_bandit_report = json.load(si_bandit_json)

    security_indicator_bandit_reports.append(si_bandit_report)

print(f"Number of solver reports is {len(security_indicator_bandit_reports)}")


# In[ ]:


security_indicator_cloc_repo_path = Path('/kaggle/input/thoth-security-dataset-v10/security/si-cloc/')
security_indicator_cloc_reports = []

for security_indicator_cloc_path in security_indicator_cloc_repo_path.iterdir():
    
    with open(security_indicator_cloc_path, 'r') as si_cloc_json:
        si_cloc_report = json.load(si_cloc_json)

    security_indicator_cloc_reports.append(si_cloc_report)

print(f"Number of solver reports is {len(security_indicator_cloc_reports)}")


# ## Explore one report

# ## security indicator bandit report

# In[ ]:


security_indicator_bandit_report = security_indicator_bandit_reports[0]


# Each Security Indicator bandit report is made by two main parts:
# * **metadata** where information about the analyzer (si-bandit) itself are stored (e.g version running, type of SI)
# * **result** where the inputs and outputs of the analyzer are actually collected 

# ### Security Indicator bandit report metadata

# All the metadata available for each SI bandit report are described below:
# * **analyzer**, name of the analyzer;
# * **analyzer_version**, analyzer version;
# * **arguments**, arguments for the analyzer;
#     * **package_name** Python package name used for the analysis;
#     * **package_version** Python package version used for the analysis;
#     * **package_index** Python package index used for the analysis;
#     * **no_pretty** output style;
#     * **output** where the output is stored locally;
# * **datetime**, when the analyzer report has been created;
# * **distribution**, operating system specific info;
# * **document_id**, unique ID of the SI bandit report;
# * **duration**, duration of the analyzer run for a certain Python Package;
# * **hostname**, Container name where the analyzer was run;
# * **os_release**, OS info collected from `/etc/os-release`;
# * **python**, Python Interpreter info;
# * **thoth_deployment_name**, Thoth architecture specific info;
# * **timestamp**;

# In[ ]:


metadata_df = security_indicators.create_si_bandit_metadata_dataframe(
    si_bandit_report=security_indicator_bandit_report
)
metadata_df


# ### Security Indicator bandit report result

# All the result in SI bandit report are described below:
# * **generated_at**;
# * **metrics**, metrics collected from each file analyzed (summary of results);
#     * **CONFIDENCE** confidence on the severity of the security,
#         * **CONFIDENCE.HIGH** 
#         * **CONFIDENCE.MEDIUM** 
#         * **CONFIDENCE.LOW**
#         * **CONFIDENCE.UNDEFINED** 
#     * **SEVERITY** level about security,
#         * **SEVERITY.HIGH** 
#         * **SEVERITY.MEDIUM** 
#         * **SEVERITY.LOW**
#         * **SEVERITY.UNDEFINED** 
#     * **loc** location of the line,
#     * **nosec**, if the line in the file has been silenced,
# * **results**, information about external packages installed on the environment;
# * **errors**, errors encountered running SI bandit (if any);

# In[ ]:


si_bandit_report_result_metrics_df = pd.DataFrame(security_indicator_bandit_report["result"]['metrics'])
si_bandit_report_result_metrics_df


# In[ ]:


filename = si_bandit_report_result_metrics_df.columns.values[0]
filename


# In[ ]:


si_bandit_report_result_metrics_df[filename]


# In[ ]:


si_bandit_report_result_results_df = pd.DataFrame(security_indicator_bandit_report["result"]['results'])
si_bandit_report_result_results_df


# In[ ]:


subset_df = si_bandit_report_result_results_df[si_bandit_report_result_results_df["filename"].values == filename]
subset_df


# In[ ]:


security_confidence_df, summary_files = security_indicators.create_security_confidence_dataframe(
    si_bandit_report=security_indicator_bandit_report
)
security_confidence_df


# In[ ]:


si_bandit_report_summary_df = security_indicators.produce_si_bandit_report_summary_dataframe(
    metadata_df=metadata_df,
    si_bandit_sec_conf_df=security_confidence_df,
    summary_files=summary_files
    
)
si_bandit_report_summary_df


# ## security indicator cloc report

# In[ ]:


security_indicator_cloc_report = security_indicator_cloc_reports[0]


# Each Security Indicator cloc report is made by two main parts:
# * **metadata** where information about the analyzer (si-cloc) itself are stored (e.g version running, type of SI)
# * **result** where the inputs and outputs of the analyzer are actually collected 

# ### Security Indicator cloc report metadata

# All the metadata available for each SI bandit report are described below:
# * **analyzer**, name of the analyzer;
# * **analyzer_version**, analyzer version;
# * **arguments**, arguments for the analyzer;
#     * **package_name** Python package name used for the analysis;
#     * **package_version** Python package version used for the analysis;
#     * **package_index** Python package index used for the analysis;
#     * **no_pretty** output style;
#     * **output** where the output is stored locally;
# * **datetime**, when the analyzer report has been created;
# * **distribution**, operating system specific info;
# * **document_id**, unique ID of the SI bandit report;
# * **duration**, duration of the analyzer run for a certain Python Package;
# * **hostname**, Container name where the analyzer was run;
# * **os_release**, OS info collected from `/etc/os-release`;
# * **python**, Python Interpreter info;
# * **thoth_deployment_name**, Thoth architecture specific info;
# * **timestamp**;

# In[ ]:


metadata_df = security_indicators.create_si_cloc_metadata_dataframe(
    si_cloc_report=security_indicator_cloc_report
)
metadata_df


# ### Security Indicator cloc report result

# All the result in SI bandit report are described below:
# * **header**;
#     * **cloc_url** cloc source,
#     * **cloc_version** cloc version,
#     * **elapsed_seconds** time to evaluate a package,
#     * **files_per_second** number of files analyzed per second,
#     * **lines_per_second** number of lines analyzed per second,
#     * **n_files** total number of files analyzed,
#     * **n_lines** total number of lines analyzed,
# * **{programming_language}**, different key for each programming_language (e.g. Python) files;
#     * **blank** number of blank lines,
#     * **code** number of code lines,
#     * **comment** number of code lines,
#     * **nFiles** number of files considered,
# * **SUM**, sum of all info from the different progamming languages outputs;
#     * **blank** number of blank lines,
#     * **code** number of code lines,
#     * **comment** number of code lines,
#     * **nFiles** number of files considered.

# In[ ]:


results_df = security_indicators.create_si_cloc_results_dataframe(si_cloc_report=security_indicator_cloc_report)
results_df


# In[ ]:


summary_df = security_indicators.produce_si_cloc_report_summary_dataframe(
    metadata_df=metadata_df,
    cloc_results_df=results_df
)
summary_df


# # Consider all bandit security reports

# In[ ]:


FILTER_FILES = ["tests/", "/test"]


# In[ ]:


final_df = security_indicators.create_si_bandit_final_dataframe(
    si_bandit_reports=security_indicator_bandit_reports,
    use_external_source_data=True,
    filters_files=FILTER_FILES
)


# In[ ]:


final_df.shape


# In[ ]:


final_df.drop_duplicates(
    subset=['analyzer_version', 'package_name', "package_version", "package_index"], inplace=True
)
final_df.shape


# In[ ]:


final_df.head()


# In[ ]:


final_df.describe()


# In[ ]:


from thoth.common.helpers import parse_datetime
filter_date = parse_datetime("2018-01-01T00:00:00.000")
filtered_df = final_df[final_df['release_date'] > filter_date]
filtered_df.head()


# In[ ]:


sorted_df = filtered_df.sort_values(by=['SEVERITY.HIGH__CONFIDENCE.HIGH', 'SEVERITY.HIGH__CONFIDENCE.MEDIUM', 'SEVERITY.MEDIUM__CONFIDENCE.HIGH'], ascending=False)
sorted_df.head()


# In[ ]:


security_indicators.create_vulnerabilities_plot(
    security_df=sorted_df.head(30)
)


# In[ ]:


package_summary_df = sorted_df[(sorted_df['package_name'] == "acme") & (sorted_df['package_index'] == "https://pypi.org/simple")]
package_summary_df = package_summary_df.sort_values(by=['release_date'], ascending=True)
security_indicators.create_package_releases_vulnerabilities_trend(
    package_summary_df=package_summary_df,
)


# In[ ]:


package_summary_df = sorted_df[(sorted_df['package_name'] == "aiida-core") & (sorted_df['package_index'] == "https://pypi.org/simple")]
package_summary_df = package_summary_df.sort_values(by=['release_date'], ascending=True)
security_indicators.create_package_releases_vulnerabilities_trend(
    package_summary_df=package_summary_df
)


# In[ ]:


package_summary_df = sorted_df[(sorted_df['package_name'] == "aiohttp") & (sorted_df['package_index'] == "https://pypi.org/simple")]
package_summary_df = package_summary_df.sort_values(by=['release_date'], ascending=True)
security_indicators.create_package_releases_vulnerabilities_trend(
    package_summary_df=package_summary_df
)


# # Consider all cloc security reports

# In[ ]:


si_cloc_total_df = security_indicators.create_si_cloc_final_dataframe(
    si_cloc_reports=security_indicator_cloc_reports
)


# In[ ]:


si_cloc_total_df.shape


# In[ ]:


si_cloc_total_df.drop_duplicates(
    subset=['analyzer_version', 'package_name', "package_version", "package_index"], inplace=True
)
si_cloc_total_df.shape


# In[ ]:


si_cloc_total_df.head()


# In[ ]:




