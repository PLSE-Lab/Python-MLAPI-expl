#!/usr/bin/env python
# coding: utf-8

# <h2 style="text-align:center;font-size:200%;;">CNC milling machine - Tool Wear Detection</h2>
# <h3  style="text-align:center;">Keywords : <span class="label label-success">IoT</span> <span class="label label-success">EDA</span> <span class="label label-success">Classification</span> <span class="label label-success">Causal Analysis</span> <span class="label label-success">FFT</span></h3>

# # Table of Contents<a id='top'></a>
# 
# >1. [Overview](#1.-Overview)  
# >   * [Project Detail](#Project-Detail)
# >   * [Goal of this notebook](#Goal-of-this-notebook)
# >1. [Import libraries](#2.-Import-libraries)
# >1. [Load the dataset](#3.-Load-the-dataset)
# >1. [Pre-processing](#4.-Pre-processing)
# >1. [EDA](#5.-EDA)  
# >    * [Basic Analysis](#Basic-Analysis)
# >    * [Univariate Analysis](#Univariate-Analysis)
# >    * [Multivariate Analysis](#Multivariate-Analysis)
# >    * [Frequency Analysis](#Frequency-Analysis)
# >1. [Modeling](#6.-Modeling)
# >    * [Feature Engineering](#Feature-Engineering)
# >    * [Case1 : Tool Condition](#Case1-:-Tool-Condition)
# >    * [Case2 : Machining Finalized](#Case2-:-Machining-Finalized)
# >    * [Case3 : Passed Visual Inspection](#Case3-:-Passed-Visual-Inspection)
# >1. [Conclusion](#7.-Conclusion)
# >1. [References](#8.-References)

# # 1. Overview

# ## Project Detail
# >In [this Dataset](https://www.kaggle.com/shasun/tool-wear-detection-in-cnc-mill), collected data in machining experiments are given. Machining data was collected from a CNC machine for variations of tool condition, feed rate, and clamping pressure. <br/>
# ><ul>
# >    <li>feed rate</li>
# >        <p>relative velocity of the cutting tool along the workpiece (mm/s)</p>
# >    <li>clamping pressure</li>
# >        <p>pressure used to hold the workpiece in the vise (bar)</p>
# ></ul>
# >In 18 machining experiments, time series data was collected with a sampling rate of 100 ms from the 4 motors in the CNC (X, Y, Z axes and spindle).<br/>
# >And output of each experiments includes tool condition (unworn and worn tools) and whether or not the tool passed visual inspection.<br/>
# >We can enjoy this dataset for tool wear detection or detection of inadequate clamping.
# 
# ## Goal of this notebook
# >* Practice EDA technique
# >* Practice visualising technique(especially using bokeh via holoviews)
# >* Practice feature enginieering technique
# >    * Lag features
# >    * Differential feature
# >* Practice modeling technique
# >    * LightGBM
# >* Causal analysis skill
# >* Frequency analysis skill
# >    * FFT

# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 2. Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
from scipy import signal
import lightgbm as lgb


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 3. Load the dataset

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


experiment_result = pd.read_csv("/kaggle/input/tool-wear-detection-in-cnc-mill/train.csv")
print(f'train.csv : {experiment_result.shape}')
experiment_result.head(3)


# In[ ]:


experiment_tmp = pd.read_csv("/kaggle/input/tool-wear-detection-in-cnc-mill/experiment_01.csv")
print(f'experiment_XX.csv : {experiment_tmp.shape}')
print(experiment_tmp.columns)
experiment_tmp.head(3)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 4. Pre-processing

# >NaN in passed_visual_inspection in experiment result means machining_finalized is no, which means machining process was not finished correctly and did not proceed to visual inspection process. So, we need to fill NaN with no.

# In[ ]:


experiment_result['passed_visual_inspection'] = experiment_result['passed_visual_inspection'].fillna('no')


# >adding each experiment settings and result to experiment time series data to make one total dataframe.

# In[ ]:


frames = []
for i in range(1,19):
    #load files
    exp_num = '0' + str(i) if i < 10 else str(i)
    frame = pd.read_csv(f"/kaggle/input/tool-wear-detection-in-cnc-mill/experiment_{exp_num}.csv")

    #load each experiment result row
    exp_result_row = experiment_result[experiment_result['No'] == i]
    frame['exp_num'] = i

    #add experiment settings to features
    frame['material'] = exp_result_row.iloc[0]['material']
    frame['feedrate'] = exp_result_row.iloc[0]['feedrate']
    frame['clamp_pressure'] = exp_result_row.iloc[0]['clamp_pressure']
    
    #add experiment result to features
    frame['tool_condition'] = exp_result_row.iloc[0]['tool_condition']
    frame['machining_finalized'] = exp_result_row.iloc[0]['machining_finalized']
    frame['passed_visual_inspection'] = exp_result_row.iloc[0]['passed_visual_inspection']

    frames.append(frame)

df = pd.concat(frames, ignore_index = True)
df.head(3)


# ### Label Normalization
# <div class="alert alert-success" role="alert">
# Count of 'Starting' and 'end' label in Machining_Process column is relatevely small.<br/>
# So we need to normalize these outlier labels into alternative label.<br/>
# <ul>
#     <li>Starting -> Prep</li>
#     <li>end -> End</li>
# </ul>
# </div>

# In[ ]:


df['Machining_Process'].value_counts().sort_index()


# In[ ]:


print(f"Count of Starting label in Machining_Process column is 1 only in experimet_{df[df['Machining_Process']=='Starting'].exp_num.value_counts().index[0]}")
print(f"Count of end label in Machining_Process column is 8 only in experimet_{df[df['Machining_Process']=='end'].exp_num.value_counts().index[0]}")


# In[ ]:


df.replace({'Machining_Process': {'Starting':'Prep','end':'End'}}, inplace=True)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 5. EDA
# ### Some points to focus on
# >* Machining Process Unique
# >* Mean value of velocity,voltage,feedrate(x,y,z)
# >* distribution of feedrate, clamp_pressure
# >* correlation of each features
# >* time series plot of each experiment
# >* difference of distribution for each output feature

# ## Basic Analysis

# ### Machine Inputs
# ><div class="alert alert-success" role="alert">
# >Observation of material is only 'wax'. So this column is <u>not applicable to modeling feature</u>.
# ></div>

# In[ ]:


feedrate = hv.Distribution(df['feedrate']).opts(title="Distribution of feedrate", color="green", xlabel="Feedrate", ylabel="Density")
clamp = hv.Distribution(df['clamp_pressure']).opts(title="Distribution of clamp pressure", color="green", xlabel="Pressure", ylabel="Density")
material = hv.Bars(df['material'].value_counts()).opts(title="Material Count", color="green", xlabel="Material", ylabel="Count")
(feedrate + clamp + material).opts(opts.Bars(width=300, height=300,tools=['hover'],show_grid=True)).cols(2)


# ### Machine Outputs
# ><div class="alert alert-success" role="alert">
# ><ul>
# >    <li>Count of 'Not Passed Visual Insepction' is larger than that of 'Not Finalized Machining'.
# >        <p>This means milling by machineries is not reliable enough, and <u>humans have some tricks or technique to detect failures, which machineries can't do.</u></p></li>
# >    <li>Count of 'Not Finalized Machining' with worn tool is almost same as that with unworn tool.</li>
# >    <li>Count of 'Not Passed Visual Insepction' with worn tool is relatively larger than that with unworn tool.
# >        <p>It is thought that machining with worn tool does not affect machinig process itself, <u>but humans can detect a slight difference affected by worn tools.</u></p></li>
# ></ul>
# ></div>

# In[ ]:


tool_df = np.round(df['tool_condition'].value_counts(normalize=True) * 100)
finalized_df = np.round(df['machining_finalized'].value_counts(normalize=True) * 100)
vis_passed_df = np.round(df['passed_visual_inspection'].value_counts(normalize=True) * 100)
tool_wear = hv.Bars(tool_df).opts(title="Tool Wear Count", color="green", xlabel="Worn/Unworn", ylabel="Percentage", yformatter='%d%%')
finalized = hv.Bars(finalized_df).opts(title="Finalized Count", color="green", xlabel="Yes/No", ylabel="Percentage", yformatter='%d%%')
vis_inspection = hv.Bars(vis_passed_df).opts(title="Visual Inspection Passed Count", color="green", xlabel="Yes/No", ylabel="Percentage", yformatter='%d%%')
(tool_wear + finalized + vis_inspection).opts(opts.Bars(width=300, height=300,tools=['hover'],show_grid=True)).cols(2)


# In[ ]:


finalized_df_worn = np.round(df[df['tool_condition']=='worn']['machining_finalized'].value_counts(normalize=True) * 100)
finalized_df_unworn = np.round(df[df['tool_condition']=='unworn']['machining_finalized'].value_counts(normalize=True) * 100)
vis_passed_df_worn = np.round(df[df['tool_condition']=='worn']['passed_visual_inspection'].value_counts(normalize=True) * 100)
vis_passed_df_unworn = np.round(df[df['tool_condition']=='unworn']['passed_visual_inspection'].value_counts(normalize=True) * 100)
finalized_worn = hv.Bars(finalized_df_worn).opts(title="[WORN] Finalized Count", color="orange", xlabel="Yes/No", ylabel="Percentage", yformatter='%d%%')            * hv.Text('yes', 15, f"{np.round(finalized_df_worn['yes']/sum(finalized_df_worn)*100)}%")            * hv.Text('no', 15, f"{np.round(finalized_df_worn['no']/sum(finalized_df_worn)*100)}%")
finalized_unworn = hv.Bars(finalized_df_unworn).opts(title="[UNWORN] Finalized Count", color="orange", xlabel="Yes/No", ylabel="Percentage", yformatter='%d%%')            * hv.Text('yes', 15, f"{np.round(finalized_df_unworn['yes']/sum(finalized_df_unworn)*100)}%")            * hv.Text('no', 15, f"{np.round(finalized_df_unworn['no']/sum(finalized_df_unworn)*100)}%")
vis_inspection_worn = hv.Bars(vis_passed_df_worn).opts(title="[WORN] Visual Inspection Passed Count", color="green", xlabel="Yes/No", ylabel="Percentage", yformatter='%d%%')            * hv.Text('yes', 45, f"{np.round(vis_passed_df_worn['yes']/sum(vis_passed_df_worn)*100)}%")            * hv.Text('no', 45, f"{np.round(vis_passed_df_worn['no']/sum(vis_passed_df_worn)*100)}%")
vis_inspection_unworn = hv.Bars(vis_passed_df_unworn).opts(title="[UNWORN] Visual Inspection Passed Count", color="green", xlabel="Yes/No", ylabel="Percentage", yformatter='%d%%')            * hv.Text('yes', 15, f"{np.round(vis_passed_df_unworn['yes']/sum(vis_passed_df_unworn)*100)}%")            * hv.Text('no', 15, f"{np.round(vis_passed_df_unworn['no']/sum(vis_passed_df_unworn)*100)}%")
((finalized_worn + finalized_unworn) + (vis_inspection_worn + vis_inspection_unworn)).opts(opts.Bars(width=400, height=300,tools=['hover'],show_grid=True)).cols(2)


# In[ ]:


worn_fin_vis = pd.concat([finalized_df_worn, vis_passed_df_worn], axis=1,sort=True).rename(columns={'machining_finalized':'[WORN] Finalized', 'passed_visual_inspection':'[WORN] Visual Inspection Passed'})
worn_fin_vis = pd.melt(worn_fin_vis.reset_index(), ['index']).rename(columns={'index':'Yes/No', 'variable':'Outputs'})
hv.Bars(worn_fin_vis, ['Outputs','Yes/No'], 'value').opts(opts.Bars(title="Machining Finalized and Passed Visual Inspection by Worn Tool Count", width=700, height=400,tools=['hover'],                                                                show_grid=True, ylabel="Percentage", yformatter='%d%%'))


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ## Univariate Analysis

# ### Machining Process

# In[ ]:


hv.Bars(df['Machining_Process'].value_counts()).opts(title="Machining Process Count", color="red", xlabel="Machining Processes", ylabel="Count")                                        .opts(opts.Bars(width=500, height=300,tools=['hover'],xrotation=45,show_grid=True))


# > plot function to output all experiment time-series

# In[ ]:


def plot_ts(col, color='red', yformat='%d%%'):
    v_list = []
    for i in range(1,19):
        v = hv.Curve(df[df['exp_num']==i].reset_index()[col]).opts(title=f"{col} in  experiment {i}", xlabel="Time", ylabel=f"{col}", yformatter=yformat)                                                          .opts(width=300, height=150,tools=['hover'],show_grid=True,fontsize=8, color=color)
        v_list.append(v)
    return (v_list[0] + v_list[1] + v_list[2] + v_list[3] + v_list[4] + v_list[5] + v_list[6] + v_list[7] + v_list[8] + v_list[9] + v_list[10] + v_list[11] + v_list[12]            + v_list[13] + v_list[14] + v_list[15] + v_list[16] + v_list[17]).opts(shared_axes=False).cols(6)


# ### Velocity
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>experiments which had worn tools are not distinctive.</li>
# ><li>experiments which did not finalize machining and pass visual inspection are seem to have certain patterns in <u>X/Y/S</u> axes.
# >    <p>wide range of change<br/> 
# >       low frequecy of change</p></li>
# ></ul>
# ></div>

# In[ ]:


plot_ts('X1_ActualVelocity', color='red', yformat='%d mm/s')


# In[ ]:


plot_ts('Y1_ActualVelocity', color='orange', yformat='%d mm/s')


# In[ ]:


plot_ts('Z1_ActualVelocity', color='green', yformat='%d mm/s')


# In[ ]:


plot_ts('S1_ActualVelocity', color='blue', yformat='%d mm/s')


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Current
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>experiments which had worn tools are not distinctive.</li>
# ><li>experiments which did not finalize machining and pass visual inspection are seem to have low frequecy of change in <u>X/Y/S</u> axes.</li>
# ></ul>
# ></div>

# In[ ]:


plot_ts('X1_CurrentFeedback', color='red', yformat='%d A')


# In[ ]:


plot_ts('Y1_CurrentFeedback', color='orange', yformat='%d A')


# In[ ]:


plot_ts('Z1_CurrentFeedback', color='green', yformat='%d A')


# In[ ]:


plot_ts('S1_CurrentFeedback', color='blue', yformat='%d A')


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Voltage
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>experiments which had worn tools are not distinctive.</li>
# ><li>experiments which did not finalize machining and pass visual inspection are seem to have low frequecy of change in <u>X/Y/S</u> axes.</li>
# ></ul>
# ></div>

# In[ ]:


plot_ts('X1_DCBusVoltage', color='red', yformat='%.1f V')


# In[ ]:


plot_ts('Y1_DCBusVoltage', color='orange', yformat='%.1f V')


# In[ ]:


plot_ts('Z1_DCBusVoltage', color='green', yformat='%.1f V')


# In[ ]:


plot_ts('S1_DCBusVoltage', color='blue', yformat='%.1f V')


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ## Multivariate Analysis

# ### Feedrate / Clamp Pressure
# ><div class="alert alert-success" role="alert">
# >Tool Condition
# ><ul>
# ><li>distributions of <b>clamp pressure</b> with unworn/worn tool are almost same.</li>
# ><li>distribution of <b>feedrate</b> with worn tool is <u>more wider</u> than that with unworn tool.</li>
# ></ul>
# >Machining Finalized
# ><ul>
# ><li>distribution of <b>clamp pressure</b> in machining not finalized experiment has <u>lower kurtosis</u>.</li>
# ><li>distribution of <b>feedrate</b> in machining not finalized experiment has <u>higher kurtosis</u>.</li>
# ></ul>
# ></div>

# In[ ]:


g = sns.pairplot(df, hue='tool_condition', vars=["feedrate","clamp_pressure"])
g.fig.suptitle("Tool Condition - feedrate/clamp pressure", y=1.1, fontsize=20)
g.fig.set_figheight(6)
g.fig.set_figwidth(9)
plt.show()


# In[ ]:


g = sns.pairplot(df, hue='machining_finalized', vars=["feedrate","clamp_pressure"])
g.fig.suptitle("Machining Finalized - feedrate/clamp pressure", y=1.1, fontsize=20)
g.fig.set_figheight(6)
g.fig.set_figwidth(9)
plt.show()


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Velocity
# ><div class="alert alert-success" role="alert">
# >Machining Finalized
# ><ul>
# ><li>distribution of <b>velocity in S axis</b> in machining not finalized experiment has <u>lower kurtosis</u>.</li>
# ></ul>
# ></div>

# In[ ]:


g = sns.pairplot(df, hue='tool_condition', vars=['X1_ActualVelocity','Y1_ActualVelocity','Z1_ActualVelocity','S1_ActualVelocity'])
g.fig.suptitle("Tool Condition - velocity", y=1.1, fontsize=20)
g.fig.set_figheight(6)
g.fig.set_figwidth(9)
g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
plt.show()


# In[ ]:


g = sns.pairplot(df, hue='machining_finalized', vars=['X1_ActualVelocity','Y1_ActualVelocity','Z1_ActualVelocity','S1_ActualVelocity'])
g.fig.suptitle("Machining Finalized - velocity", y=1.1, fontsize=20)
g.fig.set_figheight(6)
g.fig.set_figwidth(9)
g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
plt.show()


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Current
# ><div class="alert alert-success" role="alert">
# >Machining Finalized
# ><ul>
# ><li>distribution of <b>current in X/Y/S axes</b> in machining not finalized experiment has <u>higher kurtosis</u>.</li>
# ></ul>
# ></div>

# In[ ]:


np.seterr(divide='ignore', invalid='ignore')
g = sns.pairplot(df, hue='tool_condition', vars=['X1_CurrentFeedback','Y1_CurrentFeedback','Z1_CurrentFeedback','S1_CurrentFeedback'])
g.fig.suptitle("Tool Condition - Current", y=1.1, fontsize=20)
g.fig.set_figheight(6)
g.fig.set_figwidth(9)
g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
plt.show()


# In[ ]:


g = sns.pairplot(df, hue='machining_finalized', vars=['X1_CurrentFeedback','Y1_CurrentFeedback','Z1_CurrentFeedback','S1_CurrentFeedback'])
g.fig.suptitle("Machining Finalized - Current", y=1.1, fontsize=20)
g.fig.set_figheight(6)
g.fig.set_figwidth(9)
g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
plt.show()


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Voltage
# ><div class="alert alert-success" role="alert">
# >Machining Finalized
# ><ul>
# ><li>distribution of <b>voltage in X/Y/S axes</b> in machining not finalized experiment has <u>higher skewness</u>.</li>
# ></ul>
# ></div>

# In[ ]:


g = sns.pairplot(df, hue='tool_condition', vars=['X1_DCBusVoltage','Y1_DCBusVoltage','Z1_DCBusVoltage','S1_DCBusVoltage'])
g.fig.suptitle("Tool Condition - Voltage", y=1.1, fontsize=20)
g.fig.set_figheight(6)
g.fig.set_figwidth(9)
g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
plt.show()


# In[ ]:


g = sns.pairplot(df, hue='machining_finalized', vars=['X1_DCBusVoltage','Y1_DCBusVoltage','Z1_DCBusVoltage','S1_DCBusVoltage'])
g.fig.suptitle("Machining Finalized - Voltage", y=1.1, fontsize=20)
g.fig.set_figheight(6)
g.fig.set_figwidth(9)
g.fig.get_children()[-1].set_bbox_to_anchor((1.1, 0.5, 0, 0))
plt.show()


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ## Frequency Analysis
# ><div class="alert alert-success" role="alert">
# ><ul>
# ><li>experiments which did not finalize machining seem to have high amplitude in certain frequencies.
# ><li>experiments which did not pass visual inspection seem to have high amplitude in certain frequencies too.
# >    <p>this is not so obvious in univariate plot above.<br/>
# >       these discoveries observed in frequency analysis can be <u>related to why humans can detect machining failures</u>.</p></li>
# ></ul>
# ></div>

# > plot function to output all experiment fft result

# In[ ]:


def plot_fft(col, color='red', peak_thr=1):
    v_list = []
    dt = 0.1 # experiment data was collected per 100ms(0.1sec)
    for i in range(1,19):
        f = df[df['exp_num']==i].reset_index()[col]
        N = len(f)
        t = np.arange(0, N*dt, dt)
        freq = np.linspace(0, 1.0/dt, N)
        F = np.fft.fft(f)
        F_abs = np.abs(F) / (N/2) 
        F_abs[0] = F_abs[0] / 2
        
        maximal_idx = signal.argrelmax(F_abs, order=1)[0] 
        peak_cut = peak_thr
        maximal_idx = maximal_idx[(F_abs[maximal_idx] > peak_cut) & (maximal_idx <= N/2)]
        
        v = hv.Curve((freq[:int(N/2)+1], F_abs[:int(N/2)+1])).opts(title=f"{col} in  experiment {i}", xlabel="Frequency(Hz)", ylabel=f"Amplitude")                                                          .opts(width=300, height=150,tools=['hover'],show_grid=True,fontsize=8, color=color)            * hv.Scatter((freq[maximal_idx], F_abs[maximal_idx])).opts(color='lime', size=5)
        
        v_list.append(v)
    return (v_list[0] + v_list[1] + v_list[2] + v_list[3] + v_list[4] + v_list[5] + v_list[6] + v_list[7] + v_list[8] + v_list[9] + v_list[10] + v_list[11] + v_list[12]            + v_list[13] + v_list[14] + v_list[15] + v_list[16] + v_list[17]).opts(shared_axes=False).cols(6)


# ### Velocity

# In[ ]:


plot_fft('X1_ActualVelocity', color='red', peak_thr=3)


# In[ ]:


plot_fft('Y1_ActualVelocity', color='orange', peak_thr=3)


# In[ ]:


plot_fft('Z1_ActualVelocity', color='green', peak_thr=3)


# In[ ]:


plot_fft('S1_ActualVelocity', color='blue', peak_thr=9)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Current

# In[ ]:


plot_fft('X1_CurrentFeedback', color='red', peak_thr=1.2)


# In[ ]:


plot_fft('Y1_CurrentFeedback', color='orange', peak_thr=1.2)


# In[ ]:


plot_fft('Z1_CurrentFeedback', color='green', peak_thr=3)


# In[ ]:


plot_fft('X1_CurrentFeedback', color='blue', peak_thr=1.2)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ### Volatage

# In[ ]:


plot_fft('X1_DCBusVoltage', color='red', peak_thr=0.015)


# In[ ]:


plot_fft('Y1_DCBusVoltage', color='orange', peak_thr=0.02)


# In[ ]:


plot_fft('Z1_DCBusVoltage', color='green', peak_thr=3)


# In[ ]:


plot_fft('S1_DCBusVoltage', color='blue', peak_thr=0.15)


# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 6. Modeling
# * Target variables :
#     * tool_condition
#     * machining_finalized
#     * passed_visual_inspection
# * Through some experiment cases, I examined which features are more important to make robust prediction models.
#     * Especially I wanted to know which CNC imformation, for example tool position or current, is most important to detect machine failure.

# ## Feature Engineering

# ### Differential Features
# <div class="alert alert-success" role="alert">
# <ul>
# <li>differences between <b>Actual</b> and <b>Command</b> position can indicate <u>a sign of maching failure</u>.</li>
# </ul>
# </div>

# In[ ]:


for ax in ['X','Y','Z','S']:
    df[f'{ax}1_Position_Diff'] = abs(df[f'{ax}1_CommandPosition']-df[f'{ax}1_ActualPosition'])
    df[f'{ax}1_Velocity_Diff'] = abs(df[f'{ax}1_CommandVelocity']-df[f'{ax}1_ActualVelocity'])
    df[f'{ax}1_Acceleration_Diff'] = abs(df[f'{ax}1_CommandAcceleration']-df[f'{ax}1_ActualAcceleration'])


# ### FFT Features
# <div class="alert alert-success" role="alert">
# <ul>
# <li>it is thought that <u>highest frequency and amplitude</u> can be good information to detect machinig failures.</li>
# </ul>
# </div>

# In[ ]:


for col in ['ActualPosition','ActualVelocity','ActualAcceleration','CurrentFeedback','DCBusVoltage','OutputCurrent','OutputVoltage','OutputPower']:
    dt = 0.1
    for i in range(1,19):
        for ax in ['X','Y','Z','S']:
            try:
                f = df[df['exp_num']==i].reset_index()[f'{ax}1_{col}']
            except:
                continue
                
            N = len(f)
            t = np.arange(0, N*dt, dt)
            freq = np.linspace(0, 1.0/dt, N)
            F = np.fft.fft(f)
            F_abs = np.abs(F) / (N/2) 
            F_abs[0] = F_abs[0] / 2
            maximal_idx = signal.argrelmax(F_abs, order=1)[0]

            high_amp = np.max(F_abs[maximal_idx]) if len(maximal_idx) > 0 else 0
            high_freq = freq[maximal_idx][np.argmax(F_abs[maximal_idx])] if len(maximal_idx) > 0 else 0

            df.loc[df['exp_num']==i,f'{ax}1_{col}_High_Amp'] = high_amp
            df.loc[df['exp_num']==i,f'{ax}1_{col}_High_Freq'] = high_freq
            df.loc[df['exp_num']==i,f'{ax}1_{col}_High_Amp_Freq'] = high_amp * high_freq


# >label encoding & drop unnecessary columns

# In[ ]:


feature_df = df.copy()
feature_df['Machining_Process'] = LabelEncoder().fit_transform(feature_df['Machining_Process']).astype(np.int8)
feature_df['tool_condition'] = LabelEncoder().fit_transform(feature_df['tool_condition']).astype(np.int8)
feature_df['machining_finalized'] = LabelEncoder().fit_transform(feature_df['machining_finalized']).astype(np.int8)
feature_df['passed_visual_inspection'] = LabelEncoder().fit_transform(feature_df['passed_visual_inspection']).astype(np.int8)
feature_df.drop(['material','exp_num'], axis=1, inplace=True)
feature_df.head(3)


# ## Case1 : Tool Condition

# In[ ]:


y_series = feature_df['tool_condition']
x_df = feature_df.drop(['tool_condition','machining_finalized','passed_visual_inspection'], axis=1) 
X_train, X_valid, Y_train, Y_valid = train_test_split(x_df, y_series, test_size=0.2, random_state=0, stratify=y_series)

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_valid = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)


# In[ ]:


params = {
    'task' : 'train',
    'boosting' : 'gbdt',
    'objective': 'binary',
    'metric': 'l2',
    'num_leaves': 200,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 5
}
gbm_tool_wear = lgb.train(params,
            lgb_train,
            num_boost_round=100,
            valid_sets=lgb_valid,
            early_stopping_rounds=100)


# In[ ]:


feature_imp = pd.DataFrame()
feature_imp['feature'] = gbm_tool_wear.feature_name()
feature_imp['importance'] = gbm_tool_wear.feature_importance()
hv.Bars(feature_imp.sort_values(by='importance', ascending=False)[0:31][::-1]).opts(title="Feature Importance", color="purple", xlabel="Features", ylabel="Importance", invert_axes=True)                            .opts(opts.Bars(width=700, height=700, tools=['hover'], show_grid=True))


# In[ ]:


t = lgb.plot_tree(gbm_tool_wear, figsize=(20, 20), precision=3, tree_index=1, show_info=['split_gain'])
plt.title('Visulalization of Tree in Tool Condition')
plt.show()


# ><div class="alert alert-success" role="alert">
# >Feature Importrance top-20
# ><ol>
# ><li><font color='red'>X1</font>_Actual<font color='magenta'>Velocity</font></li>
# ><li><font color='red'>X1</font>_CurrentFeedback</li>
# ><li><font color='red'>X1</font>_ActualPosition</li>
# ><li><font color='red'>X1</font>_ActualAcceleration</li>
# ><li><font color='red'>X1</font>_DCBus<font color='orange'>Voltage</font></li>
# ><li><font color='red'>X1</font>_Command<font color='magenta'>Velocity</font></li>
# ><li><font color='blue'>S1</font>_Actual<font color='magenta'>Velocity</font>_<font color='lime'>High_Freq</font></li>
# ><li><font color='red'>X1</font>_Output<font color='orange'>Voltage</font></li>
# ><li>Y1_CurrentFeedback</li>
# ><li><font color='red'>X1</font>_CommandAcceleration</li>
# ><li><font color='red'>X1</font>_OutputPower</li>
# ><li><font color='red'>X1</font>_Command<font color='magenta'>Velocity</font></li>
# ><li><font color='blue'>S1</font>_Actual<font color='magenta'>Velocity</font>_<font color='lime'>High_Freq</font></li>
# ><li><font color='red'>X1</font>_DCBus<font color='orange'>Voltage</font>_<font color='lime'>High_Freq</font></li>   
# ><li><font color='red'>X1</font>_Command<font color='magenta'>Velocity</font></li>
# ><li><font color='blue'>S1</font>_Actual<font color='magenta'>Velocity</font>_<font color='lime'>High_Freq</font></li>
# ><li><font color='red'>X1</font>_ActualPosition_<font color='lime'>High_Amp_Freq</font></li>
# ><li><font color='red'>X1</font>_ActualPosition_<font color='lime'>High_Freq</font></li>
# ><li>Y1_DCBus<font color='orange'>Voltage</font></li>
# ><li><font color='blue'>S1</font>_ActualAcceleration</li>
# ></ol>
# ></div>

# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ## Case2 : Machining Finalized

# In[ ]:


y_series = feature_df['machining_finalized']
x_df = feature_df.drop(['tool_condition','machining_finalized','passed_visual_inspection'], axis=1) 
X_train, X_valid, Y_train, Y_valid = train_test_split(x_df, y_series, test_size=0.2, random_state=0, stratify=y_series)

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_valid = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)


# In[ ]:


params = {
    'task' : 'train',
    'boosting' : 'gbdt',
    'objective': 'binary',
    'metric': 'l2',
    'num_leaves': 200,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 5
}
gbm_machining_finalized = lgb.train(params,
            lgb_train,
            num_boost_round=100,
            valid_sets=lgb_valid,
            early_stopping_rounds=100)


# In[ ]:


feature_imp = pd.DataFrame()
feature_imp['feature'] = gbm_machining_finalized.feature_name()
feature_imp['importance'] = gbm_machining_finalized.feature_importance()
hv.Bars(feature_imp.sort_values(by='importance', ascending=False)[0:31][::-1]).opts(title="Feature Importance", color="purple", xlabel="Features", ylabel="Importance", invert_axes=True)                            .opts(opts.Bars(width=700, height=700, tools=['hover'], show_grid=True))


# In[ ]:


t = lgb.plot_tree(gbm_machining_finalized, figsize=(20, 20), precision=3, tree_index=1, show_info=['split_gain'])
plt.title('Visulalization of Tree in Machining Finalized')
plt.show()


# ><div class="alert alert-success" role="alert">
# >Feature Importrance top-20
# ><ol>
# ><li><font color='red'>X1</font>_ActualVelocity</li>
# ><li><font color='red'>X1</font>_CurrentFeedback</li>
# ><li><font color='red'>X1</font>_Actual<font color='magenta'>Position</font></li>
# ><li><font color='red'>X1</font>_ActualAcceleration</li>
# ><li><font color='red'>X1</font>_DCBus<font color='orange'>Voltage</font></li>
# ><li><font color='red'>X1</font>_Output<font color='orange'>Voltage</font></li>
# ><li>S1_Actual<font color='magenta'>Position</font>_<font color='lime'>High_Amp</font></li>
# ><li><font color='red'>X1</font>_OutputPower</li>
# ><li><font color='blue'>Y1</font>_Output<font color='orange'>Voltage</font></li>
# ><li><font color='blue'>Y1</font>_CurrentFeedback</li>
# ><li><font color='blue'>Y1</font>_ActualVelocity</li>
# ><li><font color='red'>X1</font>_CommandVelocity</li>
# ><li><font color='blue'>Y1</font>_DCBus<font color='orange'>Voltage</font></li>
# ><li><font color='red'>X1</font>_CommandAcceleration</li>   
# ><li>S1_Actual<font color='magenta'>Position</font></li>
# ><li><font color='blue'>Y1</font>_Actual<font color='magenta'>Position</font></li>
# ><li><font color='red'>X1</font>_Command<font color='magenta'>Position</font></li>
# ><li>S1_CurrentFeedback</li>
# ><li>S1_ActualAcceleration</li>
# ><li><font color='blue'>Y1</font>_OutputPower</li>
# ></ol>
# ></div>

# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# ## Case3 : Passed Visual Inspection

# In[ ]:


y_series = feature_df['passed_visual_inspection']
x_df = feature_df.drop(['tool_condition','machining_finalized','passed_visual_inspection'], axis=1) 
X_train, X_valid, Y_train, Y_valid = train_test_split(x_df, y_series, test_size=0.2, random_state=0, stratify=y_series)

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_valid = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)


# In[ ]:


params = {
    'task' : 'train',
    'boosting' : 'gbdt',
    'objective': 'binary',
    'metric': 'l2',
    'num_leaves': 200,
    'feature_fraction': 1.0,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 5
}
gbm_passed_vis_inspection = lgb.train(params,
            lgb_train,
            num_boost_round=100,
            valid_sets=lgb_valid,
            early_stopping_rounds=100)


# In[ ]:


feature_imp = pd.DataFrame()
feature_imp['feature'] = gbm_passed_vis_inspection.feature_name()
feature_imp['importance'] = gbm_passed_vis_inspection.feature_importance()
hv.Bars(feature_imp.sort_values(by='importance', ascending=False)[0:31][::-1]).opts(title="Feature Importance", color="purple", xlabel="Features", ylabel="Importance", invert_axes=True)                            .opts(opts.Bars(width=700, height=700, tools=['hover'], show_grid=True))


# In[ ]:


t = lgb.plot_tree(gbm_passed_vis_inspection, figsize=(20, 20), precision=3, tree_index=1, show_info=['split_gain'])
plt.title('Visulalization of Tree in Passed Visual Inspection')
plt.show()


# ><div class="alert alert-success" role="alert">
# >Feature Importrance top-20
# ><ol>
# ><li><font color='red'>X1</font>_Actual<font color='magenta'>Velocity</font></li>
# ><li><font color='red'>X1</font>_CurrentFeedback</li>
# ><li><font color='red'>X1</font>_ActualPosition</li>
# ><li><font color='red'>X1</font>_DCBus<font color='orange'>Voltage</font></li>
# ><li><font color='red'>X1</font>_ActualAcceleration</li>
# ><li><font color='red'>X1</font>_Output<font color='orange'>Voltage</font></li>
# ><li><font color='red'>X1</font>_Actual<font color='magenta'>Velocity</font>_<font color='lime'>High_Amp</font></li>
# ><li><font color='blue'>Y1</font>_CurrentFeedback</li>
# ><li>Z1_ActualPosition_<font color='lime'>High_Amp</font></li>
# ><li><font color='red'>X1</font>_CommandAcceleration</li>
# ><li><font color='blue'>Y1</font>_Output<font color='orange'>Voltage</font></li>
# ><li><font color='red'>X1</font>_Command<font color='magenta'>Velocity</font></li>
# ><li><font color='red'>X1</font>_OutputPower</li>
# ><li><font color='blue'>Y1</font>_DCBus<font color='orange'>Voltage</font></li>
# ><li>S1_ActualAcceleration</li>
# ><li>S1_ActualPosition</li>
# ><li><font color='blue'>Y1</font>_Actual<font color='magenta'>Velocity</font></li>
# ><li>S1_OutputPower</li>
# ><li><font color='blue'>Y1</font>_Command<font color='magenta'>Velocity</font></li>
# ><li><font color='blue'>Y1</font>_OutputPower</li>
# ></ol>
# ></div>

# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 7. Conclusion
# ><div class="alert alert-success" role="alert">
# ><h4>Tool Wear</h4>
# ><ul>
# >    <li><b>X-axis</b> and <b>S-axis</b> data have a great influence to tool wears, and the movement of X-axis and S-axis can be a bad affect to tool wear.</li>
# >    <li><b>Velocity</b> and <b>voltage</b> have a large influence in the collected data.</li>
# >    <li>A certain number of features created by <b>FFT</b> are also included in the top of feature importances, and it is considered that <u>tool wear increased the amplitude in the specific frequency range</u>.</li>
# ></ul>   
# ><br/>
# ><h4>Machining Finalized</h4>
# ><ul>
# >    <li><b>X-axis</b> and <b>Y-axis</b> data have a great influence to tool wears, and the movement of X-axis and Y-axis can be a bad affect to machining process.</li>
# >    <li><b>Position</b> and <b>voltage</b> have a large influence in the collected data.</li>
# ></ul>    
# ><br/>  
# ><h4>Passed Visual Inspection</h4>
# ><ul>
# >    <li><b>X-axis</b> and <b>Y-axis</b> data have a great influence to tool wears, and the movement of X-axis and Y-axis may be a bad affect to machining process.</li>
# >    <li><b>Velocity</b> and <b>voltage</b> have a large influence in the collected data.</li>
# >    <li>The importance of features created by <b>FFT</b> increased compared to Machining Finalized case.
#         <p>It is considered that the influence of tool weariness to the specific frequency range is <u>related to an external processing result that can be detected by humans</u>.</p></li>
# ></ul>
# ><br/>
# ><h4>Others</h4>
# ><ul>
# >    <li>Through all the experiment, there is almost no importance in Z-axis data.</li>
# >    <li>The <b>differential features</b> do not have a large influence on the detection of anomalies throughout, and it is thought that <u>the difference between the command and the actual position was negligibly small for tool wear and machining process</u> in most cases.</li>
# ></ul>
# ><br/>  
# ><p>In order to proceed to more detailed analysis, it is necessary to deepen the understanding of the operating principles of processing machines.</p>
# ><br/> 
# ></div>

# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>

# # 8. References
# >* **Seaborn Tricks**  
# >https://catherineh.github.io/programming/2016/05/24/seaborn-pairgrid-tips
# >* **Numpy FFT Tricks**  
# >https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
# >* **LightGBM Parameter List**  
# >https://lightgbm.readthedocs.io/en/latest/Parameters.html
# >* **Introduction of CNC anomaly detection**  
# >https://medium.com/machinemetrics-techblog/using-pca-and-clustering-to-detect-machine-anomalies-part-1-ba89f6a6a8cd

# <a href="#top" class="btn btn-success btn-sm active" role="button" aria-pressed="true" style="color:white;">Table of Contents</a>
