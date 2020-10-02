#!/usr/bin/env python
# coding: utf-8

# # CORD-19 using Anserini + BioASQ by [0699](https://www.kaggle.com/b0699f), [melaniebalaz](https://www.kaggle.com/melaniebalaz), [kiqo](https://www.kaggle.com/kiqokiqo/)
# 
# ## Based on Anserini+BERT-SQuAD for Semantic Corpus Sea
# These results are mainly based on version 54 of [Anserini+BERT-SQuAD for Semantic Corpus Sea](https://www.kaggle.com/dirktheeng/anserini-bert-squad-for-semantic-corpus-search) with the main difference being another model for computing the span scores of the Question Answering. 
# Instead of using the regular BERT model trained on SQuAD for Question Answering, we use a BERT model which was trained on biomedical texts and then trained on SQuAD for Question Answering.
# 
# More exactly, we use the pre-trained language model for biomedical question answering ([BioASQ](https://github.com/dmis-lab/bioasq-biobert)), which is trained using the [BioBERT](https://github.com/dmis-lab/biobert) embeddings.
# 
# ## Acknowledgements
# Thanks to [dirktheeng](https://www.kaggle.com/dirktheeng/) for making his implementation of the Yang et al combined Anserini and Bert-SQuAD methodology into a publicly available web tool. This gave us a good start point for improving the already well-working Question Answering tool for the COVID challenge.
# 
# In addition, credit for setting up the Anserini + BERT framework goes to [Jimmy Lin](https://cs.uwaterloo.ca/~jimmylin/), Edwin Zhang, and Nikhil Gupta at the University of Waterloo.
# Thanks also to [Wonjin Yoon et al.](https://arxiv.org/abs/1909.08229) for creating and proving the pre-trained BIOASQ model. 
# 
# ## Findings & Results
# 
#  * In general, the Question Answering results differ only slightly. It is hard for non-experts to determine which results answer a question more accurately. The resulting abstractive summary can be used to see the slight differences on a quick glance.
#  * The exact same answer (including span section) is often returned. The resulting confidence is then also the same, as this confidence is computed in both cases using the Universal Sentence Encoder.
#  * In other cases, either a complete new result is returned or a result where only the span sections differ. These cases are the ones where the BioASQ model answered a question differently than the BERT-SQuAD. 
# 
# 
# ### Example Query - How long is the incubation period for the virus?
# 
# #### BioASQ Abstractive Summary: 
# The incubation period for covid-19 is between 2-10 days, according to the world health organization. to avoid the risk of virus spread, all potentially exposed subjects are required to be isolated for 14 days, which is the longest predicted incubation time. <font color='red'>screening for the virus infection should be carried out for all patients, both preoperatively and postoperatively.</font>
# 
# #### BERT-SQuAD Abstractive Summary: 
# The incubation period for covid-19 is between 2-10 days, according to the world health organization  <font color='red'>(who). source of infection including the patients, asymptomatic carrier and patients in the incubation period are contagious.</font> to avoid the risk of virus spread, all potentially exposed subjects are required to be isolated for 14 days, which is the longest predicted incubation time.
# 
# <table border="1" class="dataframe">
#   <thead> 
#     <tr style="text-align: right;">
#       <th></th>
#       <th style="width: 30%">BERT-SQuAD Answer with Highlights</th>
#       <th style="width: 20%">BERT-SQuAD Title/Link</th>
#       <th style="width: 30%">BioASQ Answer with Highlights</th>
#       <th style="width: 20%">BioASQ Title/Link</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> to avoid the risk of virus spread, all potentially exposed subjects are required to be isolated  <font color="red">for 14 days</font> , which is the longest predicted incubation time.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1002/jmv.25708" target="_blank">Does SARS-CoV-2 has a longer incubation period than SARS and MERS?</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> to avoid the risk of virus spread, all potentially exposed subjects are required to be isolated  <font color="red">for 14 days</font> , which is the longest predicted incubation time.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1002/jmv.25708" target="_blank">Does SARS-CoV-2 has a longer incubation period than SARS and MERS?</a></td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> given  <font color="red">the long and uncertain incubation period of covid-19</font> , screening for the virus infection should be carried out for all patients, both preoperatively and postoperatively.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1634/theoncologist.2020-0157" target="_blank">Clinical Characteristics of COVID-19 After Gynecologic Oncology Surgery in Three Women: A Retrospective Review of Medical Records.</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> given  <font color="red">the long and uncertain</font>  incubation period of covid-19, screening for the virus infection should be carried out for all patients, both preoperatively and postoperatively.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1634/theoncologist.2020-0157" target="_blank">Clinical Characteristics of COVID-19 After Gynecologic Oncology Surgery in Three Women: A Retrospective Review of Medical Records.</a></td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> the incubation period  <font color="red">is 2-9 days</font>  and the majority of cases are mild.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.4102/safp.v62i1.5115" target="_blank">Primary care management of the coronavirus (COVID-19).</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> the incubation period  <font color="red">is 2-9 days</font>  and the majority of cases are mild.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.4102/safp.v62i1.5115" target="_blank">Primary care management of the coronavirus (COVID-19).</a></td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> incubation period for covid-19  <font color="red">is between 2-10 days</font> , according to the world health organization (who).</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/" target="_blank">Novel Coronavirus Disease 2019 (COVID-19): An Emerging Infectious Disease in the 21st Century</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> incubation period for covid-19  <font color="red">is between 2-10 days</font> , according to the world health organization (who).</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/" target="_blank">Novel Coronavirus Disease 2019 (COVID-19): An Emerging Infectious Disease in the 21st Century</a></td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> source of infection including the patients, asymptomatic carrier and patients in  <font color="red">the incubation period are contagious</font> .</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/" target="_blank">Analysis on the epidemic factors for the Corona Virus Disease</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> the mean incubation period for the entire period was estimated  <font color="red">at 5. 2 days</font>  (1.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1016/S1473-3099(20)30230-9" target="_blank">Evolving epidemiology and transmission dynamics of coronavirus disease 2019 outside Hubei province, China: a descriptive and modelling study.</a></td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> in worldwide, the incubation period of covid-19  <font color="red">was 3 to 7 days</font>  and approximately 80 % of infections are mild or asymptomatic, 15 % are severe, requiring oxygen, and 5 % are critical infections, requiring ventilation.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1002/rmv.2107" target="_blank">Immune responses and pathogenesis of SARS-CoV-2 during an outbreak in Iran: Comparison with SARS and MERS.</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> in worldwide, the incubation period of covid-19  <font color="red">was 3 to 7 days</font>  and approximately 80 % of infections are mild or asymptomatic, 15 % are severe, requiring oxygen, and 5 % are critical infections, requiring ventilation.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1002/rmv.2107" target="_blank">Immune responses and pathogenesis of SARS-CoV-2 during an outbreak in Iran: Comparison with SARS and MERS.</a></td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> our results show that the incubation period falls within the range  <font color="red">of 2-14 days</font>  with 95 % confidence and has a mean of around 5 days when approximated using the best-fit lognormal distribution.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.3390/jcm9020538" target="_blank">Incubation Period and Other Epidemiological Characteristics of 2019 Novel Coronavirus Infections with Right Truncation: A Statistical Analysis of Publicly Available Case Data</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> conclusion : the si of covid-19 may  <font color="red">be shorter than the preliminary estimates in previous works. given the likelihood that si could be shorter than the incubation period</font> , pre-symptomatic transmission may occur, and extra efforts on timely contact tracing and quarantine are recommended in combating the covid-19 outbreak.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1101/2020.02.21.20026559" target="_blank">Estimating the serial interval of the novel coronavirus disease (COVID-19): A statistical analysis using the public data in Hong Kong from January 16 to February 15, 2020</a></td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> the mean incubation period for the entire period was estimated  <font color="red">at 5. 2 days (1. 8-12. 4)</font>  and the mean serial interval at 5.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1016/S1473-3099(20)30230-9" target="_blank">Evolving epidemiology and transmission dynamics of coronavirus disease 2019 outside Hubei province, China: a descriptive and modelling study.</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> our results show that the incubation period falls within the range  <font color="red">of 2-14 days</font>  with 95 % confidence and has a mean of around 5 days when approximated using the best-fit lognormal distribution.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.3390/jcm9020538" target="_blank">Incubation Period and Other Epidemiological Characteristics of 2019 Novel Coronavirus Infections with Right Truncation: A Statistical Analysis of Publicly Available Case Data</a></td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> with an estimation  <font color="red">of 8 days</font>  incubation period and 6 days 17 serial interval, our results indicate that there may exist infectiousness during the 18 incubation period for 2019-ncov.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1101/2020.03.01.20028944" target="_blank">Epidemiologic Characteristics of COVID-19 in Guizhou, China</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> the full range of incubation periods of the covid-19 cases ranged  <font color="red">from 0 to 33 days</font>  among 2015 cases.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1101/2020.03.15.20036533" target="_blank">Is a 14-day quarantine period optimal for effectively controlling coronavirus disease 2019 (COVID-19)?</a></td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> <font color="red">backgrounds : the emerging virus, severe acute respiratory syndrome coronavirus 2 (sars-cov-2), has caused a large outbreak of novel coronavirus disease (covid-19) in wuhan, china since december 2019. based on the publicly available surveillance data, we identified 21 transmission chains in hong kong and estimated the serial interval (si) of covid-19. methods : index cases were identified and reported after symptoms onset, and contact tracing was conducted to collect the data of the associated secondary cases. an interval censored likelihood framework is adopted to fit a gamma distribution function to govern the si of covid-19. findings : assuming a gamma distributed model, we estimated the mean of si at 4. 4 days (95 % ci : 2. 9&#226;&#710;&#8217;6. 7) and sd of si at 3. 0 days (95 % ci : 1. 8&#226;&#710;&#8217;5. 8) by using the information of all 21 transmission chains in hong kong. conclusion : the si of covid-19 may be shorter than the preliminary estimates in previous works. given the likelihood that si could be shorter than the incubation period</font> , pre-symptomatic transmission may occur, and extra efforts on timely contact tracing and quarantine are recommended in combating the covid-19 outbreak.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1101/2020.02.21.20026559" target="_blank">Estimating the serial interval of the novel coronavirus disease (COVID-19): A statistical analysis using the public data in Hong Kong from January 16 to February 15, 2020</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> with an estimation  <font color="red">of 8 days incubation period and 6 days 17 serial interval, our results indicate that there may exist infectiousness during the 18 incubation period</font>  for 2019-ncov.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1101/2020.03.01.20028944" target="_blank">Epidemiologic Characteristics of COVID-19 in Guizhou, China</a></td>
#     </tr>
#     <tr>
#       <th>10</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml">the covid-19 had a high r0,  <font color="red">a long incubation period</font> , and a short serial interval.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1016/j.ijid.2020.03.071" target="_blank">Insight into 2019 novel coronavirus &#226;&#8364;&#8221; an updated intrim review and lessons from SARS-CoV and MERS-CoV</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml">the covid-19 had a high r0,  <font color="red">a long incubation period</font> , and a short serial interval.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1016/j.ijid.2020.03.071" target="_blank">Insight into 2019 novel coronavirus &#226;&#8364;&#8221; an updated intrim review and lessons from SARS-CoV and MERS-CoV</a></td>
#     </tr>
#     <tr>
#       <th>11</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> six children had a family exposure and could provide the exact dates of close contact with someone who was confirmed to have 2019-ncov infection, among whom the median incubation period  <font color="red">was 7. 5 days</font> .</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1002/ppul.24762" target="_blank">Novel coronavirus infection in children outside of Wuhan, China.</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> six children had a family exposure and could provide the exact dates of close contact with someone who was confirmed to have 2019-ncov infection, among whom the median incubation period  <font color="red">was 7. 5 days</font> .</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1002/ppul.24762" target="_blank">Novel coronavirus infection in children outside of Wuhan, China.</a></td>
#     </tr>
#     <tr>
#       <th>12</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> the median incubation period of children and adults  <font color="red">was 5 days (range 3-12 days) and 4 days (range 2-12 days)</font> , respectively.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1002/jmv.25835" target="_blank">A comparative-descriptive analysis of clinical characteristics in 2019-Coronavirus-infected children and adults.</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> the median incubation period of children and adults  <font color="red">was 5 days</font>  (range 3-12 days) and 4 days (range 2-12 days), respectively.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1002/jmv.25835" target="_blank">A comparative-descriptive analysis of clinical characteristics in 2019-Coronavirus-infected children and adults.</a></td>
#     </tr>
#     <tr>
#       <th>13</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> the covid-19 generally had a high reproductive number,  <font color="red">a long incubation period</font> , a short serial interval and a low case fatality rate (much higher in patients with comorbidities) than sars and mers.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1016/j.ijid.2020.03.071" target="_blank">Insight into 2019 novel coronavirus - an updated intrim review and lessons from SARS-CoV and MERS-CoV.</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> the covid-19 generally had a high reproductive number,  <font color="red">a long incubation period</font> , a short serial interval and a low case fatality rate (much higher in patients with comorbidities) than sars and mers.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1016/j.ijid.2020.03.071" target="_blank">Insight into 2019 novel coronavirus - an updated intrim review and lessons from SARS-CoV and MERS-CoV.</a></td>
#     </tr>
#     <tr>
#       <th>14</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> we reported  <font color="red">the median (iqr) incubation period</font>  of sars-cov-2.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1016/s0140-6736(20)30528-6" target="_blank">Investigation of three clusters of COVID-19 in Singapore: implications for surveillance and response measures</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> we reported  <font color="red">the median (iqr)</font>  incubation period of sars-cov-2.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1016/s0140-6736(20)30528-6" target="_blank">Investigation of three clusters of COVID-19 in Singapore: implications for surveillance and response measures</a></td>
#     </tr>
#     <tr>
#       <th>15</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml">  <font color="red">we collected extensive individual case reports across china and estimated key epidemiologic parameters</font> , including the incubation period.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.3201/eid2607.200282" target="_blank">High Contagiousness and Rapid Spread of Severe Acute Respiratory Syndrome Coronavirus 2.</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> results show that the doubling time early in the epidemic in wuhan  <font color="red">was 2. 3-3. 3 days</font> .</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.3201/eid2607.200282" target="_blank">High Contagiousness and Rapid Spread of Severe Acute Respiratory Syndrome Coronavirus 2.</a></td>
#     </tr>
#     <tr>
#       <th>16</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> it is characterized by untypical clinical symptoms,  <font color="red">long incubation period</font> , concealment and strong infectivity.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/" target="_blank">Expert consensus on clinical diagnosis and treatment in foot and ankle surgery department during the epidemic of novel coronavirus pneumonia</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> <font color="red">since december 2019, corona virus disease 2019 (covid-19), an emerging infection disease occurred in wuhan, has spread in the mainland china. the epidemic factors on the basis of knowledge of sars-cov-2 were discussed in this paper. this puts a lot of pressure on clinical resources and care. sars-cov-2 is a novel corona virus, the onset of covid-19 is slow, and the pathogenesis of sars-cov-2 remains unclear and may lead to multiple organ damage. these put a lot of pressure on clinical resources and care. source of infection including the patients, asymptomatic carrier and patients in the incubation period</font>  are contagious.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/" target="_blank">Analysis on the epidemic factors for the Corona Virus Disease</a></td>
#     </tr>
#     <tr>
#       <th>17</th>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml">there is a new public health crises threatening the world with the emergence and spread  <font color="red">of 2019</font>  novel coronavirus (2019-ncov)</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1007/s12098-020-03263-6" target="_blank">A Review of Coronavirus Disease-2019 (COVID-19)</a></td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> <font color="red">since december 2019, corona virus disease 2019 (covid-19), an emerging infection disease occurred in wuhan, has spread in the mainland china. the epidemic factors on the basis of knowledge of sars-cov-2 were discussed in this paper. this puts a lot of pressure on clinical resources and care. sars-cov-2 is a novel corona virus, the onset of covid-19 is slow, and the pathogenesis of sars-cov-2 remains unclear and may lead to multiple organ damage. these put a lot of pressure on clinical resources and care. source of infection including the patients, asymptomatic carrier and patients in the incubation period</font>  are contagious.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.3760/cma.j.cn112150-20200227-00196" target="_blank">Analysis on the epidemic factors for the Corona Virus Disease</a></td>
#     </tr>
#     <tr>
#       <th>18</th>
#       <td>NaN</td>
#       <td>NaN</td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> it is characterized by untypical clinical symptoms,  <font color="red">long incubation period</font> , concealment and strong infectivity.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/" target="_blank">Expert consensus on clinical diagnosis and treatment in foot and ankle surgery department during the epidemic of novel coronavirus pneumonia</a></td>
#     </tr>
#     <tr>
#       <th>19</th>
#       <td>NaN</td>
#       <td>NaN</td>
#       <td><div xmlns:html="http://www.w3.org/1999/xhtml"> we estimated that the mean and median of its incubation  <font color="red">were 5. 84 and 5. 0 days</font>  via bootstrap and proposed monte carlo simulations.</div></td>
#       <td><a xmlns:html="http://www.w3.org/1999/xhtml" href="https://doi.org/10.1101/2020.02.24.20027474" target="_blank">Estimate the incubation period of coronavirus 2019 (COVID-19)</a></td>
#     </tr>
#   </tbody>
# </table>
# 
# ## Possible improvements
# * Use [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) trained on BioBERT instead of the Universal Sentence Encoder
# * Adapt keywords and the questions to obtain better-fitting answers
# * Add a minimal length for the results of the Question Answering

# Set up some parameters in the kernel

# In[ ]:


USE_SUMMARY = True
FIND_PDFS = False


# First we need to go get openJDK 11 set up

# In[ ]:


import os
#%%capture
get_ipython().system('curl -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz')
get_ipython().system('mv openjdk-11.0.2_linux-x64_bin.tar.gz /usr/lib/jvm/; cd /usr/lib/jvm/; tar -zxvf openjdk-11.0.2_linux-x64_bin.tar.gz')
get_ipython().system('update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-11.0.2/bin/java 1')
get_ipython().system('update-alternatives --set java /usr/lib/jvm/jdk-11.0.2/bin/java')
os.environ["JAVA_HOME"] = "/usr/lib/jvm/jdk-11.0.2"


# Now lets get Pyserini (python wrapped Anserini) setup

# In[ ]:


#%%capture
get_ipython().system('pip install pyserini==0.8.1.0')
from pyserini.search import pysearch


# Now we need the lucene searchable CORD-19 database

# In[ ]:


#%%capture
get_ipython().system('wget -O lucene.tar.gz https://www.dropbox.com/s/d6v9fensyi7q3gb/lucene-index-covid-2020-04-03.tar.gz?dl=0')
get_ipython().system('tar xvfz lucene.tar.gz')
minDate = '2020/04/02'
luceneDir = 'lucene-index-covid-2020-04-03/'


# Now Lets get the Universal Sentence Encoder

# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub
get_ipython().system('mkdir /kaggle/working/sentence_wise_email/')
get_ipython().system('mkdir /kaggle/working/sentence_wise_email/module/')
get_ipython().system('mkdir /kaggle/working/sentence_wise_email/module/module_useT')
# Download the module, and uncompress it to the destination folder. 
get_ipython().system('curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC /kaggle/working//sentence_wise_email/module/module_useT')


# Now lets get the transformer models:
# 
# Instead of using the BertForQuestionAnswering pretrained on SQuAD with the normal BERT embeddings, we use the pre-trained language model for biomedical question answering ([BioASQ](https://github.com/dmis-lab/bioasq-biobert)), which is also trained on SQuAD but with the [BioBERT](https://github.com/dmis-lab/biobert) embeddings.
# 
# The model has been converted from tensorflow checkpoints to the pytorch format using (an adapted) version of the [convert_tf_checkpoint_to_pytorch](https://github.com/yuzcccc/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py) skript.

# In[ ]:


import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

BIOASQ_DIR = '/kaggle/input/bert-pubmed-model/bioasq-biobert'
# QA_MODEL = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# QA_TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
QA_MODEL = BertForQuestionAnswering.from_pretrained(BIOASQ_DIR)
QA_TOKENIZER = BertTokenizer.from_pretrained(BIOASQ_DIR)
QA_MODEL.to(torch_device)
QA_MODEL.eval()

if USE_SUMMARY:
    SUMMARY_TOKENIZER = BartTokenizer.from_pretrained('bart-large-cnn')
    SUMMARY_MODEL = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
    SUMMARY_MODEL.to(torch_device)
    SUMMARY_MODEL.eval()


# now lets get metapub to be able to find pdfs if available

# In[ ]:


if FIND_PDFS:
    get_ipython().system('pip install metapub')


# Now lets get biopython set up so we can go search pubmed if we want to

# In[ ]:


get_ipython().system('pip install biopython')
from Bio import Entrez, Medline

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import re


# # Now we should be set to make our first query of the CORD-19 database.  Lets look at the top 10 results based on our query.
# 
# lest break our query up into two parts:
# 
# 1) a natural language question
# 
# 2) a set of keywords that can help drive the answerini search enginge towards the most interesting results for the question we want to ask
# 
# This is beneficial becuase the answerini portion of the search is not really contextual and cant discipher meaning so the keywords will help drive the search.  This could be refined eventually by using a BERT model to create an embedding from the question being asked.  For right now, this is good enough.

# In[ ]:


query = 'Which non-pharmaceutical interventions limit tramsission'
keywords = '2019-nCoV, SARS-CoV-2, COVID-19, non-pharmaceutical interventions, npi'


# In[ ]:


import json
searcher = pysearch.SimpleSearcher(luceneDir)
hits = searcher.search(query + '. ' + keywords)
n_hits = len(hits)
## collect the relevant data in a hit dictionary
hit_dictionary = {}
for i in range(0, n_hits):
    doc_json = json.loads(hits[i].raw)
    idx = str(hits[i].docid)
    hit_dictionary[idx] = doc_json
    hit_dictionary[idx]['title'] = hits[i].lucene_document.get("title")
    hit_dictionary[idx]['authors'] = hits[i].lucene_document.get("authors")
    hit_dictionary[idx]['doi'] = hits[i].lucene_document.get("doi")

## scrub the abstracts in prep for BERT-SQuAD
for idx,v in hit_dictionary.items():
    abs_dirty = v['abstract']
    # looks like the abstract value can be an empty list
    v['abstract_paragraphs'] = []
    v['abstract_full'] = ''

    if abs_dirty:
        # looks like if it is a list, then the only entry is a dictionary wher text is in 'text' key
        # looks like it is broken up by paragraph if it is in that form.  lets make lists for every paragraph
        # and a new entry that is full abstract text as both could be valuable for BERT derrived QA


        if isinstance(abs_dirty, list):
            for p in abs_dirty:
                v['abstract_paragraphs'].append(p['text'])
                v['abstract_full'] += p['text'] + ' \n\n'

        # looks like in some cases the abstract can be straight up text so we can actually leave that alone
        if isinstance(abs_dirty, str):
            v['abstract_paragraphs'].append(abs_dirty)
            v['abstract_full'] += abs_dirty + ' \n\n'


# Lets try doing a simple BERT-SQuAD QA model first and see how it does
# 
# Originally, found a good example of runnign a BertSQuAD model by Raj Kamil at:
# https://github.com/kamalkraj/BERT-SQuAD
# 
# However, the link to the completly pretrained model broke on 3/28.  Unfortunatly, I did not think to download it and save it myself.  Thus I rebuilt this book based on the completely pretrained model in transformers.  I don't have as good of a text formatting as before, but I think this model works and has the ability (i think) to return an "i don't know, or this isn't relevant" which the other model didn't.  Also, I seem to get an invalid number crash with this new model.  Thats not good as the abstract is not searched.  I will have to look into training a model myself at some point.
# 
# Here is a great tutorial and a helpful resource to me as I got the transformers code up and running:
# https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/
# 
# You will find many similarities between his example code and the function I built to extract abstract information.

# # Build a semantic similarity search capability to rank answers in terms of how closely they line up to the meaning of the NL question

# See this [notebook](https://www.kaggle.com/dirktheeng/universal-sentence-encoder-for-nlp-matching) for a stripped own example.

# In[ ]:


def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
embed_fn = embed_useT('/kaggle/working/sentence_wise_email/module/module_useT')


# In[ ]:


import numpy as np
def reconstructText(tokens, start=0, stop=-1):
    tokens = tokens[start: stop]
    if '[SEP]' in tokens:
        sepind = tokens.index('[SEP]')
        tokens = tokens[sepind+1:]
    txt = ' '.join(tokens)
    txt = txt.replace(' ##', '')
    txt = txt.replace('##', '')
    txt = txt.strip()
    txt = " ".join(txt.split())
    txt = txt.replace(' .', '.')
    txt = txt.replace('( ', '(')
    txt = txt.replace(' )', ')')
    txt = txt.replace(' - ', '-')
    txt_list = txt.split(' , ')
    txt = ''
    nTxtL = len(txt_list)
    if nTxtL == 1:
        return txt_list[0]
    newList =[]
    for i,t in enumerate(txt_list):
        if i < nTxtL -1:
            if t[-1].isdigit() and txt_list[i+1][0].isdigit():
                newList += [t,',']
            else:
                newList += [t, ', ']
        else:
            newList += [t]
    return ''.join(newList)


def makeBERTSQuADPrediction(document, question):
    ## we need to rewrite this function so that it chuncks the document into 250-300 word segments with
    ## 50 word overlaps on either end so that it can understand and check longer abstracts
    nWords = len(document.split())
    input_ids_all = QA_TOKENIZER.encode(question, document)
    tokens_all = QA_TOKENIZER.convert_ids_to_tokens(input_ids_all)
    overlapFac = 1.1
    if len(input_ids_all)*overlapFac > 2048:
        nSearchWords = int(np.ceil(nWords/5))
        quarter = int(np.ceil(nWords/4))
        docSplit = document.split()
        docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                     ' '.join(docSplit[quarter-int(nSearchWords*overlapFac/2):quarter+int(quarter*overlapFac/2)]),
                     ' '.join(docSplit[quarter*2-int(nSearchWords*overlapFac/2):quarter*2+int(quarter*overlapFac/2)]),
                     ' '.join(docSplit[quarter*3-int(nSearchWords*overlapFac/2):quarter*3+int(quarter*overlapFac/2)]),
                     ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
        input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]        
        
    elif len(input_ids_all)*overlapFac > 1536:
        nSearchWords = int(np.ceil(nWords/4))
        third = int(np.ceil(nWords/3))
        docSplit = document.split()
        docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                     ' '.join(docSplit[third-int(nSearchWords*overlapFac/2):third+int(nSearchWords*overlapFac/2)]),
                     ' '.join(docSplit[third*2-int(nSearchWords*overlapFac/2):third*2+int(nSearchWords*overlapFac/2)]),
                     ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
        input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]        
        
    elif len(input_ids_all)*overlapFac > 1024:
        nSearchWords = int(np.ceil(nWords/3))
        middle = int(np.ceil(nWords/2))
        docSplit = document.split()
        docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), 
                     ' '.join(docSplit[middle-int(nSearchWords*overlapFac/2):middle+int(nSearchWords*overlapFac/2)]),
                     ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
        input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]
    elif len(input_ids_all)*overlapFac > 512:
        nSearchWords = int(np.ceil(nWords/2))
        docSplit = document.split()
        docPieces = [' '.join(docSplit[:int(nSearchWords*overlapFac)]), ' '.join(docSplit[-int(nSearchWords*overlapFac):])]
        input_ids = [QA_TOKENIZER.encode(question, dp) for dp in docPieces]
    else:
        input_ids = [input_ids_all]
    absTooLong = False    
    
    answers = []
    cons = []
    for iptIds in input_ids:
        tokens = QA_TOKENIZER.convert_ids_to_tokens(iptIds)
        sep_index = iptIds.index(QA_TOKENIZER.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(iptIds) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        assert len(segment_ids) == len(iptIds)
        n_ids = len(segment_ids)
        #print(n_ids)

        if n_ids < 512:
            start_scores, end_scores = QA_MODEL(torch.tensor([iptIds]).to(torch_device), 
                                     token_type_ids=torch.tensor([segment_ids]).to(torch_device))
        else:
            #this cuts off the text if its more than 512 words so it fits in model space
            #need run multiple inferences for longer text. add to the todo
            print('****** warning only considering first 512 tokens, document is '+str(nWords)+' words long.  There are '+str(n_ids)+ ' tokens')
            absTooLong = True
            start_scores, end_scores = QA_MODEL(torch.tensor([iptIds[:512]]).to(torch_device), 
                                     token_type_ids=torch.tensor([segment_ids[:512]]).to(torch_device))
        start_scores = start_scores[:,1:-1]
        end_scores = end_scores[:,1:-1]
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        #print(answer_start, answer_end)
        answer = reconstructText(tokens, answer_start, answer_end+2)
    
        if answer.startswith('. ') or answer.startswith(', '):
            answer = answer[2:]
            
        c = start_scores[0,answer_start].item()+end_scores[0,answer_end].item()
        answers.append(answer)
        cons.append(c)
    
    maxC = max(cons)
    iMaxC = [i for i, j in enumerate(cons) if j == maxC][0]
    confidence = cons[iMaxC]
    answer = answers[iMaxC]
    
    sep_index = tokens_all.index('[SEP]')
    full_txt_tokens = tokens_all[sep_index+1:]
    
    abs_returned = reconstructText(full_txt_tokens)

    ans={}
    ans['answer'] = answer
    #print(answer)
    if answer.startswith('[CLS]') or answer_end.item() < sep_index or answer.endswith('[SEP]'):
        ans['confidence'] = -1000000
    else:
        #confidence = torch.max(start_scores) + torch.max(end_scores)
        #confidence = np.log(confidence.item())
        ans['confidence'] = confidence
    #ans['start'] = answer_start.item()
    #ans['end'] = answer_end.item()
    ans['abstract_bert'] = abs_returned
    ans['abs_too_long'] = absTooLong
    return ans


# Now we can write a function to do an Open Domain QA on all the abstracts

# In[ ]:


from tqdm import tqdm
def searchAbstracts(hit_dictionary, question):
    abstractResults = {}
    for k,v in tqdm(hit_dictionary.items()):
        abstract = v['abstract_full']
        if abstract:
            ans = makeBERTSQuADPrediction(abstract, question)
            if ans['answer']:
                confidence = ans['confidence']
                abstractResults[confidence]={}
                abstractResults[confidence]['answer'] = ans['answer']
                #abstractResults[confidence]['start'] = ans['start']
                #abstractResults[confidence]['end'] = ans['end']
                abstractResults[confidence]['abstract_bert'] = ans['abstract_bert']
                abstractResults[confidence]['idx'] = k
                abstractResults[confidence]['abs_too_long'] = ans['abs_too_long']
                
    cList = list(abstractResults.keys())

    if cList:
        maxScore = max(cList)
        total = 0.0
        exp_scores = []
        for c in cList:
            s = np.exp(c-maxScore)
            exp_scores.append(s)
        total = sum(exp_scores)
        for i,c in enumerate(cList):
            abstractResults[exp_scores[i]/total] = abstractResults.pop(c)
    return abstractResults


# In[ ]:


answers = searchAbstracts(hit_dictionary, query)


# Lets put this together in a more eye pleasing way
# 
# I noticed that the more confident the BERT-SQuAD is, the less text it seems to highlight.  To make sure that we get the full human understandable concept highlighted, I will set it to highlight the sentance that BERT-SQuAD identified.

# In[ ]:


workingPath = '/kaggle/working'
import pandas as pd
if FIND_PDFS:
    from metapub import UrlReverse
    from metapub import FindIt
from IPython.core.display import display, HTML

#from summarizer import Summarizer
#summarizerModel = Summarizer()
def displayResults(hit_dictionary, answers, question):
    
    question_HTML = '<div style="font-family: Times New Roman; font-size: 28px; padding-bottom:28px"><b>Query</b>: '+question+'</div>'
    #all_HTML_txt = question_HTML
    confidence = list(answers.keys())
    confidence.sort(reverse=True)
    
    confidence = list(answers.keys())
    confidence.sort(reverse=True)
    

    for c in confidence:
        if c>0 and c <= 1 and len(answers[c]['answer']) != 0:
            if 'idx' not in  answers[c]:
                continue
            rowData = []
            idx = answers[c]['idx']
            title = hit_dictionary[idx]['title']
            authors = hit_dictionary[idx]['authors'] + ' et al.'
            doi = '<a href="https://doi.org/'+hit_dictionary[idx]['doi']+'" target="_blank">' + title +'</a>'

            
            full_abs = answers[c]['abstract_bert']
            bert_ans = answers[c]['answer']
            
            
            split_abs = full_abs.split(bert_ans)
            sentance_beginning = split_abs[0][split_abs[0].rfind('.')+1:]
            if len(split_abs) == 1:
                sentance_end_pos = len(full_abs)
                sentance_end =''
            else:
                sentance_end_pos = split_abs[1].find('. ')+1
                if sentance_end_pos == 0:
                    sentance_end = split_abs[1]
                else:
                    sentance_end = split_abs[1][:sentance_end_pos]
                
            #sentance_full = sentance_beginning + bert_ans+ sentance_end
            answers[c]['full_answer'] = sentance_beginning+bert_ans+sentance_end
            answers[c]['sentence_beginning'] = sentance_beginning
            answers[c]['sentence_end'] = sentance_end
            answers[c]['title'] = title
            answers[c]['doi'] = doi
        else:
            answers.pop(c)
    
    
    ## now rerank based on semantic similarity of the answers to the question
    cList = list(answers.keys())
    allAnswers = [answers[c]['full_answer'] for c in cList]
    
    messages = [question]+allAnswers
    
    encoding_matrix = embed_fn(messages)
    similarity_matrix = np.inner(encoding_matrix, encoding_matrix)
    rankings = similarity_matrix[1:,0]
    
    for i,c in enumerate(cList):
        answers[rankings[i]] = answers.pop(c)

    ## now form pandas dv
    confidence = list(answers.keys())
    confidence.sort(reverse=True)
    pandasData = []
    ranked_aswers = []
    for c in confidence:
        rowData=[]
        title = answers[c]['title']
        doi = answers[c]['doi']
        idx = answers[c]['idx']
        rowData += [idx]            
        sentance_html = '<div>' +answers[c]['sentence_beginning'] + " <font color='red'>"+answers[c]['answer']+"</font> "+answers[c]['sentence_end']+'</div>'
        
        rowData += [sentance_html, c, doi]
        pandasData.append(rowData)
        ranked_aswers.append(' '.join([answers[c]['full_answer']]))
    
    if FIND_PDFS:
        pdata2 = []
        for rowData in pandasData:
            rd = rowData
            idx = rowData[0]
            if str(idx).startswith('pm_'):
                pmid = idx[3:]
            else:
                try:
                    test = UrlReverse('https://doi.org/'+hit_dictionary[idx]['doi'])
                    if test is not None:
                        pmid = test.pmid
                    else:
                        pmid = None
                except:
                    pmid = None
            pdfLink = None
            if pmid is not None:
                try:
                    pdfLink = FindIt(str(pmid))
                except:
                    pdfLink = None
            if pdfLink is not None:
                pdfLink = pdfLink.url

            if pdfLink is None:

                rd += ['Not Available']
            else:
                rd += ['<a href="'+pdfLink+'" target="_blank">PDF Link</a>']
            pdata2.append(rowData)
    else:
        pdata2 = pandasData
        
    
    display(HTML(question_HTML))
    
    if USE_SUMMARY:
        ## try generating an exacutive summary with extractive summarizer
        allAnswersTxt = ' '.join(ranked_aswers[:6]).replace('\n','')
    #    exec_sum = summarizerModel(allAnswersTxt, min_length=1, max_length=500)    
     #   execSum_HTML = '<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:18px"><b>BERT Extractive Summary:</b>: '+exec_sum+'</div>'

        answers_input_ids = SUMMARY_TOKENIZER.batch_encode_plus([allAnswersTxt], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
        summary_ids = SUMMARY_MODEL.generate(answers_input_ids,
                                               num_beams=10,
                                               length_penalty=1.2,
                                               max_length=1024,
                                               min_length=64,
                                               no_repeat_ngram_size=4)

        exec_sum = SUMMARY_TOKENIZER.decode(summary_ids.squeeze(), skip_special_tokens=True)
        execSum_HTML = '<div style="font-family: Times New Roman; font-size: 18px; margin-bottom:1pt"><b>BART Abstractive Summary:</b>: '+exec_sum+'</div>'
        display(HTML(execSum_HTML))
        warning_HTML = '<div style="font-family: Times New Roman; font-size: 12px; padding-bottom:12px; color:#CCCC00; margin-top:1pt"> Warning this is an autogenerated summary based on semantic search of abstracts, always examine the sources before accepting this conclusion.  If the evidence only mentions topic in passing or the evidence is not clear, the summary will likely not clearly answer the question.</div>'
        display(HTML(warning_HTML))

#    display(HTML('<div style="font-family: Times New Roman; font-size: 18px; padding-bottom:18px"><b>Body of Evidence:</b></div>'))
    
    if FIND_PDFS:
        df = pd.DataFrame(pdata2, columns = ['Lucene ID', 'BioASQ Answer with Highlights', 'Confidence', 'Title/Link','PDF Link'])
    else:
        df = pd.DataFrame(pdata2, columns = ['Lucene ID', 'BioASQ Answer with Highlights', 'Confidence', 'Title/Link'])
    
    display(HTML(df.to_html(render_links=True, escape=False)))
    return df
    
displayResults(hit_dictionary, answers, query)


# Lets search pubmed too to fill in the gaps and get the latest papers that may not be in the lucene database

# In[ ]:


def getrecord(id, db):
    handle = Entrez.efetch(db=db, id=id, rettype='Medline', retmode='text')
    rec = handle.read()
    handle.close()
    return rec

def pubMedSearch(terms, db='pubmed', mindate='2019/12/01'):
    handle = Entrez.esearch(db = db, term = terms, retmax=10, mindate=mindate)
    record = Entrez.read(handle)
    record_db = {}
    for id in record['IdList']:
        record = getrecord(id,db)
        recfile = StringIO(record)
        rec = Medline.read(recfile)
        if 'AB' in rec and 'AU' in rec and 'LID' in rec and 'TI' in rec:
            if '10.' in rec['LID'] and ' [doi]' in rec['LID']:
                record_db['pm_'+id] = {}
                record_db['pm_'+id]['authors'] = ' '.join(rec['AU'])
                record_db['pm_'+id]['doi'] = '10.'+rec['LID'].split('10.')[1].split(' [doi]')[0]
                record_db['pm_'+id]['abstract'] = rec['AB']
                record_db['pm_'+id]['title'] = rec['TI']
        
    return record_db


Entrez.email = 'pubmedkaggle@gmail.com'


# In[ ]:


def searchDatabase(question, keywords, pysearch, lucene_database, pm_kw = '', minDate='2019/12/01', k=20):
    ## search the lucene database with a combination of the question and the keywords
    searcher = pysearch.SimpleSearcher(lucene_database)
    hits = searcher.search(question + '. ' + keywords, k=k)
    n_hits = len(hits)
    ## collect the relevant data in a hit dictionary
    hit_dictionary = {}
    for i in range(0, n_hits):
        doc_json = json.loads(hits[i].raw)
        idx = str(hits[i].docid)
        hit_dictionary[idx] = doc_json
        hit_dictionary[idx]['title'] = hits[i].lucene_document.get("title")
        hit_dictionary[idx]['authors'] = hits[i].lucene_document.get("authors")
        hit_dictionary[idx]['doi'] = hits[i].lucene_document.get("doi")
        
    
    if pm_kw:
        new_hits = pubMedSearch(pm_kw, db='pubmed', mindate=minDate)
        hit_dictionary.update(new_hits)
    
    ## scrub the abstracts in prep for BERT-SQuAD
    for idx,v in hit_dictionary.items():
        abs_dirty = v['abstract']
        # looks like the abstract value can be an empty list
        v['abstract_paragraphs'] = []
        v['abstract_full'] = ''

        if abs_dirty:
            # looks like if it is a list, then the only entry is a dictionary wher text is in 'text' key
            # looks like it is broken up by paragraph if it is in that form.  lets make lists for every paragraph
            # and a new entry that is full abstract text as both could be valuable for BERT derrived QA


            if isinstance(abs_dirty, list):
                for p in abs_dirty:
                    v['abstract_paragraphs'].append(p['text'])
                    v['abstract_full'] += p['text'] + ' \n\n'

            # looks like in some cases the abstract can be straight up text so we can actually leave that alone
            if isinstance(abs_dirty, str):
                v['abstract_paragraphs'].append(abs_dirty)
                v['abstract_full'] += abs_dirty + ' \n\n'
    ## Search collected abstracts with BERT-SQuAD
    answers = searchAbstracts(hit_dictionary, question)
    
    ## display results in a nice format
    return displayResults(hit_dictionary, answers, question)


# Lets try this with the same question and kw to see if it produces the same results we just got

# In[ ]:


#searchDatabase(query, keywords, pysearch, luceneDir, minDate=minDate)
query = "How long is the incubation period for the virus"
keywords = "2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, hours, days, period"
pm_kw = "2019-nCoV, incubation period"
df = searchDatabase(query, keywords, pysearch, luceneDir, pm_kw=pm_kw, minDate=minDate)


# Great that worked as expected.  Now lets try some new questions

# # Define All The Questions for the Competition

# In[ ]:


all_topics=[
    'What is known about transmission, incubation, and environmental stability?',
    'What do we know about COVID-19 risk factors?',
    'What do we know about virus genetics, origin, and evolution?',
    'What do we know about vaccines and therapeutics?',
    'What do we know about non-pharmaceutical interventions?',
    'What has been published about medical care?',
    'What do we know about diagnostics and surveillance?'
    'What has been published about information sharing and inter-sectoral collaboration?',
    'What has been published about ethical and social science considerations?'
]
topic_area = {}

#0
#What is known about transmission, incubation, and environmental stability?
question_list = []
kw_list = []
pm_kw_list = []
question_list.append("Is the virus transmitted by aerisol, droplets, food, close contact, fecal matter, or water")
kw_list.append("2019-nCoV,SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, person to person, human to human, interpersonal contact, air, water,fecal, surfaces, aerisol, transmission, shedding")
pm_kw_list.append("2019-nCoV, transmission, shedding")

question_list.append( "How long is the incubation period for the virus")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, hours, days, period")
pm_kw_list.append("2019-nCoV, incubation period")

question_list.append("Can the virus be transmitted asymptomatically or during the incubation period")
kw_list.append("2019-nCoV, COVID-19, coronavirus, novel coronavirus, asymptomatic, person to person, human to human, transmission")
pm_kw_list.append("2019-nCoV, asymptomatic, transmission")

question_list.append("What is the quantity of asymptomatic shedding")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, asymptomatic, shedding, percentage, rate, quantity, pediatric")
pm_kw_list.append("2019-nCoV, asymptomatic, shedding")

question_list.append("How does temperature and humidity affect the tramsmission of 2019-nCoV")
kw_list.append("2019-nCoV, COVID-19, coronavirus, novel coronavirus, temperature, humidity")
pm_kw_list.append("2019-nCoV, temperature, humidity")

question_list.append("How long can 2019-nCoV remain viable on inanimate, environmental, or common surfaces")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, inanimate, environmental, touch, copper, plastic, steel, wood, fabric, glass, porous, nonporous")
pm_kw_list.append("2019-nCoV, surface")

question_list.append("What types of inanimate or environmental surfaces affect transmission, survival, or  inactivation of 2019-nCov")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, inanimate, environmental, touch, copper, plastic, steel, wood, fabric, glass, porous, nonporous")
pm_kw_list.append("2019-nCoV, surface")

question_list.append("Can the virus be found in nasal discharge, sputum, urine, fecal matter, or blood")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, shedding, body fluid")
pm_kw_list.append("2019-nCoV, body fluids")

topic_area['What is known about transmission, incubation, and environmental stability?'] = list(zip(question_list,kw_list, pm_kw_list))



#1
#What do we know about COVID-19 risk factors?
question_list = []
kw_list = []
pm_kw_list = []
question_list.append("What risk factors contribute to the severity of 2019-nCoV")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, susceptible, smoking, smoker, neonates, pregnant, socio-economic, behavioral, age, elderly, young, old, children")
pm_kw_list.append("2019-nCoV, risk factors")

question_list.append("How does hypertension affect patients")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, hypertension, blood pressure, comorbidity")
pm_kw_list.append("2019-nCoV, hypertension, comorbidity")

question_list.append("How does heart disease affect patients")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, heart disease, comorbidity")
pm_kw_list.append("2019-nCoV, heart disease, comorbidity")

question_list.append("How does copd affect patients")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, copd, chronic obstructive pulmonary disease")
pm_kw_list.append("2019-nCoV, copd, chronic obstructive pulmonary disease")

question_list.append("How does smoking affect 2019-nCoV patients")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, smoking, smoker")
pm_kw_list.append("2019-nCoV, smoking, smoker")

question_list.append("How does pregnancy affect patients")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, pregnant, pregnancy")
pm_kw_list.append("2019-nCoV, pregnant, pregnancy")

question_list.append("What are the case fatality rates for 2019-nCoV patients")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, fatality rate")
pm_kw_list.append("2019-nCoV, fatality, statistics, death")

question_list.append("What is the case fatality rate in Italy")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, fatality rate, Italy")
pm_kw_list.append("2019-nCoV, fatality, statistics, death")

question_list.append("What public health policies prevent or control the spread of 2019-nCoV")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, guidance, prevention measures, public health, community, prevention, administration, government, health department, policy, control measures, travel")
pm_kw_list.append("2019-nCoV, guidance, public health,  policy, control measures")

topic_area['What do we know about COVID-19 risk factors?'] = list(zip(question_list,kw_list, pm_kw_list))


#2
#What do we know about virus genetics, origin, and evolution?
question_list = []
kw_list = []
pm_kw_list = []
question_list.append("Can animals transmit 2019-nCoV")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, animals, zoonotic, farm, spillover, animal to human, human to animal, reservoir")
pm_kw_list.append("2019-nCoV, spillover, reservoir")

question_list.append("What animal did 2019-nCoV come from")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, animals, zoonotic, farm, spillover, animal to human, bats, snakes, exotic animals")
pm_kw_list.append("2019-nCoV, zoonotic")

question_list.append("What real-time genomic tracking tools exist")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, real-time, gene, typing, tracking, software, reporting")
pm_kw_list.append('"2019-nCoV, real-time, genomic, tracking')

question_list.append("What regional genetic variations (mutations) exist")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, geography, region, genome, mutations")
pm_kw_list.append("2019-nCoV, geneome, region")

question_list.append("What effors are being done in asia to prevent further outbreaks")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, surveylance, wildlife, livestock, monitoring, asia, prevent, prevention, outbreaks")
pm_kw_list.append("2019-nCoV, surveylance")

topic_area['What do we know about virus genetics, origin, and evolution?'] = list(zip(question_list,kw_list, pm_kw_list))

#3
#What do we know about vaccines and therapeutics?
question_list = []
kw_list = []
pm_kw_list = []
question_list.append("What drugs or therapies are being investigated")
kw_list.append("2019-nCoV,  COVID-19, coronavirus, novel coronavirus, drug, antiviral, testing, clinical trial, study")
pm_kw_list.append("2019-nCoV,  drug, therapy")

question_list.append("What clinical trials for hydroxychloroquine have been completed")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, hydroxychloroquine, clinical trial")
pm_kw_list.append("2019-nCoV, hydroxychloroquine")

question_list.append("What antiviral drug clinical trials have been completed")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, clinical trial")
pm_kw_list.append("2019-nCoV, antiviral")

question_list.append("Are anti-inflammatory drugs recommended")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, ibuprofen, advil, NSAID, anti-inflamatory, treatment")
pm_kw_list.append('2019-nCoV, ibuprofen, NSAID')

topic_area['What do we know about vaccines and therapeutics?'] = list(zip(question_list,kw_list, pm_kw_list))


#4
#What do we know about non-pharmaceutical interventions?
question_list = []
kw_list = []
pm_kw_list = []
question_list.append("Which non-pharmaceutical interventions limit tramsission")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, non-pharmaceutical interventions, npi")
pm_kw_list.append("2019-nCoV, npi")

question_list.append("What are most important barriers to compliance")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, non-pharmaceutical interventions, npi")
pm_kw_list.append('2019-nCoV, npi, barrier to compliance')

topic_area['What do we know about non-pharmaceutical interventions?'] = list(zip(question_list,kw_list, pm_kw_list))

#5
#What has been published about medical care?
question_list = []
kw_list = []
pm_kw_list = []
question_list.append("How does extracorporeal membrane oxygenation affect 2019-nCoV patients")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, extracorporeal membrane oxygenation, ecmo")
pm_kw_list.append('2019-nCoV, ecmo')

question_list.append("What telemedicine and cybercare methods are most effective")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, telemedicine, 5G, cell phone, cyber, cybercare, information technolog, remote, over the phone, internet, web")
pm_kw_list.append('2019-nCoV, telemedicine, cybercare')

question_list.append("How is artificial intelligence being used in real time health delivery")
kw_list.append("2019-nCoV, ai, real-time")
pm_kw_list.append('')

question_list.append("What adjunctive or supportive methods can help patients")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, adjunctive, supportive")
pm_kw_list.append('')

topic_area['What has been published about medical care?'] = list(zip(question_list,kw_list, pm_kw_list))

#6
#What do we know about diagnostics and surveillance?
question_list = []
kw_list = []
pm_kw_list = []
question_list.append("What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, diagnosis, tools, detetion")
pm_kw_list.append('2019-nCoV, diagnostic, tools, detetion')

question_list.append("What is being done to increase testing capacity or throughput")
kw_list.append("2019-nCoV, sars-cov-2, covid-19, diagnostic, testing, throughput")
pm_kw_list.append("2019-nCoV, testing capacity OR throughput")

question_list.append("What point of care tests are exist or are being developed")
kw_list.append("2019-nCoV, sars-cov-2, covid-19")
pm_kw_list.append("2019-nCoV, point-of-care")

question_list.append("What is the minimum viral load for detection")
kw_list.append("2019-nCoV, sars-cov-2, covid-19")
pm_kw_list.append("2019-nCoV, viral load")

question_list.append("What markers are used to detect or track COVID-19")
kw_list.append("2019-nCoV, sars-cov-2, covid-19")
pm_kw_list.append("2019-nCoV, markers")

topic_area['What do we know about diagnostics and surveillance?'] = list(zip(question_list,kw_list, pm_kw_list))



#7
#What has been published about information sharing and inter-sectoral collaboration?
question_list = []
kw_list = []
pm_kw_list = []
question_list.append('What collaborations are happening within the research community')
kw_list.append('inter-sectorial, international, collaboration, global, coronavirus, novel coronavirus, sharing')
pm_kw_list.append('2019-nCoV, collaboration, sharing')

topic_area['What has been published about information sharing and inter-sectoral collaboration?'] = list(zip(question_list,kw_list, pm_kw_list))


#8
#What has been published about ethical and social science considerations?
question_list = []
kw_list = []
pm_kw_list = []

question_list.append("What are the major ethical issues related pandemic outbreaks")
kw_list.append("ehtics, pandemic")
pm_kw_list.append("ethics, pandemic")

question_list.append("How do pandemics affect the physical and/or psychological health of doctors and nurses")
kw_list.append("2019-nCoV, sars-cov-2, covid-19, caregivers, health care workers")
pm_kw_list.append("2019-nCoV, physical OR psychological")

question_list.append("What strategies can help doctors and nurses cope with stress in a pandemic")
kw_list.append("2019-nCoV, sars-cov-2, covid-19, caregivers, health care workers")
pm_kw_list.append("2019-nCoV, physical OR psychological")

question_list.append("What factors contribute to rumors and misinformation")
kw_list.append("2019-nCoV, sars-cov-2, covid-19, social media")
pm_kw_list.append("2019-nCoV, misinformation OR social media")

topic_area['What has been published about ethical and social science considerations?'] = list(zip(question_list,kw_list, pm_kw_list))



#-1
# Other interesting Questions
question_list = []
kw_list = []
pm_kw_list = []
question_list.append("What is the immune system response to 2019-nCoV")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, immune, immune system, response, immunity, antibodies")
pm_kw_list.append('2019-nCoV, immune system, immunity, antibodie')

question_list.append("Can personal protective equipment prevent the transmission of 2019-nCoV")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, ppe, masks, gloves, face shields, gown, eye protection")
pm_kw_list.append('2019-nCoV, ppe')

question_list.append("Can 2019-nCoV infect patients a second time")
kw_list.append("2019-nCoV, SARS-CoV-2, COVID-19, coronavirus, novel coronavirus, reinfected, multiple infections, second time, permenant immunity")
pm_kw_list.append('2019-nCoV, reinfected')

topic_area['Other interesting Questions'] = list(zip(question_list,kw_list, pm_kw_list))


def runAllQuestionsByTopic(topic_dict, topic_name):
    for q,kw, pm_kw in topic_dict[topic_name]:
        if q:
            searchDatabase(q, kw, pysearch, luceneDir, pm_kw=pm_kw, minDate=minDate)


# # Use this block to refine specific questions before adding them to the list of all questions

# In[ ]:


question_list = []
kw_list = []
pm_kw_list=[]

question_list.append("What genetic markers are used to detect or track COVID-19")
kw_list.append("2019-nCoV, sars-cov-2, covid-19")
pm_kw_list.append("2019-nCoV, markers")


searchDatabase(question_list[0], kw_list[0], pysearch, luceneDir, pm_kw=pm_kw_list[0], minDate=minDate, k=20)


# # What is known about transmission, incubation, and environmental stability?

# In[ ]:


runAllQuestionsByTopic(topic_area, 'What is known about transmission, incubation, and environmental stability?')


# # What do we know about COVID-19 risk factors?

# In[ ]:


runAllQuestionsByTopic(topic_area, 'What do we know about COVID-19 risk factors?')


# # What do we know about virus genetics, origin, and evolution?

# In[ ]:


runAllQuestionsByTopic(topic_area, 'What do we know about virus genetics, origin, and evolution?')


# # What do we know about vaccines and therapeutics?

# In[ ]:


runAllQuestionsByTopic(topic_area, 'What do we know about vaccines and therapeutics?')


# # What do we know about non-pharmaceutical interventions?

# In[ ]:


runAllQuestionsByTopic(topic_area, 'What do we know about non-pharmaceutical interventions?')


# # What has been published about medical care?

# In[ ]:


runAllQuestionsByTopic(topic_area, 'What has been published about medical care?')


# # What do we know about diagnostics and surveillance?

# In[ ]:


runAllQuestionsByTopic(topic_area, 'What do we know about diagnostics and surveillance?')


# # What has been published about information sharing and inter-sectoral collaboration?

# In[ ]:


runAllQuestionsByTopic(topic_area, 'What has been published about information sharing and inter-sectoral collaboration?')


# # What has been published about ethical and social science considerations?

# In[ ]:


runAllQuestionsByTopic(topic_area, 'What has been published about ethical and social science considerations?')

