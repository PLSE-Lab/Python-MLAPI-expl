#!/usr/bin/env python
# coding: utf-8

# ## This kernel explores Indian card payment data using interactive visualizations in Tableau
# RBI releases bank wise payment statistics on a monthly basis. This data contains useful statistics related to ATMs, PoS machines, Credit cards, Debit cards etc., It can can be used to check trend of card based payment in India.
# 
# More info on the statistics can be found here [here](https://www.rbi.org.in/scripts/ATMView.aspx),
# 
# Also, if you are interested in knowing how we gathered this dataset, please chek out this kernel. everything is documented(web scraping, data cleaning etc.,) 
# https://www.kaggle.com/karvalo/indian-card-payment-data-gathering-and-analysis

# [Tableau](http://tableau.com)
# Tableau is one of the powerful and fastest growing data visualization tool used in the Business Intelligence Industry. It helps in creating interactive visualizations in the form of dashboards and worksheets.
# 
# In this kernel we use [Tableau Public](http://public.tableau.com/) to create interactive visualizations for Indian payment dataset
# 

# The data contains monthly statistics of the following information 
# 1. Number of ATM deployed on site by the bank.
# 1. Number of ATM deployed off site by the bank.
# 1. Number of POS deployed online by the bank
# 1. Number of POS deployed offline by the bank
# 1. Total number of credit cards issued outstanding (after adjusting the number of cards withdrawan/cancelled).
# 1. Total number of financial transactions done by the credit card issued by the bank at ATMs
# 1. Total number of financial transactions done by the credit card issued by the bank at POS terminals
# 1. Total value of financial transactions done by the credit card issued by the bank at ATMs
# 1. Total value of financial transactions done by the credit card issued by the bank at POS terminals.
# 1. Total number of debit cards issued outstanding (after adjusting the number of cards withdrawan/cancelled).
# 1. Total number of financial transactions done by the debit card issued by the bank at ATMs
# 1. Total number of financial transactions done by the debit card issued by the bank at POS terminals
# 1. Total value of financial transactions done by the debit card issued by the bank at ATMs
# 1. Total value of financial transactions done by the debit card issued by the bank at POS terminals.
# 
# ### We have data from Apr'2011 to Aug'2019

# So, let is get started with visualization.

# ### Bankwise payment data trend dashboard
# The below dashboard shows the importatant payment indicators bankwise.
# You can select bank of your choice by clicking dropdown list.By deafault, all banks trends are shown but this can be changed by uncheckig (All) options and selecting the required bank from the list.
# Based on your choice the indicators in the dashboard will be updated.
# 
#  click this button to access more information regarding any indicator, And click the same button to come back to home page
#  ![image.png](attachment:image.png)
# 
# Visualizations developed on tableau are interactive in nature and they come with useful tooltips. It can be checked by hovering mouse on the visualization types(bars, lines, pies etc,)

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1573240626563' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;CardPaymentSystemsInIndia&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IndianPaymentEcosystem&#47;CardPaymentSystemsInIndia' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;CardPaymentSystemsInIndia&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1573240626563');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1300px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1300px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1627px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# ### Dashboard with additional insights( historical growth data)

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1573240765302' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;CardPaymentSystemsInIndiapart2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IndianPaymentEcosystem&#47;CardPaymentSystemsInIndiapart2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;CardPaymentSystemsInIndiapart2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1573240765302');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1300px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1300px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1927px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# ### Now, Lets us look at the individual components of the dashboard in detail
# #### First Let's us explore ATMs deployed by banks as of Aug'2019
# The below chart shows the bankwise ATM deployment with their marketshare in percentages
# 
# Visualizations developed on tableau are interactive in nature and they come with useful tooltips. It can be checked by hovering mouse on the visualization types(bars, lines, pies etc,)

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1573746084209' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HT&#47;HT59MJ783&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;HT59MJ783' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;HT&#47;HT59MJ783&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1573746084209');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# ###  PoS deployed by banks as of Aug'2019

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1573746114004' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;FZ&#47;FZJQX4YMK&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;FZJQX4YMK' /> <param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;FZ&#47;FZJQX4YMK&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1573746114004');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# ### Cards(Credit/debit) issued by banks as of Aug'2019

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1573240514662' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;Cardsissued&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IndianPaymentEcosystem&#47;Cardsissued' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;Cardsissued&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1573240514662');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# ### Trend of ATM deployment and transactions over a time(from Apr'2011 to Aug'2019) Using dual axis line chart

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1573240588676' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;ATMtrend&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IndianPaymentEcosystem&#47;ATMtrend' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;ATMtrend&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1573240588676');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# ### some observation on ATM trend
# Number ATM's deployment has grown significantly from Apr'2011 to Nov'2016. But after that there is no notable increase in deployment. It could be because of "bank note demonetisation" and indian govt's push towards digital payments
# More info regarding demonetisation is here https://en.wikipedia.org/wiki/2016_Indian_banknote_demonetisation

# ### Trend of PoS deployment and transactions over a time(from Apr'2011 to Aug'2019) Using dual axis line chart

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1573240554249' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;PoSTrend&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IndianPaymentEcosystem&#47;PoSTrend' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;In&#47;IndianPaymentEcosystem&#47;PoSTrend&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1573240554249');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")

