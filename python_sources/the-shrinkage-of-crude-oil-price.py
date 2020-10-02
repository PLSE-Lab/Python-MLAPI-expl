#!/usr/bin/env python
# coding: utf-8

# ### Oil demand is collapsing because of the coronavirus crisis. Supply is shrinking -- but not nearly fast enough. The world is literally running out of room to store unneeded barrels of oil piling up during the coronavirus pandemic. That storage problem is so dire that it caused oil prices to turn negative this week for the first time ever.
# ###      The jaw dropping fall in global crude oil prices may seem a bonanza for an oil guzzler like India, but the plunge is set to change the fortunes of many countries across the world. For most the oil exporters, the demand destruction due to COVID-19 and the supply glut are poised to deal a body blow to one of their most lucrative sources of cash. While countries like Saudi Arabia, Russia and US are large enough to manage the crisis, many small exporters in Africa and West Asia may be faced with an acute cash crunch this year. Let's see a detailed look at how the world's economis may be impacted going forward in this notebook.
# ![](https://image.freepik.com/free-vector/world-oil-crisis-2020-fall-price-per-barrel-influence-coronavirus-dispute-russia-saudi-arabia-stock-concept-illustration_119217-1046.jpg)
# ## How the current oil crisis came about?
# 
# *  Wild trading in an oil futures contract has sent the Wild Texas Intermediate crude to unprecedented low of -$40.32
# *  The negative prices meant the holders of the May contract were unable to accept the physical delivery of the oil and they had to pay to get rid of it, as the worls's storage is at bursting point
# *  Storage is a particularly big problem in the US where WTI oil is delivered at a single, island point - the oil storage terminal in Cushing, Oklahoma, labelled the 'Pipeline Crossroads of the world'
# 
# | Some major oil storage locations      | Capacity in barrels           |
# | ------------- |:-------------:| 
# | Antwerp-Rotterdam-Amsterdam     | 100 mn | 
# | Cushing, Oklahoma    | 76.1 mn | 
# | South Africa Saldanha | 50 mn   | 
# |Egypt terminal at Sidi Kerir | 20 mn|   
# 
# ------                                                                                                        
# ### What is a Future Contract?
# #### Most of the oil trading is in futures, not physical barrels of oil. With such a contract the delivery of the commodity is carried out at a later date. Futures trading help market participants by allowing them to lock into prices, either to sell or buy, but also allows for speculation. 
# ------
# 

# # World's Largest Oil Producers

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go

country_producing = ['United States','Saudi Arabia','Russia','Canada','China','Iraq','UAE','Brazil','Iran','Kuwait']
oil_produced = [19.51,11.81,11.49,5.50,4.89,4.74,4.01,3.67,3.19,2.94]

fig = go.Figure(data=[go.Pie(labels=country_producing,
                             values=oil_produced)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20)
fig.update_layout(title=go.layout.Title(text="<b>World's Largest Oil Producers (mn barrels/day)</b>", font=dict(
                family="Courier New, monospace",
                size=22,
                color="black"
            )))

fig.update_layout(annotations=[
       go.layout.Annotation(
            showarrow=False,
            text='Source: EIA, 2019 data',
            xanchor='right',
            x=0.75,
            xshift=275,
            yanchor='top',
            y=0.05,
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="black"
            )
        )])

fig.show()


# ### The crisis in the amid the coronavirus crisis was compunded by a price war between Russis and Saudi Arabia, but earlier market turmoil saw them agreeing to cut output. Analysts, however, say that will not be enough to address the massive oversupply.
# 
# # World's Largest Oil Consumers

# In[ ]:


country_consuming = ['United States','China','India','Japan','Russia','Saudi Arabia','Brazil','South Korea','Germany','Canada']
consumption = [19.96,13.57,4.32,3.92,3.69,3.33,3.03,2.63,2.45,2.42]

fig = go.Figure(data=[go.Pie(labels=country_consuming,
                             values=consumption)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20)
fig.update_layout(title=go.layout.Title(text="<b>World's Largest Oil Consumers (mn barrels/day)</b>", font=dict(
                family="Courier New, monospace",
                size=22,
                color="black"
            )))

fig.update_layout(annotations=[
       go.layout.Annotation(
            showarrow=False,
            text='Source: EIA, 2017 data',
            xanchor='right',
            x=0.75,
            xshift=275,
            yanchor='top',
            y=0.05,
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="black"
            )
        )])

fig['layout']['xaxis'].update(side='top')

fig.show()


# ### Though output cuts by producers may help push up oil prices, the main problem is less demand. Even with social distancing restrictions gradually easing, it may take time for people to spend and travel as they did before. 66,000 petrol pumps in India are full due to the fall in demand.
# # World's Largest Oil Exporters

# In[ ]:


country_export = ['Saudi Arabia','Russia','Iraq','Canada','UAE','Kuwait','Iran','United States','Nigeria','Kazakhstan','Angola','Norway','Libya','Mexico','Venezuela']
export = [182.5,129,91.7,66.9,58.4,51.7,50.8,48.3,43.6,37.8,36.5,33.3,26.7,26.5,26.4]

fig = go.Figure(data=[go.Pie(labels=country_export,
                             values=export)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20)
fig.update_layout(title=go.layout.Title(text="<b>World's Largest Oil Exporters (US$ billion)</b>", font=dict(
                family="Courier New, monospace",
                size=22,
                color="black"
            )))

fig.update_layout(annotations=[
       go.layout.Annotation(
            showarrow=False,
            text='Source: OECD, 2019 data',
            xanchor='right',
            x=0.75,
            xshift=275,
            yanchor='top',
            y=0.05,
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="black"
            )
        )])

fig['layout']['xaxis'].update(side='top')

fig.show()


# ### The exporting countries depend substancially on oil revenues for their budgets. Saudi Arabia, unlike some of its rivals, has the financial capacity to withstand low oil prices for years, though the price collapse has already inflicted significant pain on Gulf states.

# # Countries Heavily Dependent on Oil Profits to Power GDP
# 

# In[ ]:


affected_country = ['Iraq','Libya','Congo Republic','Kuwait','South Sudan','Saudi Arabia','Oman','Equatorial Guinea','Azerbaijan','Angola','Iran','Gabon','Timor-Leste','Qatar','UAE']
oil_rent = [37.8,37.3,36.7,36.6,31.3,23.1,21.8,19.2,17.9,15.8,15.3,15.3,14.5,14.2,13.1]
affected_country = affected_country[::-1]
oil_rent = oil_rent[::-1]


fig = go.Figure(go.Bar(
            x=oil_rent,
            y=affected_country,
            orientation='h',
            text = oil_rent,
            textposition='auto'))
fig.update_traces(marker_color='purple')

fig.update_layout(title=go.layout.Title(text="<b>Countries Heavily Dependent on Oil Profits to Power GDP</b>", font=dict(
                family="Courier New, monospace",
                size=22,
                color="black"
            )))
fig.update_layout(annotations=[
       go.layout.Annotation(
            showarrow=False,
            text='Source: World Bank',
            xanchor='right',
            x=35,
            xshift=275,
            yanchor='top',
            y=0.05,
            font=dict(
                family="Courier New, monospace",
                size=10,
                color="black"
            )
        )])

fig['layout']['xaxis'].update(side='top')

fig.show()


# ### Oil rents are the effective profits made by countries from the sale of oil produced domestically to foreign customers. Saudi Arabia, Iran, Iraq and, Russia are seeing cancellations of oil shipments.

# ## Gain some, lose some - The United States
# * The US has traditionally been a net gainer when oil prices fall, but this relationship has become one of the slightly murky now, since the country has become one of the largest crude oil exporters in the world
# * Its shale oil fields are only profitable when oil prices remain around $40 a barrel and above. With WTI prices crashing, analysts believe that over a hundred oil firms may file for bankruptcy in the country this year
# * Such a collapse will also impact large American banks such as Wells Fargo and Bank of America, who have substantial exposures to the oil and gas sector
# 
# ## Russia
# * Lower oil prices are bad news for the Russian Economy, since they impact both the country's day-to-day spending while also denting its resources
# * While Russia is in a better position to deal woth its low prices now, the crash is expected to shave off 3% of the country's GDP growth this year
# 
# ## Saudi Arabia and West Asia
# * Countries like Saudi Arabia, Kuwait and others in the region largely depend on oil production to fund most of their social and welfare projects
# * With such low prices, these countries will end up running fiscal deficits if they do not rebalance their budgets
# * However, Saudi Arabisa and a few other nations in the region are capable of churning profits even if Brent prices fall as low as $10 a barrel. 
# 
# ## African Nations
# * While they aren't the largest exporters, nations such as Congo, South Sudan and Angola have some of the highest dependencies in the world
# * Nigeria's 2020 budget, for instance, was predicted oil prices remaning at around $53, and the countrie's govenrnment is scrambling to rebalance its finances
