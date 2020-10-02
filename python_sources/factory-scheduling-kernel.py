#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pulp


# In[ ]:


factories = pd.DataFrame.from_csv('factory.csv', index_col=['Month', 'Factory'])


# In[ ]:


factories


# In[ ]:


demand = pd.DataFrame.from_csv('demand.csv', index_col=['Month'])


# In[ ]:


demand = pd.DataFrame.from_csv('demand.csv', index_col=['Month'])


# In[ ]:


production = pulp.LpVariable.dicts("production",
                                     ((month, factory) for month, factory in factories.index),
                                     lowBound=0,
                                     cat='Integer')


# In[ ]:


factory_status = pulp.LpVariable.dicts("factory_status",
                                     ((month, factory) for month, factory in factories.index),
                                     cat='Binary')


# In[ ]:


model = pulp.LpProblem("Cost minimising scheduling problem", pulp.LpMinimize)


# In[ ]:


model += pulp.lpSum(
    [production[month, factory] * factories.loc[(month, factory), 'Variable_Costs'] for month, factory in factories.index]
    + [factory_status[month, factory] * factories.loc[(month, factory), 'Fixed_Costs'] for month, factory in factories.index]
)


# In[ ]:


# Production in any month must be equal to demand
months = demand.index
for month in months:
    model += production[(month, 'A')] + production[(month, 'B')] == demand.loc[month, 'Demand']


# In[ ]:


# Production in any month must be between minimum and maximum capacity, or zero.
for month, factory in factories.index:
    min_production = factories.loc[(month, factory), 'Min_Capacity']
    max_production = factories.loc[(month, factory), 'Max_Capacity']
    model += production[(month, factory)] >= min_production * factory_status[month, factory]
    model += production[(month, factory)] <= max_production * factory_status[month, factory]


# In[ ]:



# Factory B is off in May# Factor 
model += factory_status[5, 'B'] == 0
model += production[5, 'B'] == 0


# In[ ]:


model.solve()
pulp.LpStatus[model.status]


# In[ ]:


output = []
for month, factory in production:
    var_output = {
        'Month': month,
        'Factory': factory,
        'Production': production[(month, factory)].varValue,
    }
    output.append(var_output)
output_df = pd.DataFrame.from_records(output).sort_values(['Month', 'Factory'])
output_df.set_index(['Month', 'Factory'], inplace=True)
output_df


# In[ ]:


print(demand)


# In[ ]:


# Print our objective function value (Total Costs)
print (pulp.value(model.objective))

