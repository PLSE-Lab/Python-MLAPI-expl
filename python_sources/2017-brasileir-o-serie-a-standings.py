#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


partidas_2017 = pd.read_csv('../input/cartolafc/2017_partidas.csv')
partidas_2017.head()


# In[ ]:


clubes_2017 = pd.read_csv('../input/cartolafc/2017_clubes.csv')
clubes_2017.head()


# In[ ]:


tabela_casa = partidas_2017[['rodada_id', 'clube_casa_id', 'clube_casa_posicao']]
tabela_fora = partidas_2017[['rodada_id', 'clube_visitante_id', 'clube_visitante_posicao']]


# In[ ]:


tabela_column_names = ['rodada_id', 'clube_id', 'posicao']
tabela_casa.columns = tabela_column_names
tabela_fora.columns = tabela_column_names


# In[ ]:


tabela = pd.concat([tabela_casa, tabela_fora])            .sort_values(['rodada_id', 'posicao'])            .reset_index(drop=True)


# In[ ]:


new_tabela = pd.merge(tabela, clubes_2017, how='left', left_on='clube_id', right_on='id')


# In[ ]:


new_tabela.head()


# In[ ]:


last_round = new_tabela['rodada_id'].max()
# getting a list of clube_id ordered by the positions on the last round
clube_id_list = new_tabela[new_tabela['rodada_id']==last_round]['clube_id'].values
tabela_grouped = new_tabela.groupby('clube_id')

plt.figure(figsize=(12, 8))
ax = plt.gca()
for clube_id in clube_id_list:
    nome = clubes_2017[clubes_2017['id']==clube_id]['nome'].values[0]
    group = tabela_grouped.get_group(clube_id)
    group.plot(x='rodada_id', y='posicao', ax=ax, label=nome)
plt.legend(bbox_to_anchor=(1.18, .99), labelspacing=1.1)
plt.xticks(range(1, last_round + 1))
plt.yticks(range(1, 21))
ax.invert_yaxis()


# In[ ]:




