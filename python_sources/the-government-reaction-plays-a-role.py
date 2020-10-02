#!/usr/bin/env python
# coding: utf-8

# # The speed of government reaction affects the development of COVID-19

# ## Introduction
# The outbreak of the COVID-19 virus has given the world a heavy punch in 2020. The growth infection over 30 countries in the world has seen the overwhelming power of the exponential spread of COVID-19 at the early stage. While the pandemic increasingly causes damage in all aspects of people's life, no one knows how long it will last. With eventually realizing the serious situation, more and more countries start to take restrictions on people's social interactions. While the classic epidemic models succeed in predicting the exponential growth of infection, whether and to what extent the government's restriction on people's social life is effective are still hotly debated. Incorporating the external interference to the classic virus spread model adds complexity and therefore prevents analytic solutions. Hence, I present a spatially explicit social interaction model to simulate how coronavirus spreads among population, in which I consider the dynamic change of the government restriction on social activities according to how serious the pandemic becomes.   

# ## Goal
# The aim of this notebook is to introduce a social-interaction model on individual basis to measure how virus spreads under different speed of government reactions on social restriction. 
# 

# ## Approach
# A simulation model is designed. By setting a set of parameters in which the speed of the government's reaction and the virulence of the virus are of great interest, one can simulate how the infection grows in a given region. Comparing the results of different scenarios may hint at the relation between the epidemic development and the speed of government's reaction.    

# ## Model description
# 
# The model is an individual-based model. At each time step, a prescribed number (input as an argument `contactsize`) of people are sampled at a contact rate $C_{i,j}$ for each individual. The contact rate is computed for all pairs of the focal individual and the other individuals in the region. The contact rate is affected by multiple factors. First, it depends on the passions (willingness) to social activities ($Pas_i, Pas_j$) of the focal individual ($i$) and his meeting friend ($j$). If both are active to social interaction, their meeting is likely arranged, thus, leading to a high contact rate. In addition, the passion depends on individual's health state ($D_i$). If one is sick, he is not willing to go out meeting friends. The health states of all individuals are traced at each time step. Second, the contact rate is affected by the geographic distance ($d_{i,j}$) between the meeting friends. If they are far away from each other, they are not likely to meet. Third, due to the epidemic development the government restriction has influence on the contact rate. The intensity of the restriction is denoted as ($m(s,N_I)$), which is a function of the number of the infected cases ($N_I$) and the speed of reaction ($s$). Severe restriction is expected to be taken when more infected cases are confirmed, thus, reduces the contact rate to prevent virus spread. Following the intuition, the system is defined as
# \begin{align}
# C_{i,j} &= e^{-m(s,N_I)\cdot d^2_{i,j}} \cdot Pas_i(D_i) \cdot Pas_j(D_j)\\
# m(s,N_I) &= 2\frac{1}{1+e^{-s\cdot N_I/L^2}}-1\\
# Pas_i(D_i) &= e^{\frac{(D_i-IP)^2}{V^2}}
# \end{align}
# where $L^2$ denotes the total population size of consideration, $IP$ denotes the incubation period of the virus and $V$ denotes the virulence of the virus. 
# The health states are recorded in $D_k,k=1,\cdots,n$, which is related to time elapsed after getting infected. Thus, $D_k = 0$ denotes that the individual just gets infected. At each time step, $D_k$ is added by 1. After a recovery time period (input as an argument `recover_time`), the infected individual recovers to normal but again become susceptible to the virus. Thus, the healthy state is denoted by $D_i =$ `recover_time` which will not increase in values. This sets a upper bound to the passion function (Eq. 3) to prevent it going infinity. In the future, the mechanism of antibody of the virus that the recovered individuals have a lower chance to be reinfected can be implemented by modifying this function. Back to the model, in such a way recording the health state by elapsed days a dynamic passion to social activities based on one's health state is correctly captured. During the incubation period, the infected individual still has a certain degree of willingness to social interaction because of slight symptoms. After the incubation period, the state of illness of the individual becomes worse, which stops his outing. Eventually, the patient recovers from the illness and becomes active again. For simplicity, individual death is not modeled. The virulence ($1/V$) determines the sensitivity of the social passion to the symptom development. A large virulence (small $V$) denotes that the difference in social passions between states of illness and health is large (more convex curve in Figure 1) because severe illness substantially changes people's life style. A small virulence with a flat curve in Figure 1 accounts for little difference in social passions between illness and healthy state.      
# 
# 

# ## Simulation processes
# Before starting simulations, a number of parameters are to be initialized.
# A squared region is initialized by setting the width of the grid (`grid_size`). This also determines the total population size in this region. Each cell hosts one individual. To trigger the epidemic, a number (`initial_virus`) of infected individuals are launched. At each time step (day), each individual can meet a certain number (`contactsize`) of friends, which can be interpreted as the connectivity or vitality of the society. The virus is characterized by setting the recovery time length (`recover_time`), the incubation time length (`incubation`) and the inversed virulence (`virulence`). The speed of the government reaction is given by `speedreaction`. The simulation is implemented in parallel. By default, 4 cpu cores are used, which can be modified by setting `num_cores=x`.
# Once the parameters are set, one can check the curves of the following relations to get an idea of the model behavior.

# In[ ]:


# %%
import numpy as np
import itertools
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
from multiprocessing import Pool


class CoronaSim:
    def __init__(self, grid_size, initial_virus, recover_time, speedreaction, incubation, virulence, contactsize=1, num_cores=4):
        self.sim_grid = np.zeros(shape=[grid_size, grid_size])
        ini_x_virus = np.random.randint(
            low=0, high=grid_size, size=initial_virus)
        ini_y_virus = np.random.randint(
            low=0, high=grid_size, size=initial_virus)
        self.inistate_matrix = np.zeros(shape=[grid_size, grid_size])
        self.inistate_matrix.fill(float(recover_time))
        self.recover_time = recover_time
        self.inistate_matrix[ini_x_virus, ini_y_virus] = 7
        self.speedreaction = speedreaction
        self.incubation = incubation
        self.samplesize = contactsize
        self.virulence = virulence
        self.num_cores = num_cores
        self.all_sites = list(itertools.product(
            range(self.sim_grid.shape[0]), range(self.sim_grid.shape[0])))

    def mechanismcheck(self):
        state_value = np.arange(31)
        valuedf = pd.DataFrame(
            {'state': state_value, 'Activity': self.activity(state_value)})
        f1 = px.scatter(valuedf, x="state", y="Activity")
        f1.data[0].update(mode='markers+lines')
        f1.update_traces(line_color='#B54434',
                         marker_line_width=3, marker_size=4)

        distance = np.arange(200)
        disp = np.exp(-self.gm_virulence(20)*distance**2)
        contactdf = pd.DataFrame({'distance': distance, 'disp': disp})
        f2 = px.line(contactdf, x="distance", y="disp")
        f2.data[0].update(mode='markers+lines')
        f2.update_traces(line_color='#1B813E',
                         marker_line_width=3, marker_size=4)

        infected_num = np.arange(10000)
        measuredf = pd.DataFrame(
            {'infected_num': infected_num, 'measure': self.gm_virulence(infected_num)})
        f3 = px.line(measuredf, x="infected_num", y="measure")
        f3.update_traces(line_color='#66327C',
                         marker_line_width=3, marker_size=4)

        trace1 = f1['data'][0]
        trace2 = f2['data'][0]
        trace3 = f3['data'][0]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=False, subplot_titles=(
            "Figure 1", "Figure 2", "Figure 3"))
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)
        fig.add_trace(trace3, row=3, col=1)

        # Update xaxis properties
        fig.update_xaxes(title_text="Health state", row=1, col=1)
        fig.update_xaxes(title_text="Distance", range=[10, 50], row=2, col=1)
        fig.update_xaxes(title_text="The number of infected cases",
                         showgrid=False, row=3, col=1)

        # Update yaxis properties
        fig.update_yaxes(title_text="Willingness", row=1, col=1)
        fig.update_yaxes(title_text="Contact rate",
                         showgrid=False, row=2, col=1)
        fig.update_yaxes(
            title_text="Intensity of the restriction", row=3, col=1)

        # fig['layout'].update(height=800, width=800, showlegend=False)
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                ),
            ),
            xaxis2=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis2=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                ),
            ),
            xaxis3=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis3=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                ),
            ),
            autosize=True,

            plot_bgcolor='white',
            height=800, width=800,
        )
        fig.show()

    def activity(self, state):
        disp = np.exp((state-self.incubation) ** 2 /
                      self.virulence ** 2)
        return disp

    def gm_virulence(self, infected_num):
        return 100*(2/(1+np.exp(-infected_num*self.speedreaction/(self.sim_grid.shape[0]*self.sim_grid.shape[1])))-1)

    def spread_prob(self, x_row, y_col, state, seed=1):
        np.random.seed(seed)
        distance_sites = np.linalg.norm(
            np.array(self.all_sites) - np.array([x_row, y_col]), axis=1)
        Act = self.activity(state)
        gm_virulence = self.gm_virulence(
            infected_num=len(np.where(state < self.recover_time)[0]))
        prob_spread = np.exp(-gm_virulence *
                             distance_sites ** 2) * Act[x_row, y_col] * Act.flatten()
        prob_spread[x_row*self.sim_grid.shape[1]+y_col] = 0
        focal_state = np.random.choice(range(
            self.sim_grid.shape[0]*self.sim_grid.shape[1]), size=self.samplesize, p=prob_spread/sum(prob_spread))
        focal_state_value = 0 if min(state.flatten()[focal_state]) < self.recover_time else self.recover_time
        return focal_state_value

    def simspread(self, t_end, savefile):
        self.savefile = savefile
        state_matrix = self.inistate_matrix
        output_list = []
        parallel_cores = Pool(self.num_cores)
        for t in range(t_end):
            num_infected = len(np.where(state_matrix < self.recover_time)[0])
            print(
                f'At Day {t}, {num_infected} infected cases are confirmed...')
            healthy_individual_index_row = np.where(state_matrix >= self.recover_time)[0]
            healthy_individual_index_col = np.where(state_matrix >= self.recover_time)[1]
            change_state = parallel_cores.starmap(self.spread_prob,
                                                  zip(healthy_individual_index_row, healthy_individual_index_col, itertools.repeat(state_matrix)))
            state_matrix[healthy_individual_index_row,
                         healthy_individual_index_col] = change_state
            state_matrix += 1
            output_list.append(state_matrix.tolist())
        np.savez(self.savefile, *output_list)
        return state_matrix
    
if __name__ == "__main__":
    test = CoronaSim(grid_size=100, initial_virus=5, contactsize=2,num_cores=6,
                         recover_time=30, speedreaction=0.01, incubation=10, virulence=25)
    test.mechanismcheck()


# The first figure shows how active the individual under different health states is to social activities. When the individual is healthy ($D_i=30$), the highest willingness is assigned, meaning he is very willing to have social activities. When the individual gets infected but still under the incubation period ($D_i\in (0,10)$), the individual willingness to have social activities is low. However, because only slight symptoms appear the individual still has a degree of willingness to go out. With the symptoms become severe, the individual prefers stay at home with lowest contact willingness to others. Again, when the individual eventually recovers from the illness, his willingness recovers as well. Instead of Eq. 3, functions of other forms that show different mechanisms can be applied. 
# The second figure shows the curve of the contact level only against geographic distance ($e^{-m(N_I)\cdot d^2_{i,j}}$) when there are ($N_I=20$) infected cases given the speed of government reaction (`speedreaction`). It reflects the spatial contact spirit of the model that people that are far away from each other are not likely to meet.
# The third figure shows how the intensity of government restriction changes with the increase of the infected cases. Intuitively, with increase of the number of the infected cases, governments are likely to take strict restrictions.
# Now, we can start simulations by setting the number of time steps (`t_end`) and the output file name (`savefile`) which is in a `.npz` format. 

# In[ ]:


# Start running simulations
result = test.simspread(t_end=10, savefile='test.npz')


# ## Experiment setup
# Direct to our goal, I want to assess the influence of the government reaction speed on regional pandemic development.Thus, as an illustration, I designed three scenarios with three values to `speedreaction`, i.e. $0.01,0.1,1$, which account for slow, moderate and fast reaction of the government to take restriction. All the other initial parameters are set the same as follows

# In[ ]:


# Simulation setup
scenario1 = CoronaSim(grid_size=100, initial_virus=5, contactsize=2, num_cores=6,
                     recover_time=30, speedreaction=0.01, incubation=7, virulence=25)


# ## Simulation results
# 
# From the figure below, a large proportion of population is infected when the government reacts slowly (the red growth curve). A plateau is reached because that the government starts to take strict restrictions only when too many individuals are infected. In contrast, only a small number of people get infected when the government takes fast response to the pandemic (the green curve). The result indicates that a fast response of the government to pandemic plays a positive role in inhibiting virus spread.
# We also see a second growth phase in the accumulation curve. This is due to that the recovered individuals get reinfected. Implementing immune system to the model can resolve this (probably) unrealistic issue.  

# In[ ]:


# %%
import plotly.graph_objects as go
import numpy as np
import pandas as pd

num_infected = []
Day = []
batch_list = []
for batch in range(1, 4):
    savefile = f'../input/simulation-scripts/outfile_s{batch}.npz'
    container = np.load(savefile)
    sim_result = [container[key] for key in container]
    for t in range(len(sim_result)):
        num_infected.append(len(np.where(sim_result[t] < 30)[0]))
    Day.extend(np.arange(len(sim_result)).tolist())
    batch_list.extend(np.repeat(batch, len(sim_result)))

infected_growth_df = pd.DataFrame(
    {'num_infected': num_infected, 'Day': Day, 'batch': batch_list})

# %%


# Add data

fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 1].Day, y=infected_growth_df[infected_growth_df['batch'] == 1].num_infected, name='Speed 0.01',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 2].Day, y=infected_growth_df[infected_growth_df['batch'] == 2].num_infected, name='Speed 0.1',
                         line=dict(color='royalblue', width=4,
                                   dash='dot')))
fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 3].Day, y=infected_growth_df[infected_growth_df['batch'] == 3].num_infected, name='Speed 1',
                         line=dict(color='green', width=4,
                                   dash='dash')  # dash options include 'dash', 'dot', and 'dashdot'
                         ))

# Edit the layout
fig.update_layout(title='The influence of government reaction speed on the pandemic development',
                  xaxis_title='Day',
                  yaxis_title='Number of infected cases',
                  xaxis=dict(
                        showline=True,
                        showgrid=False,
                        showticklabels=True,
                        linecolor='rgb(204, 204, 204)',
                        linewidth=2,
                        ticks='outside',
                        tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                        ),
                  ),
                  yaxis=dict(
                      showline=True,
                      showgrid=False,
                      showticklabels=True,
                      linecolor='rgb(204, 204, 204)',
                      linewidth=2,
                      ticks='outside',
                      tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                      ),
                  ),
                  autosize=True,
                  plot_bgcolor='white',
                  height=600, width=800
                  )

fig.show()

# %%


# The number of newly increase of the infected cases is shown below. Again, under fast restriction the curve is more flatten than those of other speeds. 

# In[ ]:


# %%
import plotly.graph_objects as go
import numpy as np
import pandas as pd

num_infected = []
Day = []
batch_list = []
for batch in range(1, 4):
    savefile = f'../input/simulation-scripts/outfile_s{batch}.npz'
    container = np.load(savefile)
    sim_result = [container[key] for key in container]
    acc_list = []
    for t in range(1,len(sim_result)):
        acc_list.append(len(np.where(sim_result[t] < 30)[0])-len(np.where(sim_result[t-1] < 30)[0]))
    num_infected.extend(acc_list)
    Day.extend(np.arange(len(sim_result)-1).tolist())
    batch_list.extend(np.repeat(batch, len(sim_result)-1))

infected_growth_df = pd.DataFrame(
    {'num_infected': num_infected, 'Day': Day, 'batch': batch_list})

# %%


# Add data

fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 1].Day, y=infected_growth_df[infected_growth_df['batch'] == 1].num_infected, name='Speed 0.01',
                         line=dict(color='firebrick', width=4),fill='tozeroy'))
fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 2].Day, y=infected_growth_df[infected_growth_df['batch'] == 2].num_infected, name='Speed 0.1',
                         line=dict(color='royalblue', width=4,
                                   dash='dot'),fill='tozeroy'))
fig.add_trace(go.Scatter(x=infected_growth_df[infected_growth_df['batch'] == 3].Day, y=infected_growth_df[infected_growth_df['batch'] == 3].num_infected, name='Speed 1',
                         line=dict(color='green', width=4,
                                   dash='dash'),  # dash options include 'dash', 'dot', and 'dashdot'
                         fill='tozeroy'))

# Edit the layout
fig.update_layout(title='',
                  xaxis_title='Day',
                  yaxis_title='Number of newly increase infected cases',
                  xaxis=dict(
                        showline=True,
                        showgrid=False,
                        showticklabels=True,
                        linecolor='rgb(204, 204, 204)',
                        linewidth=2,
                        ticks='outside',
                        tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                        ),
                  ),
                  yaxis=dict(
                      showline=True,
                      showgrid=False,
                      showticklabels=True,
                      linecolor='rgb(204, 204, 204)',
                      linewidth=2,
                      ticks='outside',
                      tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                      ),
                  ),
                  autosize=True,

                  plot_bgcolor='white',
                  height=600, width=800,
                  )

fig.show()

# %%


# ## Application
# To predict the cumulative curve of the infected cases under this model, multiple ways can be exploited. The first one is to develop an analytic likelihood formulation to the model. However, due to the complexity of the system, it is hard to achieve to my knowledge so far. Thus, some likelihood-free methods are naturally alternative avenue. 
# ### Approximate Bayesian Computation (ABC)
# Approximate Bayesian computation (ABC) is a kind of likelihood-free method that utilizes computational power to generate a huge amount of simulations with randomly chosen parameters to hit the target - the observations. The generating parameters that produce the results that are most analogous to the observations are chosen as the best inference of the true generating parameters. A simple version of ABC program can be found on my [blog](https://xl0418.github.io/2020/03/18/2020-03-18-generalABC/#more). But it is impossible to use in this notebook and in this competition because this individual-based simulation study together with ABC inference requires a high performance computer cluster.  
# ### Deep Learning 
# Deep Learning built upon multiple neural layers has advantage on recognizing character of complex data patterns. Thus, one can use the model to generate a number of data sets by setting different parameters and label the data sets for training the neural networks. Once the network is trained, people can feed the empirical data to obtain estimation of the parameters. Again, to obtain accurate parameter estimation, one needs to generate as many simulation as possible to cover the unknown parameter space. So, I am not ganna apply it here but throw out this proposal and leave it for future exercises.
# 
# ### Summary statistics
# Both methods mentioned above ask for summary statistics to measure the similarity between the simulated data and the empirical data. Here, I present one summary statistic that circumvents the obstacle of the difference between the real population size and the simulation limit. The summary statistic is defined by a vector of the ratios of the newly increasing number of the infected cases and the number of the infected cases on the previous day, $ip = (ip(t_0),ip(t_1),\cdots,ip(t_{end}-1))$, in which the elements yielding:
# \begin{align}
# ip_{t} = \frac{N_I(t+1)-N_I(t)}{N_I(t)}.
# \end{align}
# In such a way, at the early stage of the simulation when the grid size is relatively large enough so that it doesn't limit the development of the pandemic the growth pattern of the infected cases is comparable to the reality. The following figure shows the $ip$ patterns across all countries in the data set.

# In[ ]:


import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np

datafile = '../input/covid19-global-forecasting-week-2/train.csv'
data = pd.read_csv(datafile)
data['PSCR'] = data.Province_State.map(str)+ '' + data.Country_Region.map(str)

region = pd.unique(data['PSCR']).tolist()
f_region = []
time_list = []
region_name = []
for ci in range(len(region)):
    region_data = data[data['PSCR'] == region[ci]]
    region_data = region_data[region_data.ConfirmedCases > 0]
    inc_percentage = (region_data.ConfirmedCases[1:].to_numpy(
    )-region_data.ConfirmedCases[:-1].to_numpy())/region_data.ConfirmedCases[:-1].to_numpy()
    # Only considering the countries with effective data
    if len(np.where(inc_percentage > 0)[0]) > 0:
        inc_percentage = inc_percentage[np.where(inc_percentage > 0)[0][0]:]
        f_region.extend(inc_percentage)
        time_list.extend([i for i in range(len(inc_percentage))])
        region_name.extend([region[ci] for i in range(len(inc_percentage))])
    else:
        pass
f_df = pd.DataFrame(
    {'increase': f_region, 'Day': time_list, 'region': region_name})

fig = px.line(f_df, x='Day',
              y='increase', color='region')
fig.update_layout(title='ip patterns',
                  xaxis_title='Day',
                  yaxis_title='Increasing percentage',
                  xaxis=dict(
                        showline=True,
                        showgrid=False,
                        showticklabels=True,
                        linecolor='rgb(204, 204, 204)',
                        linewidth=2,
                        ticks='outside',
                        tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                        ),
                  ),
                  yaxis=dict(
                      showline=True,
                      showgrid=False,
                      showticklabels=True,
                      linecolor='rgb(204, 204, 204)',
                      linewidth=2,
                      ticks='outside',
                      tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                      ),
                  ),
                  autosize=True,

                  plot_bgcolor='white',
                  height=600, width=800,
                  )
fig.show()


# ## Case studies
# Till now, I have presented a theoretical framework to study the influence of the speed of the government reaction on the pandemic development. Due to the computational limit of my desktop, I just present some case studies to compare the $ip$ patterns of three simulation results and some countries data. The choosing criteria is determined by the model limitation and assumptions. For example, I haven't implemented individual death and the immune system. Therefore, we better chose the data of small countries and small number of fatality cases. Also, due to the lack of immune system, recovered individuals can be reinfected in the current model. Thus, we should focus on the curve character at the early stage (in the first 30 days). 

# ### Comparing $ip$ between Japan and Israel
# According to the criteria mentioned above, I chose Japan and Israel as an illustration.

# In[ ]:


import plotly.express as px
import pandas as pd
import numpy as np

datafile = '../input/covid19-global-forecasting-week-2/train.csv'
data = pd.read_csv(datafile)

# %%
all_region_data = data[pd.isna(data['Province_State'])]
region = ['Japan', 'Israel']
# region = pd.unique(all_region_data['Country_Region']).tolist()
f_region = []
time_list = []
region_name = []
for ci in range(len(region)):
    region_data = data[data['Country_Region'] == region[ci]]
    region_data = region_data[region_data.ConfirmedCases > 0]
    inc_percentage = (region_data.ConfirmedCases[1:].to_numpy(
    )-region_data.ConfirmedCases[:-1].to_numpy())/region_data.ConfirmedCases[:-1].to_numpy()
    # Only considering the countries with effective data
    if len(np.where(inc_percentage > 0)[0]) > 0:
        inc_percentage = inc_percentage[np.where(inc_percentage > 0)[0][0]:]
        f_region.extend(inc_percentage)
        time_list.extend([i for i in range(len(inc_percentage))])
        region_name.extend([region[ci] for i in range(len(inc_percentage))])
    else:
        pass
f_df = pd.DataFrame(
    {'increase': f_region, 'Day': time_list, 'region': region_name})


# %%
sim_data = []
speed = [0.01,0.1,1]
for batch in range(1,4):
    result = f'../input/simulation-scripts/outfile_s{batch}.npz'
    container = np.load(result)
    speed_batch = f'Sim: speed {speed[batch-1]}'

    sim_result = [container[key] for key in container]
    num_infected = []
    for t in range(len(sim_result)):
        num_infected.append(len(np.where(sim_result[t] < 30)[0]))

    inc_infected = [(num_infected[i+1]-num_infected[i])/num_infected[i]
                    for i in range(len(num_infected)-1)]
    infected_growth_df = pd.DataFrame({'increase': inc_infected, 'Day': [
        i for i in range(len(sim_result)-1)], 'region': speed_batch})
    sim_data.append(infected_growth_df)
sim_df = pd.concat(sim_data)
# %%
newf = f_df.append(sim_df)

# %%
fig = px.line(newf, x='Day',
              y='increase', color='region')
fig.update_layout(title='ip patterns of Japan and Israel against 3 simulations',
                  xaxis_title='Day',
                  yaxis_title='Increasing percentage',
                  xaxis=dict(
                        showline=True,
                        showgrid=False,
                        showticklabels=True,
                        linecolor='rgb(204, 204, 204)',
                        linewidth=2,
                        ticks='outside',
                        tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                        ),
                  ),
                  yaxis=dict(
                      showline=True,
                      showgrid=False,
                      showticklabels=True,
                      linecolor='rgb(204, 204, 204)',
                      linewidth=2,
                      ticks='outside',
                      tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                      ),
                  ),
                  autosize=True,

                  plot_bgcolor='white',
                  height=400, width=600,
                  )

fig.show()


# To clearly compare with the simulation curves, please only **keep one simulation curve by shutting off the other two (clicking on the legend)**. Apparently, The $ip$ pattern of Japan is quite consistent with the simulation with speed $1$, which indicates Japanese government took relatively fast reaction to the pandemic that effectively reduces the cumulation at the later phase. In contrast, the $ip$ curve of Israel shows a flat shape at the early stage that is close to the simulation with speed $0.01$, indicating a slower reaction to take quarantine measures compared to Japan.  
# But as I mentioned, this model is built on several simplifications like homogeneous individual dispersal and limited population size. Thus, making conclusions based on current data of countries is not wise. Further extensions of this model may achieve that goal but this one. This study just indicates a fast response to the urgent pandemic is helpfull to inhibit epidemic development.

# ## Future plan
# This model can be used to measure the behavior of governments in reality. One can compute summary statistics like the slope of the growth at a certain phase to compare the empirical data with the simulated data. In such a way, the speed of government reaction can be estimated. Furthermore, as the information of how virus spread regionally is stored, one can investigate the spread mechanism of the virus (see the heatmap below).  
# The model has great potential to be extended to a more realistic version. As mentioned above, one can implement immune system in the model to see whether herd immunity is achieved and to what extent it helps stop virus spread. In addition, whether wearing masks can help prevent virus spread can be implemented by adding a probability to get infected when meeting an infected individual. 

# In[ ]:


# %%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class plotresult:
    def __init__(self, savefile):
        container = np.load(savefile)
        self.sim_result = [container[key] for key in container]

    def infectiongrowth(self):
        num_infected = []
        for t in range(len(self.sim_result)):
            num_infected.append(len(np.where(self.sim_result[t] < 30)[0]))
        infected_growth_df = pd.DataFrame({'num_infected': num_infected, 'Day': [
                                          i for i in range(len(self.sim_result))]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=infected_growth_df.Day, y=infected_growth_df['num_infected'], name="AAPL High",
                                 line_color='deepskyblue'))

        fig.update_layout(title_text='Infection growth',
                          xaxis_rangeslider_visible=True)
        fig.show()

    def infectionheatmap(self):
        infect_dis = []
        col = []
        row = []
        days = []
        for t in range(len(self.sim_result)):
            temp_re = self.sim_result[t].tolist()
            flatten_re = [item for sublist in temp_re for item in sublist]
            x_co = np.tile(range(len(temp_re)), len(temp_re))
            y_co = np.repeat(range(len(temp_re)), len(temp_re))
            day_series = np.repeat(t, len(temp_re)**2)

            infect_dis.extend(flatten_re)
            col.extend(x_co)
            row.extend(y_co)
            days.extend(day_series)

        heatmapdf = pd.DataFrame(
            {'dis': infect_dis, 'Day': days, 'col': col, 'row': row})
        fig = px.scatter(heatmapdf, x="col", y="row", color='dis', animation_frame="Day",
                         color_continuous_scale=[(0, "#81C7D4"), (0.2, "#D0104C"), (1, "#81C7D4")])
        fig.update_layout(title='The pandemic development',
                          xaxis_title='',
                          yaxis_title='',
                          xaxis=dict(
                              showline=False,
                              showgrid=False,
                              showticklabels=False,
                          ),
                          yaxis=dict(
                              showline=False,
                              showgrid=False,
                              showticklabels=False,
                          ),
                          autosize=True,
                          plot_bgcolor='white',
                          height=600, width=600,
                          coloraxis_colorbar=dict(
                              title="Healthy state"
                          )
                          )

        fig.show()


        # %%
if __name__ == "__main__":
    result = '../input/simulation-scripts/outfile_s1.npz'
    testplot = plotresult(result)
    # testplot.infectiongrowth()
    testplot.infectionheatmap()

# %%


# ## Prediction
# As aforementioned, this model currently is not ready to make prediction because of highly computational demanding. However, to accomplish this competition, I would like to take an expedient. I classify the data sets of different countries to the three simulation scenarios, i.e. a strategy of slow speed of reaction, of a moderate speed of reaction and of a fast speed of reaction. 

# In[ ]:


import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np

datafile = '../input/covid19-global-forecasting-week-2/train.csv'
data = pd.read_csv(datafile)
data['PSCR'] = data.Province_State.map(str)+data.Country_Region.map(str)

# %%
# ip pattern of the empirical data from 2020/03/19 onwards
region = pd.unique(data['PSCR']).tolist()
f_region = []
time_list = []
region_name = []
actual_date = []
no_infection_country = []
for ci in range(len(region)):
    region_data = data[data['PSCR'] == region[ci]]
    region_data = region_data[region_data.ConfirmedCases > 0]
    inc_percentage = (region_data.ConfirmedCases[1:].to_numpy(
    )-region_data.ConfirmedCases[:-1].to_numpy())/region_data.ConfirmedCases[:-1].to_numpy()
    # Only considering the countries with effective data
    if len(np.where(inc_percentage > 0)[0]) > 0:
        inc_percentage = inc_percentage[np.where(inc_percentage > 0)[0][0]:]
        actual_date.append(region_data.Date[1:])
        f_region.extend(inc_percentage)
        time_list.extend([i for i in range(len(inc_percentage))])
        region_name.extend([region[ci] for i in range(len(inc_percentage))])
    else:
        no_infection_country.append(region[ci])
f_df = pd.DataFrame(
    {'increase': f_region, 'Day': time_list, 'PSCR': region_name})


# %%
# Simulation data for training
sim_data = []
speed = [0.01,0.1,1]
for batch in range(1,4):
    result = f'../input/simulation-scripts/outfile_s{batch}.npz'
    container = np.load(result)
    speed_batch = f'Sim: speed {speed[batch-1]}'

    sim_result = [container[key] for key in container]
    num_infected = []
    for t in range(len(sim_result)):
        num_infected.append(len(np.where(sim_result[t] < 30)[0]))

    inc_infected = [(num_infected[i+1]-num_infected[i])/num_infected[i]
                    for i in range(len(num_infected)-1)]
    infected_growth_df = pd.DataFrame({'increase': inc_infected, 'Day': [
        i for i in range(len(sim_result)-1)], 'PSCR': speed_batch})
    sim_data.append(infected_growth_df)
sim_df = pd.concat(sim_data)

# %%
criteria_day_length = 10
sim_class_ip = []
for speed in pd.unique(sim_df.PSCR):
    sim_class_ip.append(sim_df[sim_df['PSCR'] == speed].increase.tolist())
sim_class_ip_array = np.array(sim_class_ip)

#%%
labels = []
effective_region = []
for region_loop in region:
    if region_loop not in no_infection_country:
        ip = f_df[f_df['PSCR'] == region_loop].increase[:criteria_day_length].tolist()
        euclidean_dis = np.linalg.norm(np.array(ip)-sim_class_ip_array[:,:len(ip)],axis = 1)
        labels.append(np.where(euclidean_dis == min(euclidean_dis))[0][0])
        effective_region.append(region_loop)
    else:
        pass

xlabels = ['Slow','Moderate','Fast']
scenario_class = {'ip': [xlabels[i] for i in labels], 'Area':effective_region, 'width': [1 for i in range(len(labels))]}
sce_df = pd.DataFrame(scenario_class)
#%%
fig = px.bar(sce_df, x="ip", y="width", color='Area', height=400)
fig.update_layout(title='Strategies of regions',
                  xaxis_title='Strategy',
                  yaxis_title='Areas and regions',
                  xaxis=dict(
                        showline=True,
                        showgrid=False,
                        showticklabels=True,
                        linecolor='rgb(204, 204, 204)',
                        linewidth=2,
                        ticks='outside',
                        tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                        )
                  ),
                  yaxis=dict(
                      showline=True,
                      showgrid=False,
                      showticklabels=True,
                      linecolor='rgb(204, 204, 204)',
                      linewidth=2,
                      ticks='outside',
                      tickfont=dict(
                          family='Arial',
                          size=12,
                          color='rgb(82, 82, 82)',
                      ),
                  ),
                  autosize=True,
                  plot_bgcolor='white',
                  height=600, width=800,
                  )
fig.show()


# Subsequently, I use the simulation $ip$ patterns of different strategies to make predictions. However, in this model I excluded the fatality. Thus, there is no forecasting for the number of the fatalities.

# In[ ]:


# Using the data on 18 Mar to calculate the tendency of the pandemic.
date_datause = '2020-03-18'
date_actualdata = '2020-03-30'
date_length = (pd.to_datetime(date_actualdata) - pd.to_datetime(date_datause)).days
predict_region_list = []
effect_ind = 0
for it in range(len(region)):
    region_it = region[it]
    if region_it not in no_infection_country:
        time_length_it = actual_date[effect_ind]
        sim_class_it = labels[effect_ind]
        predict_ip_it = sim_class_ip_array[sim_class_it,(len(actual_date[0])-date_length):]
        while len(predict_ip_it)< (date_length+31):
            predict_ip_it = np.append(predict_ip_it,predict_ip_it[len(predict_ip_it)-1])
        retion_df = data[data['PSCR'] == region_it]
        num_infected_it = retion_df[retion_df['Date'] == date_datause]['ConfirmedCases'].astype(float)
        predict_region_list_it = []
        ini_infected = num_infected_it.tolist()[0]
        for predict_day in range(len(predict_ip_it)):
            predict_region_list_it.append(ini_infected * (1+predict_ip_it[predict_day]))
            ini_infected = predict_region_list_it[predict_day]
        predict_region_list.extend(predict_region_list_it)
        effect_ind += 1
    else:
        predict_region_list.extend([0 for i in range(43)])

# %%
# Write output csv file
import csv
from itertools import zip_longest
list1 = [i+1 for i in range(len(predict_region_list))]
list2 = predict_region_list
list3 = [0 for i in range(len(predict_region_list))]
d = [list1, list2,list3]
export_data = zip_longest(*d, fillvalue = '')
with open('submission.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("ForecastId", "ConfirmedCases", "Fatalities"))
      wr.writerows(export_data)
myfile.close()


# ## Statement 
# - Again, the purpose of this notebook is to introduce one angle of studying how the government's reaction influences pandemic development. All predictions and classifications of government strategies are unreliable, which need more time to investigate instead of making conclusions in such short time. 
# - The method introduced to make prediction of further trend of pandemic is just an expedient for accomplishing this competition.More accurate inference methods require a huge amount of time to explore the likelihood in large parameter space.  
