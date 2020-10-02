###############################################################################
################## Coro patient prediction 2.0__in_Denmark__ ##################

#############################################
#*Please download and use Spyder to run it.*#
#############################################


# The prediction is achieved by applying the Geometric series, along with a 
# recrusive algorithm. In this case, each person who get infected by this virus,
# will go through a 8 days period (T_period):(referred by thesis in 'Lancet')
#                                 3 days generate virus inside their body (T_delay)
#                                 5 days spread to other people (T_infect)
# Then after this 8-days period, this patient will go to hospital and no longer
# be a infection source.

###############################################################################

# Assumption: [1]. You might meet ['nr_people_meet_a_day'] diffreent people in a day.
#             [2]. You have ['percent_close_talking'*100%] talk to them closely.
#             [3]. During those closely talk, you have ['percent_infect'*100%] infect them.
#             [4]. Values assumed above is in line 40-43, and of course you can 
#                  tune them to see how these values influence the final prediction.
# Anyway, you will see how sensitive this final prediction when you tune these hyparameters

# In[1].Infection prediction 
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Data preparation
Base = '2020-02-24'
NOW =  '2020-03-10'

# Get information from the thesis in 'Lancet'
T_incu = 8                     # Incubation period, [Day]
T_period = T_incu               # How many days that one person go into the hospital after getting infection, [Day]
T_delay = 3                     # How many days can be infectious after getting the virus, [Day]
T_infect = T_period - T_delay   # How many days that one patient can be infectious, [Day]

###############################################################################
# Hyperparameters that can be tuned for this model.
# The hyperparameters below are very sensitive in the final prediction.
nr_patient_zero = 1             # Patient zero
nr_people_meet_a_day = 50       # How many people you meet roughly in a day
percent_close_talking = 0.2     # The probability of talking closely
percent_infect = 0.15           # The probability of infection in this close-talk.

# infection coe [q]
# This ratio is only valid until the city get blocked.
q = nr_people_meet_a_day*percent_close_talking*percent_infect

# Date ==> how many dates away from the 'Base'
def date_transfer(date, Base):
    
    sec_for_a_day = 24*60*60   # How many secs there ae for a day

    # Transfer the base to sec
    Base = datetime.strptime(Base, '%Y-%m-%d')
    un_time_Base = time.mktime(Base.timetuple())
    
    # Transfer the date to sec
    date = datetime.strptime(date, '%Y-%m-%d')
    un_time_date = time.mktime(date.timetuple())
    day_from_base = (un_time_date - un_time_Base)/sec_for_a_day
    
    return day_from_base


###############################################################################
# Weight & ratio initialization
def weight_initialize(n2, q, nr_patient_zero):
    weight = {}
    ratio = {}
    length_1 = T_infect + (T_period - T_infect + 1)*(1 - 1)
    weight['layer' + str(1)] = np.ones(length_1)*nr_patient_zero
    ratio['layer' + str(1)] = np.ones(length_1)*q
    for i in range(2, n2 + 1):
        length = T_infect + (T_period - T_infect + 1)*(i - 1)
        weight['layer' + str(i)] = np.zeros(length)
        ratio['layer' + str(i)] = np.ones(length)*np.power(q, i)
    
    return weight, ratio


#####################################################
## Here is the fucking critical part of this model ## 
#####################################################
# Weights calculation basing on the model assumption
# The mathematical theory behind this [unique] structure, can be found in 'READ_ME', 
# or you can ask me by yourself directly. Actually I prefer the later one, since do
# the explaination in right here will be another torment for me.
def weight_calculate(weight, n2):
    for j in range(2, n2 + 1):
        for k in range(len(weight['layer' + str(j)])):
            if k <= T_infect-1:
                for ii in range(k+1):
                    weight['layer' + str(j)][k] += weight['layer' + str(j-1)][ii]
            else:
                if k >= len(weight['layer' + str(j-1)]):
                    weight['layer' + str(j)][k] = weight['layer' + str(j)][k-1] - \
                    weight['layer' + str(j-1)][k-T_infect]
                else:
                    weight['layer' + str(j)][k] = weight['layer' + str(j)][k-1] - \
                    weight['layer' + str(j-1)][k-T_infect] + weight['layer' + str(j-1)][k]
                    
    return weight


#####################
# Weights calculation
# [1].Information preparation
day_from_base = int(date_transfer(NOW, Base))  # How many days away from 'Base'. 
observing_starting_point = 1
time_series = range(observing_starting_point, day_from_base+1)
N2 = (time_series[-1] - 1)//T_delay                # How many layers does this time-series finally have

# [2].Calculate the geometric series                  
weight, ratio = weight_initialize(N2, q, nr_patient_zero)       
weight_calculated = weight_calculate(weight, N2)
multiply = {}
for i in range(1, N2 + 1):
    multiply['layer' + str(i)] = weight['layer' + str(i)]*ratio['layer' + str(i)]


###########################
# The infection prediction
infection = []
for t in time_series:
    
    n1 = (t - 1)//(T_period - 1)    # How many layers have been finished in this time-piece
    n2 = (t - 1)//T_delay           # How many layers have been started  in this time-piece
        
    # Calculate the layers information
    starting_day = np.zeros(n2)
    length_of_each_layer = np.zeros(n2)
    for i in range(n2):
        
        # The starting day of each layer
        starting_day[i] = T_delay*(i+1) + 1

        # How many days this layer stands for
        length_of_each_layer[i] = T_infect + (T_period - T_infect + 1)*(i-1+1)  
        
    
    strated_length_for_unfinished_layers = np.zeros(n2, dtype = int)
    for j in range(n1, n2):
        strated_length_for_unfinished_layers[j] = int(t - starting_day[j] + 1) 
    
    # Now calculate the how many people get infected until 'NOW'
    # Initialize the infected people number by the 'number of patient zero'
    people_get_infected = nr_patient_zero
    
    # First of all, calculate the layers who are already finished
    for k in range(n1):
        people_get_infected += np.sum(multiply['layer' + str(k+1)])
    
    # Then, plus those layers who have not finished
    for kk in range(n1+1, n2+1):
        people_get_infected += np.sum(multiply['layer' + str(kk)][:strated_length_for_unfinished_layers[kk-1]])
    
    infection.append(people_get_infected)


R0 = q*T_infect
print('')
print('Since the danish patient_zero came back from Italy in %s.' % Base)
print('So the infection prediction is made from %s, to %s(right now).' % (Base, NOW))
print('')
print('PREDICTION:')
print('R0 value: %.3f' % R0, str('per person'))
print('There might be #%d# people got infected until %s' % (int(infection[-1]), NOW))


# In[2].Infection dynamic plot
# PLot as a gif
location = 'Denmark'
plt.ion()
plt.figure(figsize=(12,8))
plt.title('Infection through time, when R0 = %.3f, and %d patient_zero' % (R0, nr_patient_zero), fontsize = 20)
plt.xlabel('Days away from %s' % Base, fontsize = 23)
plt.ylabel('Infection population in %s' % location, fontsize = 23)
t_list = []
result_list = []
t = 0
while t < len(time_series):
    t_list = time_series[t]
    result_list = infection[t]
    plt.plot(t_list, result_list,c='r',ls='-', marker='o', mec='r',mfc='w')
    #plt.plot(t, np.sin(t), 'o')
    t+=1
    plt.pause(0.1)
    
plt.plot(time_series, infection)
plt.vlines(time_series[-1], 0, (infection[-1] + 10), colors = "k", linestyles = "dashed")
