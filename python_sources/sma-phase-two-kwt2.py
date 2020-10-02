#!/usr/bin/env python
# coding: utf-8

# lifecycle of a person
# - born with a certain wealth
# - buys a house immediately
# - dies and goes nowhere (probability of death) 
# 
# characterisitics of a person
# - age
# - net income
# - wealth (which changes with purchases and net income)
# - satisfaction (an indicator which cumulates)
# - probability of death
# 
# objective of a person
# - maximise satisfaction
# 
# objective of the government
# - maximise satisfaction of all the people
# 
# lifecycle of a house
# - empty initially
# - bought by a person
# - vacated when it is empty
# 
# characteristics of a house
# - location and coordinates 
# - occupancy
# - amentities
# 
# pricing mechanism
# - ???
# 
# process of purchase
# - each person only know a limited number of choices
# - each person will get the option that is 
#   - possible (house is empty and they have the money)
#   - best utility (x-factor for now)

# In[ ]:


import copy
import secrets  # python 3.6 necessary
import random
import numpy as np
import pandas as pd  # we try not to depend on pandas, to better translate later?
import matplotlib.pyplot as plt  # for viz

DEATH_RATE = 0.05
def generate_hex():
    return secrets.token_hex(4)


# # Initial state

# In[ ]:


def utility_function(amenities):
    # KIV: can add idiosyncracies
    return amenities["location"]

def generate_person():
    person = {
        "age": 20,
        "income": 10,
        "wealth": 400*np.random.uniform(),
        "housing_staying": None,
        "housing_selling": None,
        "utility": utility_function
    }
    return person


global persons
persons = {}
for _ in range(10):
    persons[generate_hex()] = generate_person()


# In[ ]:


global houses
houses = {}
for x in range(10):
    for y in range(10):
        # on init, houses have last_bought_price = market_price = value from rand expression
        houses[(x,y)] = {
            "last_bought_price": 400*np.random.uniform(),
            "status": "empty",  # "empty", "occupied", "selling" 
            "amenities": {"location" : np.random.uniform() + 1./ ((x-5.67)**2. + (y-5.43)**2.)},
            "occupant": None
        }
        # set market_price of house to be the same val as last_bought_price
        houses[(x,y)]["market_price"] = houses[(x,y)]["last_bought_price"]
        
def status_to_float(status):
    if status == "empty": return 0 
    if status == "occupied": return 1 
    if status == "selling": return 2


# # Activities in timestep
# In every time step, people age, people die, and people get born.

# ### Aging

# In[ ]:


def aging(verbose = False): # change this a function of age
    for person_id in persons:
        persons[person_id]["age"] += 1


# ### Death

# In[ ]:


def dying_prob_function(age):
    return 1./(1.+np.exp(-(0.04*(age-50))))
plt.figure(figsize = (14,2))
plt.plot([dying_prob_function(age) for age in np.arange(100)])
plt.title("death probability over age")
plt.show()


# In[ ]:


def dying(verbose = False): # change this a function of age
    persons_id_dead = []
    for person_id in persons:
        if np.random.uniform() < dying_prob_function(persons[person_id]["age"]):
            if verbose: print(person_id, " died")
            dead_person = persons[person_id]
            if "addr" in dead_person:
                if verbose: print("vacated ", dead_person["addr"])
                houses[dead_person["addr"]]["status"] = "empty"
                houses[dead_person["addr"]]["occupant"] = None
            persons_id_dead.append(person_id)
            
    for person_id_dead in persons_id_dead:
        del persons[person_id_dead]


# ### Birth

# In[ ]:


def birth(verbose = False):
    born = np.random.binomial(10, 0.2)
    for _ in range(born):
        persons[generate_hex()] = generate_person()


# # Transactions
# People without a house will try and buy the best house available to them.

# In[ ]:


def choose(person_id):
    # original
    candidates = []
    if "addr" in persons[person_id]:
        return None
    for h,house in houses.items():
        if house["price"] > persons[person_id]["wealth"]:
            continue
        if house["status"] != "empty":
            continue
        candidates.append((h,house))
    
    best = 0
    best_option = None
    for h,c in candidates:
        user_utility_on_house = persons[person_id]["utility"](c["amenities"])
        if user_utility_on_house > best:
            best = user_utility_on_house
            best_option = h,c
    return best_option


# In[ ]:


def allocation():
    # original
    for person_id,v in persons.items():
        decision = choose(person_id)
        if not decision:
            continue
        addr, house = decision
        persons[person_id]["wealth"] -= house["price"]
        persons[person_id]["addr"] = addr
        houses[addr]["status"] = "occupied"
        houses[addr]["occupant"] = person_id


# In[ ]:


# this is meant to be ran just once at the start
ask_df = pd.DataFrame(columns = ['house_pos','current_occupant_id','amenities', 'ask_price']) # init empty ask_df with col

def gen_empty_house_listing(house_pos, house_detail_dt):
    ''' Phase 2: bid-ask, used in gen_asks()
    Generates a listing dict for empty house.
    Input
    -----
    house_pos: (x,y), the key for each item in `houses`
    house_detail_dt: a house dict value from `houses`, containing key:value pairs
        "last_bought_price": 400*np.random.uniform(),
        "market_price": same as last_bought_price on init
        "status": "empty",  # "empty", "occupied", "selling" 
        "amenities": {"location" : np.random.uniform() + 1./ ((x-5.67)**2. + (y-5.43)**2.)},
        "occupant": None
    Output
    ------
    empty_house_listing_dt = 
        {'house_pos': (x,y),
        'current_occupant_id': hex id,
        'amenities': amenities value,
        'ask_price': num}
    '''
    empty_house_listing_dt = {'house_pos':house_pos,
                             'current_occupant_id': house_detail_dt['occupant'],
                             'amenities': house_detail_dt['amenities'], # TODO: check
                             'ask_price': house_detail_dt['last_bought_price'] # ask_price will be the initial-generated price of the empty house 
                             }
    return empty_house_listing_dt

def gen_listing_if_can_and_want_sell(person_id, person_dt):
    ''' Phase 2: bid-ask, used in gen_asks()
    1. Check if can sell
    2. Check if want to sell
    3. Generate listing
    
    Note:
    - ask_price is set to be market price for that house
    - this assumes that market price changes with each time step -- there should be a function for it later
    - also, a PROBA_SELL is defined here, set to 0.4 arbitrarily
    '''

    PROBA_SELL = 0.4 # arbitrary threshold
    
    # 1. Check if can sell
    house_pos_to_sell = person_dt['housing_selling']
    if house_pos_to_sell != None: # must have a second house to sell
        house_to_sell_dt = houses[house_pos_to_sell] # this is the house dict obj
        cost_price = house_to_sell_dt['last_bought_price']
        market_price = house_to_sell_dt['market_price']
        
        # 2. Check if want to sell
        if market_price >= cost_price: # makes sense to sell
            if np.random.uniform() <= PROBA_SELL: # random chance that person wants to sell given that it is sensible to do so
                assert person_id == house_to_sell_dt['occupant'], "ERROR: person_id != occupant_id of house being sold"
                
                # 3. Generate listing
                house_listing_dt = {'house_pos': house_pos_to_sell,
                             'current_occupant_id': house_to_sell_dt['occupant'],
                             'amenities': house_to_sell_dt['amenities'], # TODO: check
                             'ask_price': market_price # TODO: check & update if nec
                             }
                return house_listing_dt

def gen_asks():
    ''' phase 2 bid-ask
    1. Refresh ask_df pd.DataFrame()
    2. Add empty houses from `houses` to ask_df
    3. Add more listings from persons who can and want to sell houses
    '''
    global ask_df # may not be necessary
    # 1. Refresh ask_df pd.DataFrame()
    ask_df.drop(ask_df.index, inplace=True)
    
    # 2. Add empty houses from `houses` to ask_df
    empty_houses_ask_ls =     [gen_empty_house_listing(house_pos, house_detail_dt)      for house_pos, house_detail_dt in houses.items() if house_detail_dt['status']=='empty']
    
    ## convert ls of dt to df and append new df to main ask_df
    empty_houses_ask_df = pd.DataFrame(empty_houses_ask_ls)
    ask_df = ask_df.append(empty_houses_ask_df, ignore_index=True)
    
    # 3. Add more listings from persons who can and want to sell houses
    main_listing_ask_ls = [gen_listing_if_can_and_want_sell(person_id, person_dt) for person_id, person_dt in persons.items()]
    main_listing_ask_df = pd.DataFrame(main_listing_ask_ls)
    ask_df = ask_df.append(main_listing_ask_ls, ignore_index=True)

# test run
gen_asks() #works


# In[ ]:


#     person = {
#         "age": 20,
#         "income": 10,
#         "wealth": 400*np.random.uniform(),
#         "housing_staying": None,
#         "housing_selling": None,
#         "utility": utility_function
#     }


# In[ ]:


ask_df.sample(10)


# In[ ]:





# # Begin simulation

# In[ ]:


get_ipython().system('apt-get -y install ffmpeg > /dev/null')


# In[ ]:


from IPython.display import display, HTML
import matplotlib.animation as animation


# In[ ]:


from pprint import pprint

fig, ax = plt.subplots(1,3)
im0 = ax[0].imshow(np.random.randn(10,10), vmin=0, vmax=5)
im1 = ax[1].imshow(np.random.randn(10,10), vmin=0, vmax=500)
im2 = ax[2].imshow(np.random.randn(10,10), vmin=0, vmax=1)
ax[0].set_title("amenities")
ax[1].set_title("price")
ax[2].set_title("status")
patches = [im0, im1, im2]

def update_plot():
    xarr = np.random.randn(10,10)
    parr = np.random.randn(10,10)
    oarr = np.random.randn(10,10)
    for x in range(10):
        for y in range(10):
            xarr[x,y] = utility_function(houses[(x,y)]["amenities"])
            parr[x,y] = houses[(x,y)]["price"]
            oarr[x,y] = status_to_float(houses[(x,y)]["status"])
    im0.set_data(xarr)
    im1.set_data(parr)
    im2.set_data(oarr)

def init():
    return patches

def next_time_step(i):
    aging()
    birth()
    dying()
    allocation()
    update_plot()
    return patches

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, next_time_step, init_func=init,
                               frames=100, interval=100, blit=True)

vid = anim.to_html5_video()
plt.close()


# In[ ]:


HTML(vid)


# In[ ]:




