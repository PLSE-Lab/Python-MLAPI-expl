#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from kaggle_environments.utils import structify
import argparse
import pprint
import json
import sys
#replace this with your submission.py file name if you are running locally
#import sub005 as sub
#or replace this with your actual agent 
def agent(obs,config):
    print('agent processing data')
    return {}
'''
Load a json eposode record and play a particular step with a given agent
as a given player.  Use to determine failure or debug actions on a particular step
of a previously played episode
'''
'''
Use these lines to make a command line processor:

parser = argparse.ArgumentParser(
    description="replay a json file",
    epilog='''
''')
parser.add_argument("file", type=str, help="json file to play",nargs='?',default='1233550.json')
parser.add_argument("--step", type=int, help="step to play",default=0)
parser.add_argument("--id", type=int, help="player to be",default=0)
options = parser.parse_args()
'''
def replay_match(path, step, playerid):
  with open(path, 'r') as f:
    match = json.load(f)
  env = make("halite", configuration=match['configuration'], steps=match['steps'])
  
  state2=match['steps'][step][0]   # list of length 1 for each step
  obs=state2['observation']  # these are observations at this step
  config=env.configuration
  obs['player']=playerid  # change the player to the one we want to inspect
  board=Board(obs,config)
  #check that we are correct player
  print('i am ', board.current_player_id, board.current_player)
  obs=structify(obs)   # turn the dict's into structures with attributes
  #This is our agent recreating what happened on this step
  #ret=sub.agent(obs, config)
  ret=agent(obs,config)
  print('returned from agent:',ret)
'''
replay_match(options.file, options.step, options.id)
'''
replay_match('../input/replay-file/1233550.json', 236, 3)

