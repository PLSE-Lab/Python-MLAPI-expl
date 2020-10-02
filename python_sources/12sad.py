 # GO or ORIGINAL

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

your_attack_pokemon = 'Pikachu'
their_defense_pokemon = 'Bulbasaur'
mode = 'GO'

pokemon = pd.read_csv('../input/Pokemon.csv')
pokemon = pokemon[pokemon['#'] < 151 if mode == 'GO' else 10000]

your_attack_pokemon = 'Pikachu'
their_defense_pokemon = 'Bulbasaur'
mode = 'GO'

types = pd.read_csv(StringIO("""Attacking,Normal,Fire,Water,Electric,Grass,Ice,Fighting,Poison,Ground,Flying,Psychic,Bug,Rock,Ghost,Dragon,Dark,Steel,Fairy
Normal,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0,1,1,0.5,1
Fire,1,0.5,0.5,1,2,2,1,1,1,1,1,2,0.5,1,0.5,1,2,1
Water,1,2,0.5,1,0.5,1,1,1,2,1,1,1,2,1,0.5,1,1,1
Electric,1,1,2,0.5,0.5,1,1,1,0,2,1,1,1,1,0.5,1,1,1
Grass,1,0.5,2,1,0.5,1,1,0.5,2,0.5,1,0.5,2,1,0.5,1,0.5,1
Ice,1,0.5,0.5,1,2,0.5,1,1,2,2,1,1,1,1,2,1,0.5,1
Fighting,2,1,1,1,1,2,1,0.5,1,0.5,0.5,0.5,2,0,1,2,2,0.5
Poison,1,1,1,1,2,1,1,0.5,0.5,1,1,1,0.5,0.5,1,1,0,2
Ground,1,2,1,2,0.5,1,1,2,1,0,1,0.5,2,1,1,1,2,1
Flying,1,1,1,0.5,2,1,2,1,1,1,1,2,0.5,1,1,1,0.5,1
Psychic,1,1,1,1,1,1,2,2,1,1,0.5,1,1,1,1,0,0.5,1
Bug,1,0.5,1,1,2,1,0.5,0.5,1,0.5,2,1,1,0.5,1,2,0.5,0.5
Rock,1,2,1,1,1,2,0.5,1,0.5,2,1,2,1,1,1,1,0.5,1
Ghost,0,1,1,1,1,1,1,1,1,1,2,1,1,2,1,0.5,1,1
Dragon,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,0.5,0
Dark,1,1,1,1,1,1,0.5,1,1,1,2,1,1,2,1,0.5,1,0.5
Steel,1,0.5,0.5,0.5,1,2,1,1,1,1,1,1,2,1,1,1,0.5,2
Fairy,1,0.5,1,1,1,1,2,0.5,1,1,1,1,1,1,2,2,0.5,1"""))

pokemon_attack = pokemon.merge(types, left_on='Type 1', right_on='Attacking')

opponent_type = pokemon[pokemon['Name'] == their_defense_pokemon]['Type 1'].iloc[0]
opponent_multiplier = pokemon_attack[opponent_type]
adjusted_attack = pokemon_attack['Total'] * opponent_multiplier

pokemon_attack['Adjusted Attack'] = (adjusted_attack - adjusted_attack.min()) / (adjusted_attack.max() - adjusted_attack.min()) * 100

pokemon_attack.sort_values('Adjusted Attack', inplace=True)
pokemon_attack.tail(n=20).plot(kind='barh', x='Name', y='Adjusted Attack', figsize=(10, 7), title='Best 20 Pokemon to Attack %s' % their_defense_pokemon)