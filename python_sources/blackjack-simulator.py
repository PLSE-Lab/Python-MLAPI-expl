#!/usr/bin/env python
# coding: utf-8

# # Blackjack Simulation 
# 
# In my quest for knowledge I found an interesting discusion topic here: [Blackjack Simulators](https://www.kaggle.com/learn-forum/89452). Taking this and running with it (I am an avid blackjack player and spent the last week in Vegas), I want to create a blackjack simulator with the goal of finding out what the best moves are to win the most money. To do this, I will follow these steps: 
# * Create a series of classes that will represent the player and the deck. 
# * Transfer the standard ruleset for the game
# * Apply the standard ruleset to a series of hands 
# * Analyze the accuracy of the ruleset 
# 
# For more avid blackjack players, I am going to be using the following terms: 
# * Game: One cycle through the deck 
# * Round: One cycle through the players 
# * Hand: One instance of the player competeing against the dealer

# ## Step 1 - The Deck 
# The deck of cards consisst of four suits -- Spades, Clubs, Hearts, and Diamonds. Each suit has its own Ace (1 or 11), nine numbered cards (2 through 10), and three face cards (the Jack, the Queen, the King, all valued at 10); there are 13 cards per suit, with a total of 52 cards. 
# 
# In many playing enviornments (barring your home enviornment) more than one deck is used in a game. This is to limit the player's ability to 'count' the deck during play. For the same end, dealers have a very specific way that they deal the deck:
# 1. Shuffle - The dealer shuffles 
# 2. Dealter Cut - The dealer cuts
# 2. Shuffle - The dealer shuffles 
# 3. Player Cut - A random player (usually the lady at the table) cuts the deck using a special card
# 4. Dealer Final Cut - The dealer cuts the deck randomly or at a predetermined position to signify the end of the game when that card is reached (that hand is finished) 
# 
# Our deck class will allow an input of the number of decks included in the game, a simulation of the cuts, and store the dealer final cut. We will not be holding any logic in the deck. Instead it will be as it is in reality - a series of states. 

# In[24]:


class deck:

    def __init__(self, number_of_decks):
        self.number_of_decks = number_of_decks
        self._deck = None
        self.active_deck = None
        self.burnt = None
        self.dealer_final_cut = int()

    def _deck_init(self):

        create = lambda x, y: [x, y]

        self._deck = []
        self.active_deck = {}
        self.burnt = []

        suits = ['Spades', 'Clubs', 'Hearts', 'Diamonds']
        cards = ['Ace', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'Jack', 'Queen', 'King']
        for _x in range(self.number_of_decks):
            for suit in suits:
                for card in cards:
                    self._deck.append(
                        create(suit, card)
                    )

    def shuffle(self):
        self._deck_init()
        np.random.seed(1)
        working_deck = self._deck.copy()
        np.random.shuffle(working_deck)
        np.random.shuffle(working_deck)
        cut1 = np.random.randint(int(len(working_deck) * 0.33), int(len(working_deck) * 0.66))
        working_deck = working_deck[cut1:] + working_deck[:cut1]
        np.random.shuffle(working_deck)
        self.dealer_final_cut = int(len(working_deck) * 0.8)
        for x in range(len(working_deck)):
            self.active_deck[x] = working_deck[x]

    def draw(self):
        keys = list(self.active_deck.keys())
        active_card = [card for card in random.sample(keys, 1) if card not in self.burnt][0]
        self.burnt.append(active_card)
        return self.active_deck[active_card]

    def final(self):
        if len(self.burnt) >= self.dealer_final_cut:
            return True
        else:
            return False


# The class is simple and designed specifically for blackjack. It creates the deck variables, shuffles the deck according to what I remember from tables, draws and burns a card, and returns to an unknown entity if the deck is at the end of the round. Let's test it and draw a couple of cards. 

# In[25]:


ourdeck = deck(3)
ourdeck.shuffle()
card = ourdeck.draw()
print(card)
print(f"Cards burnt: {ourdeck.burnt}")


# ## Step 2 - The Player
# Cards are just pieces of paper without an interpreter. The player class will act as the function that interprets the cards. For now I am just creating a player that will act on the rules alone. In future phases of this I hope to apply a neural network that will create a rule set based on previous experience.

# In[32]:


class player:
    def __init__(self):
        self.state_single = 0
        self.state_multi = [0, 0]

    def inp(self, card):
        combine_multi = lambda x, y, z: x + y + z

        suit, value = card

        states = {'Ace': [1, 11], 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 'Jack': 10, 'Queen': 10,
                  'King': 10}
        state = states[value]
        if value == 'Ace' or self.state_multi != [0, 0]:
            self.state_multi = [combine_multi(self.state_single, state[0], self.state_multi[0]),
                                combine_multi(self.state_single, state[1], self.state_multi[1])]
        else:
            self.state_single += state
        print(f'{value} of {suit}')

    def oup(self):
        if self.state_multi != [0, 0]:
            print(f'Soft: {self.state_multi[0]}\nHard: (self.state_multi[1])')
        else:
            print(f'Hard: {self.state_single}')

    def reset(self):
        self.state_single = 0
        self.state_multi = [0, 0]


# In[40]:


ourdeck = deck(3)
ourplayer = player()

ourdeck.shuffle()

ourplayer.inp(ourdeck.draw())
ourplayer.inp(ourdeck.draw())

ourplayer.oup()
print(f"Cards burnt: {ourdeck.burnt}")


# In[ ]:




