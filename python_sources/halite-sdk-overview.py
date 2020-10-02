#!/usr/bin/env python
# coding: utf-8

# If you're new to Halite, check out Alexis' [Getting Started With Halite Notebook](https://www.kaggle.com/alexisbcook/getting-started-with-halite)

# In[ ]:


# Make sure we have the latest kaggle-environments installed
get_ipython().system('pip install kaggle-environments --upgrade')


# In[ ]:


from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

# Create a test environment for use later
board_size = 5
environment = make("halite", configuration={"size": board_size, "startingHalite": 1000})
agent_count = 2
environment.reset(agent_count)
state = environment.state[0]


# # Board
# Board is the top-level entity when working with the Halite SDK and is the main type we'll construct when building a Halite agent. A board represents the Halite simulation state at a particular turn. Boards are constructed with an observation and configuration as defined in [the halite schema](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/halite/halite.json).
# 
# Boards track all entities in the Halite simulation state including cells, players, ships, and shipyards.

# In[ ]:


board = Board(state.observation, environment.configuration)
print(board)


# **Board.\_\_str__ Legend:**  
# * Capital letters are shipyards  
# * Lower case letters are ships  
# * Digits are cell halite and scale from 0-9 where 9 is `Board.configuration.max_cell_halite`  
# * Player 1 is letter a/A  
# * Player 2 is letter b/B  
# * Player n is letter x/X

# In[ ]:


print(f"Player Ids: {[player.id for player in board.players.values()]}")
print(f"Ship Ids: {[ship.id for ship in board.ships.values()]}")
# Note there are currently no shipyards because our Board just initialized
print(f"Shiyard Ids: {[shipyard.id for shipyard in board.shipyards.values()]}")
assert len(board.cells) == board_size * board_size


# # Point
# Points are used to represent positions on the halite board as well as offsets when a ship moves from one position to another. Note that in `Board` the point `(0, 0)` is the lower left corner and the point `(board.configuration.size - 1, board.configuration.size - 1)` is the upper right corner. This differs from the raw Halite observation where the index `0` is the upper left corner and the index `(board.configuration.size * board.configuration.size) - 1` is the lower right corner.

# In[ ]:


point = Point(3, -4)

# Points are tuples
assert isinstance(point, tuple)
# Points have x and y getters (no setters)
assert point.x == 3
assert point.y == -4
# Points implement several common operators
assert point == Point(3, -4)
assert abs(point) == Point(3, 4)
assert -point == Point(-3, 4)
assert point + point == (6, -8)
assert point - point == Point(0, 0)
assert point * 3 == Point(9, -12)
# Point supports floordiv but not div since x and y are ints not floats
assert point // 2 == Point(1, -2)
assert point % 3 == Point(0, 2)
# Prints like a tuple
print(point)
print(board[point])


# # Actions
# Actions are how our agent issues commands to our ships and shipyards. The `Ship` and `Shipyard` types each have a settable `next_action` property that enqueues an action to be executed for that ship or shipyard at the end of the current turn. `ShipAction` and `ShipyardAction` types contain enum values for each action.

# In[ ]:


print([action.name for action in ShipAction])
print([action.name for action in ShipyardAction])

# Grab a ship to test with
ship = next(iter(board.ships.values()))
print(f"Initial action: {ship.next_action}")
ship.next_action = ShipAction.NORTH
print(f"New action: {ship.next_action}")


# # Simulating Actions (Lookahead)
# Above we have set the `next_action` for our ship to NORTH. The `Board.next()` method applies all currently set `next_actions` and steps time forward in the Halite simulation. This method returns a completely new board that will not be affected by subsequent changes to the previous board. Because we've already set `ship.next_action = ShipAction.NORTH` then when we call `board.next()` we should see that our ship has moved north in the next board.

# In[ ]:


print(board)
board = board.next()
print(board)

# Let's make sure we moved north!
next_ship = board.ships[ship.id]
assert next_ship.position - ship.position == ShipAction.NORTH.to_point()
# We'll use this in the next cell
ship = next_ship

# What happens if we call board.next()?
print(board.next())


# Notice that the second board looks just like the third board. This is because the call to `board.next()` clears all `next_actions` from ships and shipyards in the new board to prevent actions from automatically repeating from step to step.
# 
# What if we want to make a shipyard?

# In[ ]:


ship.next_action = ShipAction.CONVERT
board = board.next()
print(board)


# Cool, we've got a shipyard, now let's make a new ship!

# In[ ]:


shipyard = board[ship.position].shipyard
shipyard.next_action = ShipyardAction.SPAWN
board = board.next()
print(board)


# We can simulate actions for any player in the simulation, not just the current player. Of course our agent won't be able to control opponents during the actual episode evaluation but this technique can be useful for planning our own actions based on the actions we expect our opponents to take.
# 
# Let's try moving all ships south.

# In[ ]:


for ship in board.ships.values():
    ship.next_action = ShipAction.SOUTH
board = board.next()
print(board)


# As we set `Ship.next_action` and `Shipyard.next_action` for each entity in our board, the board tracks all of those actions for each player in the `Player.next_actions` property. `Board.current_player` refers to the player that our agent represents, so we can retrieve all of the queued actions for our ships and shipyards with `Board.current_player.next_actions`. Note that the return type of `Player.next_actions` is `Dict[str, str]` **not** `Dict[Union[ShipId, ShipyardId], Union[ShipAction, ShipyardAction]]`. We'll find out why in the next section.

# In[ ]:


current_player = board.current_player
for ship in current_player.ships:
    ship.next_action = ShipAction.SOUTH
print(current_player.next_actions)


# # Creating an Agent
# Now we're able to plan and simulate changes to the Halite simulation state. The last step is turning this knowledge into a working agent.
# As a reminder the signature for a Halite agent is 
# ```
# agent(observation: Dict[str, Any], configuration: Dict[str, Any]) -> Dict[str, str]
# ```
# The observation and configuration agent parameters can be passed directly to the `Board` constructor to create a board representing the current simulation state.
# 
# Lastly, we need to generate our return value. The return value of an agent is a dict where the keys are ship or shipyard ids and the values are ship or shipyard actions for the corresponding id to execute.After queueing all actions for our ships and shipyards, we can call `board.current_player.next_actions` to retrieve those actions as an agent response. As noted in the previous section, the return type of `Player.next_actions` is `Dict[str, str]` -- this is to match the return type for agents.

# In[ ]:


def move_ships_north_agent(observation, configuration):
    board = Board(observation, configuration)
    current_player = board.current_player
    for ship in current_player.ships:
        ship.next_action = ShipAction.NORTH
    return current_player.next_actions

environment.reset(agent_count)
environment.run([move_ships_north_agent, "random"])
environment.render(mode="ipython", width=500, height=450)


# We have a complete agent, but wait, there's (slightly) more! This library also vends a `@board_agent` decorator for converting a regular agent into an agent that accepts a `Board` and assigns all `next_actions`. The signature of a `@board_agent` is `agent(board: Board) -> None`, note that there is no return value, we just have to modify the board that's passed to us. Let's convert our `move_ships_north_agent` to a `@board_agent`.

# In[ ]:


@board_agent
def move_ships_north_agent(board):
    for ship in board.current_player.ships:
        ship.next_action = ShipAction.NORTH

environment.reset(agent_count)
environment.run([move_ships_north_agent, "random"])
environment.render(mode="ipython", width=500, height=450)


# # Board (Advanced)
# Until now we've always constructed our boards with just an observation and configuration but `Board` also accepts an optional third parameter:
# ```
# Board.__init__(observation: Dict[str, Any], configuration: Dict[str, Any], next_actions: Optional[List[Dict[str, str]]] = None) -> None
# ```
# This parameter can be used to populate next_actions for the board from an external source like an agent.

# In[ ]:


first_player_actions = {'0-1': 'CONVERT'}
second_player_actions = {'0-2': 'NORTH'}

actions = [first_player_actions, second_player_actions]
board = Board(state.observation, environment.configuration, actions)
print(board)
print(board.next())


# This technique utilizes the `Board` class just for its lookahead ability and not for its action forming or state browsing constructs.

# # Final Thoughts
# That about covers things! Please let me know if you have any questions, comments, or suggestions for the Halite SDK (or this tutorial) in the discussion for this notebook or [on GitHub](https://github.com/Kaggle/kaggle-environments/).

# # Schema Cheat Sheet
# ```
# Board: {
#     __init__(observation: Dict[str, Any], configuration: Dict[str, Any], next_actions: Optional[List[Dict[str, str]]] = None) -> None
#     cells -> Dict[Point, Cell]
#     ships -> Dict[ShipId, Ship]
#     shipyards -> Dict[ShipyardId, Shipyard]
#     players -> Dict[PlayerId, Player]
#     
#     current_player_id -> PlayerId
#     current_player -> Player
#     opponents -> List[Player]
#     
#     configuration -> Configuration
#     observation -> Dict[str, Any]
#     step -> int
#     
#     next() -> Board
#     
#     __deepcopy__(_) -> Board
#     __getitem__(point: Union[Tuple[int, int], Point]) -> Cell
#     __str__() -> str
# }
# 
# Cell: {
#     position -> Point
#     halite -> float
#     
#     ship_id -> Optional[ShipId]
#     ship -> Optional[Ship]
#     
#     shipyard_id -> Optional[ShipyardId]
#     shipyard -> Optional[Shipyard]
# 
#     north -> Cell
#     south -> Cell
#     east -> Cell
#     west -> Cell
#     
#     neighbor(offset: Point) -> Cell
# }
# 
# Ship: {
#     id -> ShipId
#     halite -> int
#     
#     position -> Point
#     cell -> Cell
#     
#     player_id -> PlayerId
#     player -> Player
#     
#     next_action -> Optional[ShipAction]
# }
# 
# Shipyard: {
#     id -> ShipyardId
#     
#     position -> Point
#     cell -> Cell
#     
#     player_id -> PlayerId
#     player -> Player
#     
#     next_action -> Optional[ShipyardAct
# }
# 
# Player: {
#     id -> PlayerId
#     is_current_player -> bool
#     halite -> int
#     next_actions -> Dict[str, str]
#     
#     ship_ids -> List[ShipId]
#     shipys -> List[Ship]
#     
#     shipyard_ids -> List[ShipyardId]
#     shipyards -> List[Shipyard]
# }
# 
# Point: {
#     x -> int
#     y -> int
#     
#     translate(offset: Point, size: int) -> Point
#     to_index(size: int) -> int
#     
#     @staticmethod
#     from_index(index: int, size: int) -> Point
#     
#     __abs__() -> Point
#     __add__(other: Point) -> Point
#     __eq__(other: Point) -> bool
#     __floordiv__(denominator: int) -> Point
#     __hash__() -> int
#     __mod__(mod: int) -> Point
#     __mul__(factor: int) -> Point
#     __neg__() -> Point
#     __str__() -> str
#     __sub__(other: Point) -> Point
# }
# ```
