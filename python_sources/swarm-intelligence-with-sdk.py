#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # <center> Swarm Intelligence with SDK </center>
# 
# ---
# 
# <center><img src="https://images.unsplash.com/photo-1516434233442-0c69c369b66d?auto=format&fit=min&crop=top&w=800&h=400"></center>
# 
# ---

# ---
# <a id="top"></a>
# # Contents
# 
# This notebook is a <a href="https://www.kaggle.com/sam/halite-sdk-overview">Halite SDK</a> version of the totally cool notebook, <a href="https://www.kaggle.com/yegorbiryukov/halite-swarm-intelligence">Halite Swarm Intelligence</a>.
# 
# ---
# 
# <ol>
#     <li><a href="#kaggle-environments")>Install kaggle-environments</a></li>
#     <li><a href="#create-agent")>Create Agent</a></li>
#     <li><a href="#beetle-bot")>Beetle Bot</a></li>
#     <li><a href="#mini-bot")>Mini Bot</a></li>
#     <li><a href="#test-agent")>Test Agent</a></li>
# </ol>
# 
# ---

# <a id="kaggle-environments"></a>
# # Install kaggle-environments

# In[ ]:


get_ipython().system('pip install kaggle-environments --upgrade')


# <a href="#top">&uarr; back to top</a>

# <a id="create-agent"></a>
# # Create Agent
# ## Imports and Constants

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '\nfrom kaggle_environments.envs.halite.helpers import *\nfrom random import choice\n\nBOARD_SIZE = None\nEPISODE_STEPS = None\nCONVERT_COST = None\nSPAWN_COST = None\n\nNORTH = ShipAction.NORTH\nEAST = ShipAction.EAST\nSOUTH = ShipAction.SOUTH\nWEST = ShipAction.WEST\nCONVERT = ShipAction.CONVERT\nSPAWN = ShipyardAction.SPAWN\n\nDIRECTIONS = [NORTH, EAST, SOUTH, WEST]\n\nMOVEMENT_TACTICS = [\n    [NORTH, EAST, SOUTH, WEST],\n    [NORTH, WEST, SOUTH, EAST],\n    [EAST, SOUTH, WEST, NORTH],\n    [EAST, NORTH, WEST, SOUTH],\n    [SOUTH, WEST, NORTH, EAST],\n    [SOUTH, EAST, NORTH, WEST],\n    [WEST, NORTH, EAST, SOUTH],\n    [WEST, SOUTH, EAST, NORTH],\n]\nN_MOVEMENT_TACTICS = len(MOVEMENT_TACTICS)')


# ## Swarm Controller Class
# ### Class Overview
# 
# ```
# Controller: {
#     __init__(obs: Dict[str, Any], config: Dict[str, Any]) -> None
#     
#     clear(cell: Cell) -> bool
#     hostile_ship_near(cell: Cell, halite: int) -> bool
# 
#     spawn(shipyard: Shipyard) -> None
#     convert(ship: Ship) -> None
#     move(ship: Ship, direction: ShipAction) -> None
# 
#     endgame(ship: Ship) -> bool
#     build_shipyard(ship: Ship) -> bool
#     stay_on_cell(ship: Ship) -> bool
#     go_for_halite(ship: Ship) -> bool
#     unload_halite(ship: Ship) -> bool
#     standard_patrol(ship: Ship) -> bool
#     safety_convert(ship: Ship) -> bool
#     crash_shipyard(ship: Ship) -> bool
# 
#     actions_of_ships() -> None
#     actions_of_shipyards() -> None
#     
#     next_actions() -> Dict[str, str]
# }
# ```

# ### Class Implementation

# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\nclass Controller:\n    def __init__(self, obs, config):\n        """ Initialize parameters """\n        global BOARD_SIZE, EPISODE_STEPS, CONVERT_COST, SPAWN_COST\n        self.board = Board(obs, config)\n        self.player = self.board.current_player\n        self.STEP = obs.step\n        \n        # Define global constants\n        if self.STEP == 0:\n            BOARD_SIZE = config.size\n            EPISODE_STEPS = config.episodeSteps\n            CONVERT_COST = config.convertCost\n            SPAWN_COST = config.spawnCost\n            \n        self.FINAL_STEP = self.STEP == EPISODE_STEPS - 2\n        self.N_SHIPS = len(self.player.ships)\n        self.N_SHIPYARDS = len(self.player.shipyards)\n\n        # Cell tracking to avoid collisions of current player\'s ships\n        self.ship_cells = set(s.cell for s in self.player.ships)\n        self.ship_count = self.N_SHIPS\n        self.shipyard_count = self.N_SHIPYARDS\n        self.halite = self.player.halite\n\n        # Minimum total halite before ships can convert\n        self.CONVERT_THRESHOLD = CONVERT_COST + 3 * SPAWN_COST\n\n        stocks = [c.halite for c in self.board.cells.values() if c.halite > 0]\n        average_halite = int(sum(stocks) / len(stocks)) if len(stocks) > 0 else 0\n        # Minimum halite a cell must have before a ship will harvest\n        self.LOW_HALITE = max(average_halite // 2, 4)\n\n        # Minimum number of ships at any time\n        self.MIN_SHIPS = 10\n        # Maximum number of ships to spawn\n        self.MAX_SHIPS = 0\n        # Increase MAX_SHIPS in first half of game only\n        if self.STEP < EPISODE_STEPS // 2:\n            total_ships = sum(len(p.ships) for p in self.board.players.values())\n            if total_ships > 0:\n                self.MAX_SHIPS = (average_halite // total_ships) * 10\n        # Fix MAX_SHIPS if less than MIN_SHIPS\n        self.MAX_SHIPS = max(self.MIN_SHIPS, self.MAX_SHIPS)\n\n    def clear(self, cell):\n        """ Check if cell is safe to move in """\n        if (cell.ship is not None and\n                cell.ship not in self.player.ships):\n            return False\n\n        if (cell.shipyard is not None and\n                cell.shipyard not in self.player.shipyards):\n            return False\n\n        if cell in self.ship_cells:\n            return False\n        return True\n\n    def hostile_ship_near(self, cell, halite):\n        """ Check if hostile ship is one move away and has less or equal halite """\n        neighbors = [cell.neighbor(d.to_point()) for d in DIRECTIONS]\n        for neighbor in neighbors:\n            if (neighbor.ship is not None and\n                neighbor.ship not in self.player.ships and\n                    neighbor.ship.halite <= halite):\n                return True\n        return False\n\n    def spawn(self, shipyard):\n        """ Spawn ship from shipyard """\n        shipyard.next_action = SPAWN\n        self.halite -= SPAWN_COST\n        self.ship_count += 1\n        # Cell tracking to avoid collisions of current player\'s ships\n        self.ship_cells.add(shipyard.cell)\n\n    def convert(self, ship):\n        """ Convert ship to shipyard """\n        ship.next_action = CONVERT\n        self.halite -= CONVERT_COST\n        self.ship_count -= 1\n        self.shipyard_count += 1\n        # Cell tracking to avoid collisions of current player\'s ships\n        self.ship_cells.remove(ship.cell)\n\n    def move(self, ship, direction):\n        """ Move ship in direction """\n        ship.next_action = direction\n        # Cell tracking to avoid collisions of current player\'s ships\n        if direction is not None:\n            d_cell = ship.cell.neighbor(direction.to_point())\n            self.ship_cells.remove(ship.cell)\n            self.ship_cells.add(d_cell)\n            \n    def endgame(self, ship):\n        """" Final step: convert if possible """\n        if (self.FINAL_STEP and\n                ship.halite >= CONVERT_COST):\n            self.convert(ship)\n            return True\n        return False\n    \n    def build_shipyard(self, ship):\n        """ Convert to shipyard if necessary """\n        if (self.shipyard_count == 0 and\n              self.ship_count < self.MAX_SHIPS and\n              self.STEP < EPISODE_STEPS // 2 and\n              self.halite + ship.halite >= self.CONVERT_THRESHOLD and\n              not self.hostile_ship_near(ship.cell, ship.halite)):\n            self.convert(ship)\n            return True\n        return False\n    \n    def stay_on_cell(self, ship):\n        """ Stay on current cell if profitable and safe """\n        if (ship.cell.halite > self.LOW_HALITE and\n              not self.hostile_ship_near(ship.cell, ship.halite)):\n            ship.next_action = None\n            return True\n        return False\n    \n    def go_for_halite(self, ship):\n        """ Ship will move to safe cell with largest amount of halite """\n        neighbors = [(d, ship.cell.neighbor(d.to_point())) for d in DIRECTIONS]\n        candidates = [(d, c) for d, c in neighbors if self.clear(c) and\n                      not self.hostile_ship_near(c, ship.halite) and\n                      c.halite > self.LOW_HALITE]\n\n        if candidates:\n            stocks = [c.halite for d, c in candidates]\n            max_idx = stocks.index(max(stocks))\n            direction = candidates[max_idx][0]\n            self.move(ship, direction)\n            return True\n        return False\n\n    def unload_halite(self, ship):\n        """ Unload ship\'s halite if it has any and vacant shipyard is near """\n        if ship.halite > 0:\n            for d in DIRECTIONS:\n                d_cell = ship.cell.neighbor(d.to_point())\n\n                if (d_cell.shipyard is not None and\n                        self.clear(d_cell)):\n                    self.move(ship, d)\n                    return True\n        return False\n\n    def standard_patrol(self, ship):\n        """ Ship will move in circles clockwise or counterclockwise if safe"""\n        # Choose movement tactic\n        i = int(ship.id.split("-")[0]) % N_MOVEMENT_TACTICS\n        directions = MOVEMENT_TACTICS[i]\n        # Select initial direction\n        n_directions = len(directions)\n        j = (self.STEP // BOARD_SIZE) % n_directions\n        # Move to first safe direction found\n        for _ in range(n_directions):\n            direction = directions[j]\n            d_cell = ship.cell.neighbor(direction.to_point())\n            # Check if direction is safe\n            if (self.clear(d_cell) and\n                    not self.hostile_ship_near(d_cell, ship.halite)):\n                self.move(ship, direction)\n                return True\n            # Try next direction\n            j = (j + 1) % n_directions\n        # No safe direction\n        return False\n\n    def safety_convert(self, ship):\n        """ Convert ship if not on shipyard and hostile ship is near """\n        if (ship.cell.shipyard is None and\n            self.hostile_ship_near(ship.cell, ship.halite) and\n                ship.halite >= CONVERT_COST):\n            self.convert(ship)\n            return True\n        return False\n\n    def crash_shipyard(self, ship):\n        """ Crash into opponent shipyard """\n        for d in DIRECTIONS:\n            d_cell = ship.cell.neighbor(d.to_point())\n\n            if (d_cell.shipyard is not None and\n                    d_cell.shipyard not in self.player.shipyards):\n                self.move(ship, d)\n                return True\n        return False\n\n    def actions_of_ships(self):\n        """ Next actions of every ship """\n        for ship in self.player.ships:\n            # Act according to first acceptable tactic\n            if self.endgame(ship):\n                continue\n            if self.build_shipyard(ship):\n                continue\n            if self.stay_on_cell(ship):\n                continue\n            if self.go_for_halite(ship):\n                continue\n            if self.unload_halite(ship):\n                continue\n            if self.standard_patrol(ship):\n                continue\n            if self.safety_convert(ship):\n                continue\n            if self.crash_shipyard(ship):\n                continue\n            # Default random action\n            self.move(ship, choice(DIRECTIONS + [None]))\n\n    def actions_of_shipyards(self):\n        """ Next actions of every shipyard """\n        # Spawn ships from every shipyard if possible\n        for shipyard in self.player.shipyards:\n            if (self.ship_count < self.MAX_SHIPS and\n                self.halite >= SPAWN_COST and\n                not self.FINAL_STEP and\n                    self.clear(shipyard.cell)):\n                self.spawn(shipyard)\n            else:\n                shipyard.next_action = None\n\n    def next_actions(self):\n        """ Perform next actions for current player """\n        self.actions_of_ships()\n        self.actions_of_shipyards()\n        return self.player.next_actions')


# ## Main Agent Function

# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\ndef agent(obs, config):\n    controller = Controller(obs, config)\n    return controller.next_actions()')


# <a href="#top">&uarr; back to top</a>

# <a id="beetle-bot"></a>
# # Beetle Bot
# 
# This is an idle bot with one ship and one shipyard which is described in the notebook: <a href="https://www.kaggle.com/benzyx/make-sure-you-can-beat-this-baseline-idle-bot">Make sure you can beat this: Baseline Idle Bot</a>.
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'beetle_bot.py', 'from kaggle_environments.envs.halite.helpers import *\n\ndef agent(obs, config):    \n    board = Board(obs, config)\n    player = board.current_player\n    \n    for ship in player.ships:\n        if len(player.shipyards) == 0:\n            ship.next_action = ShipAction.CONVERT\n        \n    for shipyard in player.shipyards:\n        if len(player.ships) == 0:\n            shipyard.next_action = ShipyardAction.SPAWN\n            \n    return player.next_actions')


# <a href="#top">&uarr; back to top</a>

# <a id="mini-bot"></a>
# # Attack Bot
# 
# This bot tries to attack the swarm.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'attack_bot.py', 'from kaggle_environments.envs.halite.helpers import *\n\nDIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, \n              ShipAction.SOUTH, ShipAction.WEST]\n\ndef agent(obs, config):\n    board = Board(obs, config)\n    player = board.current_player\n    next_cells = set()\n\n    def safe(c, halite):\n        if c in next_cells:\n            return False\n        good = player.ships + player.shipyards + [None]\n        if c.shipyard not in good:\n            return False\n        for n in [c, c.north, c.east, c.south, c.west]:\n            if (n.ship not in good) and (n.ship.halite <= halite):\n                return False\n        return True\n\n    def next_action(s, action):\n        s.next_action = action\n        if action in DIRECTIONS:\n            next_cells.add(s.cell.neighbor(action.to_point()))\n        elif action is None:\n            next_cells.add(s.cell)\n\n    yields = [int(c.halite) for c in board.cells.values() if c.halite > 0]\n    min_halite = max(4, sum(yields) // max(1, len(yields)) // 2)\n    max_shipyards = 10\n    \n    for ship in player.ships:\n        cell = ship.cell\n        ship.next_action = None\n        \n        if len(player.shipyards) == 0 and safe(cell, ship.halite):\n            next_action(ship, ShipAction.CONVERT)\n            continue\n            \n        if (obs.step == config.episodeSteps - 2 and \n            ship.halite >= config.convertCost):\n            next_action(ship, ShipAction.CONVERT)\n            continue\n            \n        if (obs.step > config.episodeSteps - 20 and \n            len(player.shipyards) > 0 and ship.halite > 0):\n            i = sum(int(k) for k in ship.id.split("-")) % len(player.shipyards)\n            dx, dy = player.shipyards[i].position - ship.position\n            if dx > 0 and safe(cell.east, ship.halite):\n                next_action(ship, ShipAction.EAST)\n            elif dx < 0 and safe(cell.west, ship.halite):\n                next_action(ship, ShipAction.WEST)\n            elif dy > 0 and safe(cell.north, ship.halite):\n                next_action(ship, ShipAction.NORTH)\n            elif dy < 0 and safe(cell.south, ship.halite):\n                next_action(ship, ShipAction.SOUTH)\n            if ship.next_action in DIRECTIONS:\n                continue\n        \n        for d in DIRECTIONS:\n            neighbor = cell.neighbor(d.to_point())\n            if (neighbor.ship is not None and \n                neighbor.ship not in player.ships and\n                safe(neighbor, ship.halite)):\n                next_action(ship, d)\n                break\n        if ship.next_action in DIRECTIONS:\n            continue\n                \n        if cell.halite > min_halite and safe(cell, ship.halite):\n            next_action(ship, None)\n            continue\n            \n        if (ship.halite > config.convertCost * 4 and \n            len(player.shipyards) < max_shipyards and safe(cell, ship.halite)):\n            next_action(ship, ShipAction.CONVERT)\n            continue\n                \n        neighbors = [cell.neighbor(d.to_point()) for d in DIRECTIONS]\n        max_halite = max([0] + [n.halite for n in neighbors if safe(n, ship.halite)])\n        i = sum(int(k) for k in ship.id.split("-")) * config.size\n        j = ((i + obs.step) // config.size) % 4\n        safe_list = []\n        for _ in range(4):\n            d = DIRECTIONS[j]\n            n = cell.neighbor(d.to_point())\n            if safe(n, ship.halite):\n                if ((n.halite > min_halite and n.halite == max_halite) or \n                    (ship.halite > 20 and n.shipyard in player.shipyards)):\n                    next_action(ship, d)\n                    break\n                safe_list.append(d)\n            j = (j + 1) % 4\n        else:\n            if safe_list:\n                next_action(ship, safe_list[0])\n            elif safe(cell, ship.halite):\n                next_action(ship, None)\n            elif ship.halite >= config.convertCost:\n                next_action(ship, ShipAction.CONVERT)\n            else:\n                next_action(ship, None)\n                \n    max_ships = min(50, len(player.ships) +\n                    player.halite // config.spawnCost // \n                    max(1, len(player.shipyards)))\n    \n    for shipyard in player.shipyards:\n        if len(player.ships) == 0:\n            shipyard.next_action = ShipyardAction.SPAWN\n            \n        elif (len(player.ships) < max_ships and\n              obs.step < config.episodeSteps - 50 and\n              safe(shipyard.cell, ship.halite)):\n            shipyard.next_action = ShipyardAction.SPAWN\n            \n    return player.next_actions')


# <a href="#top">&uarr; back to top</a>

# <a id="test-agent"></a>
# # Test Agent

# In[ ]:


from kaggle_environments import make
environment = make("halite", configuration={"episodeSteps": 400}, debug=True)
environment.run(["submission.py", "random", "beetle_bot.py", "attack_bot.py"])
environment.render(mode="ipython", width=720, height=540)


# <a href="#top">&uarr; back to top</a>
