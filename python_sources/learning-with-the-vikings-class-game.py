#!/usr/bin/env python
# coding: utf-8

# 
# ## Introduction
# 
# The Vikings and the Saxons are at War. Both are Soldiers but they have their own methods to fight. Vikings are ported to Python. YAY!!
# 
# In this laboratory you will work with the concept of inheritance in Python.
# 
# 
# ### Soldier
# 
# Modify the `Soldier` constructor function and add 2 methods to its prototype: `attack()`, and `receiveDamage()`.
# 
# #### constructor function
# 
# - should receive **2 arguments** (health & strength)
# - should receive the **`health` property** as its **1st argument**
# - should receive the **`strength` property** as its **2nd argument**
# 
# #### `attack()` method
# 
# - should be a function
# - should receive **0 arguments**
# - should return **the `strength` property of the `Soldier`**
# 
# #### `receiveDamage()` method
# 
# - should be a function
# - should receive **1 argument** (the damage)
# - should remove the received damage from the `health` property
# - **shouldn't return** anything
# 
# ---
# 
# ### Viking
# 
# A `Viking` is a `Soldier` with an additional property, their `name`. They also have a different `receiveDamage()` method and new method, `battleCry()`.
# 
# Modify the `Viking` constructor function, have it inherit from `Soldier`, reimplement the `receiveDamage()` method for `Viking`, and add a new `battleCry()` method.
# 
# #### inheritance
# 
# - `Viking` should inherit from `Soldier`
# 
# #### constructor function
# 
# - should receive **3 arguments** (name, health & strength)
# - should receive the **`name` property** as its **1st argument**
# - should receive the **`health` property** as its **2nd argument**
# - should receive the **`strength` property** as its **3rd argument**
# 
# #### `attack()` method
# 
# (This method should be **inherited** from `Soldier`, no need to reimplement it.)
# 
# - should be a function
# - should receive **0 arguments**
# - should return **the `strength` property of the `Viking`**
# 
# #### `receiveDamage()` method
# 
# (This method needs to be **reimplemented** for `Viking` because the `Viking` version needs to have different return values.)
# 
# - should be a function
# - should receive **1 argument** (the damage)
# - should remove the received damage from the `health` property
# - **if the `Viking` is still alive**, it should return **"NAME has received DAMAGE points of damage"**
# - **if the `Viking` dies**, it should return **"NAME has died in act of combat"**
# 
# #### `battleCry()` method
# 
# [Learn more about battle cries](http://www.artofmanliness.com/2015/06/08/battle-cries/).
# 
# - should be a function
# - should receive **0 arguments**
# - should return **"Odin Owns You All!"**
# 
# ---
# 
# ### Saxon
# 
# A `Saxon` is a weaker kind of `Soldier`. Unlike a `Viking`, a `Saxon` has no name. Their `receiveDamage()` method will also be different than the original `Soldier` version.
# 
# Modify the `Saxon`, constructor function, have it inherit from `Soldier` and reimplement the `receiveDamage()` method for `Saxon`.
# 
# #### inheritance
# 
# - `Saxon` should inherit from `Soldier`
# 
# #### constructor function
# 
# - should receive **2 arguments** (health & strength)
# - should receive the **`health` property** as its **1st argument**
# - should receive the **`strength` property** as its **2nd argument**
# 
# #### `attack()` method
# 
# (This method should be **inherited** from `Soldier`, no need to reimplement it.)
# 
# - should be a function
# - should receive **0 arguments**
# - should return **the `strength` property of the `Saxon`**
# 
# #### `receiveDamage()` method
# 
# (This method needs to be **reimplemented** for `Saxon` because the `Saxon` version needs to have different return values.)
# 
# - should be a function
# - should receive **1 argument** (the damage)
# - should remove the received damage from the `health` property
# - **if the Saxon is still alive**, it should return _**"A Saxon has received DAMAGE points of damage"**_
# - **if the Saxon dies**, it should return _**"A Saxon has died in combat"**_
# 
# ---
# 
# ### (BONUS) War
# 
# Now we get to the good stuff: WAR! Our `War` constructor function will allow us to have a `Viking` army and a `Saxon` army that battle each other.
# 
# Modify the `War` constructor and add 5 methods to its prototype:
# 
# - `addViking()`
# - `addSaxon()`
# - `vikingAttack()`
# - `saxonAttack()`
# - `showStatus()`
# 
# #### constructor function
# 
# When we first create a `War`, the armies should be empty. We will add soldiers to the armies later.
# 
# - should receive **0 arguments**
# - should assign an empty array to the **`vikingArmy` property**
# - should assign an empty array to the **`saxonArmy` property**
# 
# #### `addViking()` method
# 
# Adds 1 `Viking` to the `vikingArmy`. If you want a 10 `Viking` army, you need to call this 10 times.
# 
# - should be a function
# - should receive **1 argument** (a `Viking` object)
# - should add the received `Viking` to the army
# - **shouldn't return** anything
# 
# #### `addSaxon()` method
# 
# The `Saxon` version of `addViking()`.
# 
# - should be a function
# - should receive **1 argument** (a `Saxon` object)
# - should add the received `Saxon` to the army
# - **shouldn't return** anything
# 
# #### `vikingAttack()` method
# 
# A `Saxon` (chosen at random) has their `receiveDamage()` method called with the damage equal to the `strength` of a `Viking` (also chosen at random). This should only perform a single attack and the `Saxon` doesn't get to attack back.
# 
# - should be a function
# - should receive **0 arguments**
# - should make a `Saxon` `receiveDamage()` equal to the `strength` of a `Viking`
# - should remove dead saxons from the army
# - should return **result of calling `receiveDamage()` of a `Saxon`** with the `strength` of a `Viking`
# 
# #### `saxonAttack()` method
# 
# The `Saxon` version of `vikingAttack()`. A `Viking` receives the damage equal to the `strength` of a `Saxon`.
# 
# - should be a function
# - should receive **0 arguments**
# - should make a `Viking` `receiveDamage()` equal to the `strength` of a `Saxon`
# - should remove dead vikings from the army
# - should return **result of calling `receiveDamage()` of a `Viking`** with the `strength` of a `Saxon`
# 
# #### `showStatus()` method
# 
# Returns the current status of the `War` based on the size of the armies.
# 
# - should be a function
# - should receive **0 arguments**
# - **if the `Saxon` array is empty**, should return _**"Vikings have won the war of the century!"**_
# - **if the `Viking` array is empty**, should return _**"Saxons have fought for their lives and survive another day..."**_
# - **if there are at least 1 `Viking` and 1 `Saxon`**, should return _**"Vikings and Saxons are still in the thick of battle."**_
# 

# In[ ]:


###Solution###

import random as rd

class Soldier:
    def __init__(self, health, strength):
        self.health = health
        self.strength = strength

    def attack(self):
        return self.strength

    def receiveDamage(self, damage):
        self.health = self.health - damage
        
class Viking(Soldier):
    def __init__(self, name, health, strength):
        Soldier.__init__(self, health, strength)
        self.name = name
    def receiveDamage(self, damage):
        self.health = self.health - damage
        if self.health > 0:
            return self.name + " has received " +  str(damage) + " points of damage"
        else:
            return self.name + " has died in act of combat"
    def battleCry(self):
        return "Odin Owns You All!"
    
class Saxon(Soldier):
    def receiveDamage(self, damage):
        self.health = self.health - damage
        if self.health > 0:
            return "A Saxon has received " +  str(damage) + " points of damage"
        else:
            return "A Saxon has died in combat"


# In[ ]:


class War:
    def __init__(self):
        self.vikingArmy = []
        self.saxonArmy = []

    def addViking(self, vik):
        if isinstance(vik, Viking):
            self.vikingArmy.append(vik)


    def addSaxon(self, sax):
        if isinstance(sax, Saxon):
            self.saxonArmy.append(sax)

    def vikingAttack(self):
        v = rd.choice(self.vikingArmy)
        s = rd.choice(self.saxonArmy)

        dam_sa = s.receiveDamage(v.attack())

        if s.health <= 0:
            self.saxonArmy.remove(s)

        return dam_sa

    def saxonAttack(self):
        u = rd.choice(self.vikingArmy)
        i = rd.choice(self.saxonArmy)

        dam_vi = u.receiveDamage(i.attack()) 

        if u.health <= 0 :
            self.vikingArmy.remove(u)

        return dam_vi

    def showStatus(self):
        if len(self.saxonArmy) == 0:
            return "Vikings have won the war of the century!"
        elif len(self.vikingArmy) == 0:
            return "Saxons have fought for their lives and survive another day..."
        else:
            return "Vikings and Saxons are still in the thick of battle."

