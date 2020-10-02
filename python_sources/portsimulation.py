#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../input/PortSim.csv',header=None,skiprows=1)
returns = pd.DataFrame(data)

EQ = (returns[1][0])
FI = (returns[1][1])
EM = (returns[1][2])
CM = (returns[1][2])
RE = (returns[1][3])
CS = (returns[1][4])

# user_input number of teams
team_input = ""

# asks, validates, and returns user input for number of teams
while True:
    teams_input = input("Enter total number of teams: ")
    try:
        val_team = int(teams_input)
        if val_team > 0:
            print("----> Number of teams: ", val_team)
            print("")
            break
        else:
            print("----> Number must greater than 0, please try again.")
    except ValueError:
        print("----> User input is not a number, please try again.")

# user_input amount of total assets
asset_bool = True
asset_input = ""

# asks, validates, and returns user input for total assets
while asset_bool:
    asset_input = input("Enter total asset amount: ")
    try:
        val_amt = float(asset_input.replace(",", ""))
        print("----> Total Asset Amount: $", val_amt)
        print("")
        asset_bool = False
    except ValueError:
        print("")
        print("-----> User input is not a number, please try again.")

# will create n number of 6 object lists based on the users team_input
lists = [[0 for j in range(val_team)] for k in range(6)]

# Portfolio Weights
# There are six asset classes:
# Equity, Fixed Income, Emerging Market, Commodities, Real Estate, Cash

team_index = 0

while team_index < val_team:
    port_rmd = 1.0
    print("--------------------------------------------")
    print("Team " + str(team_index + 1) + " Weights:")
    print("--------------------------------------------")
    while True:
        weight_input = input("Enter Equity Weight (-100 to 100): ")
        try:
            val_equity = float(weight_input) / 100
            lists[0][team_index] = val_equity
            port_rmd = port_rmd - val_equity
            break
        except ValueError:
            print("----> User input is not a number, please try again.")
    print("************************************")
    print("Equity Weight: ", val_equity)
    print("Remaining Allocation: ", round(port_rmd, 3))
    print("")

    while True:
        weight_input = input("Enter Fixed Income Weight: ")
        try:
            val_fi = float(weight_input) / 100
            lists[1][team_index] = val_fi
            port_rmd = port_rmd - val_fi
            break
        except ValueError:
            print("----> User input is not a number, please try again.")
    print("************************************")
    print("Fixed Income Weight: ", val_fi)
    print("Remaining Allocation: ", round(port_rmd, 3))
    print("")

    while True:
        weight_input = input("Enter Emerging Market Weight: ")
        try:
            val_em = float(weight_input) / 100
            lists[2][team_index] = val_em
            port_rmd = port_rmd - val_em
            break
        except ValueError:
            print("----> User input is not a number, please try again.")
    print("************************************")
    print("Emerging Market Weight: ", val_em)
    print("Remaining Allocation: ", round(port_rmd, 3))
    print("")

    while True:
        weight_input = input("Enter Commodities Weight: ")
        try:
            val_com = float(weight_input) / 100
            lists[3][team_index] = val_com
            port_rmd = port_rmd - val_com
            break
        except ValueError:
            print("----> User input is not a number, please try again.")
    print("************************************")
    print("Emerging Market Weight: ", val_com)
    print("Remaining Allocation: ", round(port_rmd, 3))
    print("")

    while True:
        weight_input = input("Enter Real Estate Weight: ")
        try:
            val_re = float(weight_input) / 100
            lists[4][team_index] = val_re
            port_rmd = port_rmd - val_re
            break
        except ValueError:
            print("----> User input is not a number, please try again.")
    print("************************************")
    print("Emerging Market Weight: ", val_re)
    print("Remaining Allocation: ", round(port_rmd, 3))
    print("")

    while True:
        weight_input = input("Enter Cash Weight: ")
        try:
            val_cash = float(weight_input) / 100
            lists[5][team_index] = val_cash
            port_rmd = port_rmd - val_cash
            break
        except ValueError:
            print("----> User input is not a number, please try again.")
    print("************************************")
    print("Emerging Market Weight: ", val_cash)
    print("Remaining Allocation: ", round(port_rmd, 3))
    print("")

    team_index += 1

print("--------------------------------------------")
print("All Team Weights: [Equity[Team1][Team2], Fixed Income, Emerging Market, Commodities, Real Estate, Cash]")
print("--------------------------------------------")
print(lists)
print("")
print("")
print("")

print("Results: ")
f = 0
while f < val_team:  # calculates return by team
    print("--------------------------------------------")
    print('Team ' + str(f + 1) + ' Return: ')
    print("--------------------------------------------")
    ret = lists[0][f] * EQ + lists[1][f] * FI + lists[2][f] * EM + lists[3][f] * CM + lists[4][f] * 4.23 +           lists[5][f] * CS
    print('{:,.2f}%'.format(ret))
    AUM = val_amt*(1 + ret/100)
    print("--------------------------------------------")
    print('Team ' + str(f + 1) + ' Portfolio Value: ')
    print("--------------------------------------------")
    print('$ {:,.2f}'.format(AUM))

    labels = ['Equity', 'Fixed Income', 'Emerging Markets', 'Commodities', 'Real Estate', 'Cash']
    sizes = lists[0][f], lists[1][f], lists[2][f], lists[3][f], lists[4][f], lists[5][f]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'blue', 'grey', 'purple']
    patches, texts = plt.pie(sizes, colors = colors)
    plt.legend(patches, labels = labels, loc = "best")
    plt.title("Team " + str(f + 1) + ' Portfolio Weights Graph \n' 'Portfolio Return: ' + '{:,.2f}%'.format(ret) + '\n' + 'Portfolio Value: ' + '$ {:,.2f}'.format(AUM))
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    f += 1

# In[ ]:

