# Clear the screen
import os
os.system("clear")

# Import the random module
import random

# Static datasets
bottles = ["short bottle",
           "medium bottle",
           "tall bottle",
          ]

alcohol = ["guinness",
           "lager",
           "pale ale",
           "ipa",
          ]

furniture = ["counter",
             "booth",
             "table",
            ]

# Convert datasets to string and strip brackets and single commas.
sample_bottles   = random.sample(bottles, k = 1)
sample_alcohol   = random.sample(alcohol, k = 1)
sample_furniture = random.sample(furniture, k = 1)

active_bottle    = str(sample_bottles).strip("[]''")
active_alcohol   = str(sample_alcohol).strip("[]''")
active_furniture = str(sample_furniture).strip("[]''")

# print the output when file writing isn't an option.
print("The bottle that I need is "          + active_bottle    +
      ". The alcohol for this bottle is "   + active_alcohol   +
      ". The furniture to take it is at a " + active_furniture +
      ".\n")

# Write data to file.
file = open("bottle_relationship.txt", "w")

file.write("The bottle that I need is "          + active_bottle    +
           ". The alcohol for this bottle is "   + active_alcohol   +
           ". The furniture to take it is at a " + active_furniture +
           ".\n")