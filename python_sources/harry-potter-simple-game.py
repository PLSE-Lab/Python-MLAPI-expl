# %% [code]
from sys import exit

print("What's your name?")
name = input("> ")

def help_death_eaters():
    print("""They want you to help them resurrect Voldemort. Do you help them or refuse?""")

    choice = input ("> ")
    if "help" in choice:
        print(f"""Voldemort returns and together you conquer the world. Muahahahhaahaaaa!!!""")
        exit(0)
    elif "refuse" in choice:
        print("""Draco and his crew don't like your decision at all.
        They Imperius you and make you resurrect Voldemort anyway.
        As Voldi's new servant, you help him enslave the world. Too bad.""")
        exit(0)
    else:
        print("Huh? Please try again.")
        help_death_eaters()

def sorting_hat_new():
    print("""All the students are watching you as you put on the Sorting Hat again.
    Do you choose Gryffindor, Hufflepuff, Ravenclaw or Slytherin?""")

    choice = input ("> ")
    if "Gryffindor" in choice:
        print("Gryffindor it is!!")
        hermione()
    elif "Hufflepuff" in choice:
        print("""Hufflepuff it is!!
        You become a kind and loyal member of your house.
        However, spending so much time helping your classmates with their homework doesn't leave you time to practice enough magic.
        When you encounter Voldemort, he finishes you off. Too bad.""")
        exit(0)
    elif "Ravenclaw" in choice:
        print("""Ravenclaw it is!!
        You spend hours upon hours in the library and in your sixth year invent 'Rocallas imperialis', a super powerful spell.
        This spell allows you to defeat Voldemort in the final battle. Nice work! The magical world is saved and worships you forever.""")
        exit(0)
    elif "Slytherin" in choice:
        print("""Haven't you learned anything at all?
        Once you're kicked out of Slytherin, you're out.
        You get expulsed from Hogwarts and die never having fulfilled your wizard potential.
        Meanwhile, Voldemort takes over the world.""")
        exit(0)
    else:
        print("Huh? Please try again.")
        sorting_hat_new()


def draco():
    print(f"""Draco Malfoy offers to be your friend. Do you accept his offer?""")

    choice = input ("> ")
    if "yes" in choice:
        print(f"You and {diagon_alley.your_pet} become besties with Draco and his Death Eater friends.")
        # and {pet_name}
        help_death_eaters()
    elif "no" in choice:
        print("""Draco's father Lucius has you kicked out of Slytherin. Now you have to get sorted into a new house. Shame on you!!""")
        sorting_hat_new()
    else:
        print("Huh? Please try again.")
        draco()


def pet():
    print(f"""As you walk off to your duel with Voldemort, {diagon_alley.your_pet} signals you that it wants to come with you.
    What do you do? Do you take {diagon_alley.your_pet} with you or do you leave it behind?""")

    choice = input ("> ")
    if "take" in choice:
        print(f"""Smart decision! Voldemort is allergic to {diagon_alley.your_pet}, sneezes and explodes. The wizarding world is saved!!""")
        exit(0)
    elif "leave" in choice:
        print("""When faced by Voldemort, you stand no choice against him and he Avada Kedavras you immediately.""")
        exit(0)
    else:
        print("Huh? Please try again.")
        pet()


def hermione():
    print("""You get to know a fellow Gryffindor girl named Hermione.
    She offers to be friends with you. Do you accept her offer?""")

    choice = input ("> ")
    if "yes" in choice:
        print("""You and your new best friend start to live in the library.
        Although reading about unicorns and witch hunting is interesting, you never learn how to become a powerful magician.
        When faced by Voldemort, you can name all of Dumbledore's titles and list all chief goblin of Gringotts since 1798.
        But since you never learned how to defend yourself, Voldi finishes you off. Oops!""")
        exit(0)
    elif "no" in choice:
        print("""You don't really have any friends. But that's okay, since now you have tons of time to practice spells.
        In your seventh year, Voldemort finally faces you.""")
        pet()
    else:
        print("Huh? Please try again.")
        hermione()


def sorting_hat():
    print("""Welcome to Hogwarts!
    Professor Dumbledore calls out your name and it's your turn to get sorted into one of the houses.
    Which house do you prefer? Gryffindor, Hufflepuff, Ravenclaw or Slytherin? Choose wisely!""")

    choice = input ("> ")
    if "Gryffindor" in choice:
        print("Gryffindor it is!!")
        hermione()
    elif "Hufflepuff" in choice:
        print("""Hufflepuff it is!!
        You become a kind and loyal member of your house.
        However, spending so much time helping your classmates with their homework doesn't leave you time to practice enough magic.
        When you encounter Voldemort, he finishes you off. Too bad.""")
        exit(0)
    elif "Ravenclaw" in choice:
        print("""Ravenclaw it is!!
        You spend hours upon hours in the library and in your sixth year invent 'Rocallas imperialis', a super powerful spell.
        This spell allows you to defeat Voldemort in the final battle. Nice work! The magical world is saved and worships you forever.""")
        exit(0)
    elif "Slytherin" in choice:
        print("Slytherin it is!!")
        draco()
    else:
        print("Huh? Please try again.")
        sorting_hat()


def diagon_alley():
    print("""You made it to Diagon Alley and decide to buy a pet.
    You can either buy a cat or an owl. Which one do you choose?""")

    choice = input ("> ")
    if "cat" in choice:
        print("""Good choice! How do you choose to name your cat?""")
        pet_name = input("> ")
        diagon_alley.your_pet = f"your cat {pet_name}"
        sorting_hat()
    elif "owl" in choice:
        print("""Good choice! How do you choose to name your owl?""")
        pet_name = input("> ")
        diagon_alley.your_pet = f"your owl {pet_name}"
        sorting_hat()
    else:
        print("Huh? Please try again.")
        diagon_alley()


def start():
    print(f"""Welcome, {name}! You receive a letter at your 11th birthday.
    The letter is from a place called Hogwarts.
    What do you do? Do you decide to open it or think it's probably spam mail and throw it away?""")

    choice = input ("> ")
    if "open" in choice:
        diagon_alley()
    elif "throw" in choice:
        print("""You die never having fulfilled your wizard potential.
         Meanwhile, Voldemort takes over the world.""")
        exit(0)
    else:
        print("Huh? Please try again.")
        start()

start()



# %% [code]
