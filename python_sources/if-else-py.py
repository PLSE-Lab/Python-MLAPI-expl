print("bau")

user = input("Vrei sa inveti (type: da, nu, poate)")
if user == "da":
    print("ce vrei sa inveti :")
    print("1) prog c")
    print("2) prog c#")
    print("3) prog py")

user = input("ce program vrei sa inveti? (type: c or c# or py)")
if user == "c":
    print("Este o alegere buna pentru inceput")
elif user == "c#":
    print("Este o alegere buna daca vrei sa faci aplicatii")
elif user == "py":
    print ("Este bun daca vrei sa inveti ML")
elif user == "poate":
    print("Gandestete bine")
elif user == "nu":
    print("o zi buna")
else:
    print("Alegerea dumneavoastra nu este corecta") 