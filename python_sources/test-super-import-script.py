with open("the_script.py", "w") as f:
    f.write('if __name__ == "__main__":\n')
    f.write('    print("CCCCC!!")\n')
    f.write('else:\n')
    f.write('    print("I\'M BEING IMPORTED!!")\n')
    f.write('def f():\n')
    f.write('    print("I\'M BEING INVOKED!!")\n')