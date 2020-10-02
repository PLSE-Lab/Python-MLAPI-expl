'''
ASCII (American Standard Code for Information Interchange) is the most common format for text files in computers
and on the Internet. In an ASCII file, each alphabetic, numeric, or special character is represented with a 7-bit
binary number (a string of seven 0s or 1s). 128 possible characters are defined.

UNIX and DOS-based operating systems use ASCII for text files. Windows NT and 2000 uses a newer code, Unicode.
IBM's S/390 systems use a proprietary 8-bit code called EBCDIC. Conversion programs allow different operating 
systems to change a file from one code to another.

ASCII was developed by the American National Standards Institute (ANSI).
'''

#ASCII Number to ACIII Character
def a2c(num):
    return chr(num)
#ASCII Character to ACIII Number
def c2a(char):
    return ord(char)

'''
Letters : ASCII Range
---------------------
    A-Z : 65-90
    a-z : 97-122
    0-9 : 48-57
'''

print(a2c(97))
print(c2a('a'))


