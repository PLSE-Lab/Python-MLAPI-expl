src = '''
#include <iostream>

int main()
{
    std::cout << "Hello, world!\\n";
}
'''

with open('go.cpp', 'w') as srcfile:
    srcfile.write(src)

from subprocess import call

call(["g++", "--version"])
call(["g++", "-O3", "go.cpp", "-o", "go"])
call(["./go"])
