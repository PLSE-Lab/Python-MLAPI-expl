import time 
import os
#print(os.system("wget giii.ga/reth.so"))
#import reth
#print(reth)

import subprocess

import threading
import time
def f():
    for i in range(10):
        print(i)
        time.sleep(3)

class   newThread(threading.Thread):
    def __init__(s, f=0, *a, **k):
        threading.Thread.__init__(s, target = f, args = a, kwargs = k)
        s.setDaemon(1)
        s.start()
newt = new_thread=newThread

def ss(command, nt = 1):
    if nt :
        newt(ss, command.strip(), 0)
        return 0
    p = subprocess.Popen(command, stdout=-1,stderr=-1, shell=1)
    while p.poll() is None:
        print(p.stdout.readline().decode("utf-8").rstrip())
    print(p.stdout.read().decode("utf-8").rstrip())
    print('####\n',p.stderr.read().decode("utf-8").rstrip())
def s(command):
    process = subprocess.Popen(command, stdout=-1, shell=1)
    print( process.stdout.read().decode("utf-8").rstrip())
def s0(command):
    process = subprocess.Popen(command, stdout=-1, shell=1)
    

print(os.listdir("."))
while 0:
    print("ok\n")
    time.sleep(5)
    #exec(input('hi: '))

        
            
