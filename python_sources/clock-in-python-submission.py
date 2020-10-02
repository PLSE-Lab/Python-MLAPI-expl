## How to make a Clock in python using Loops
#Variables
horas = 0
minutos = 0
segundos = 0

#Loops 
while horas < 24:
    while minutos < 60:
        while segundos < 60:
            print(f'{horas}:{minutos}:{segundos}')
            segundos += 1
        minutos +=1
        segundos =0
    horas += 1
    minutos= 0
