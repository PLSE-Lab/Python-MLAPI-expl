import random
import numpy as np 

pi_t = random.random()
n = 1000
draws = [1 if x < pi_t else 0 for x in np.random.rand(n)]
x_bar = np.mean(draws)

def u(pi):
    return n*x_bar/pi - n*(1-x_bar)/(1-pi)
def J(pi):
    return -n*x_bar/pi**2 - n*(1-x_bar)/((1-pi)**2)
def I_hat(pi):
    return u(pi)**2
def Newton(pi):
    return pi - u(pi)/J(pi)
def Fisher(pi):
    return pi + u(pi)/I_hat(pi)

def dance(method_name, method):
    print("starting iterations for: " + method_name)
    pi, prev_pi, i = 0.5, None, 0
    while i == 0 or (abs(pi-pi_t) > 0.001 and abs(pi-prev_pi) > 0.001 and i < 10):
        prev_pi, pi = pi, method(pi)
        i += 1
        print(method_name, i, "delta: ", abs(pi-pi_t))

dance("Newton", Newton)
dance("Fisher", Fisher)