#Part 2 (binary and nonbinary logistic regression)

import numpy as np

eps = 0.08
eps_2 = 0.08

def Func2(c,b,a):
    return (np.log(np.exp(c+b)/np.exp(a)))

def Func(c ,b ):
    return (np.log(1 / (1 + np.exp(c + b))))

def Diff_dc(c, b):
    return ((Func(c + eps, b) - Func(c, b)) / eps)

def Diff_dc2(c,b,a):
    return (((Func2(c + eps, b,a) - Func2(c, b,a)) / eps))

def Diff_db(c, b):
    return ((Func(c, b + eps) - Func(c, b)) / eps)

def Diff_db2(c, b, a):
    return ((Func2(c, b + eps,a) - Func2(c, b, a)) / eps)


def Gradient(c, b):
    return np.sqrt((c ** 2) + (b ** 2))

def GradietnDescent2(x, c, b, a, h):
    counter = 0
    c = c*x
    base_c = c
    base_b = b
    while (counter < 10):
        dc = Diff_dc2(c, b, a)
        db = Diff_db2(c, b, a)
        gradient = Gradient(dc, db)
        if (abs(gradient) < eps):
            return Func2(c, b, a)
        else:
            c = c - h * dc
            b = b - h * db

        if (Func2(c, b,a) - Func2(base_c, base_b,a) < 0):
            if (((abs(c - base_c) and abs(b - base_b)) < eps_2) and (
                    Func2(c, b,a) - Func2(base_c, base_b,a))< eps_2):
                return Func2(c, b, a)
        else:
            h = h / 2
            c = c - h * dc
            b = b - h * db
        counter += 1

    return Func2(c, b, a)

def GradientDescent(x, c, b,  h):
    counter = 0
    c = c*x
    base_c = c
    base_b = b
    while (counter < 10):
        dc = Diff_dc(c, b)
        db = Diff_db(c, b)
        gradient = Gradient(dc, db)
        if (abs(gradient) < eps):
            break
        else:
            c = c - h * dc
            b = b - h * db

        if (Func(c, b) - Func(base_c, base_b) < 0):
            if (((abs(c - base_c) and abs(b - base_b)) < eps_2) and (
                    Func(c, b) - Func(base_c, base_b))< eps_2):
                print('final: ', c, b, Func(c, b))
        else:
            h = h / 2
            c = c - h * dc
            b = b - h * db
        counter += 1

    return Func(c, b)

def binary_logistic_regression():
    c = np.linspace(0,1,10)
    c = np.resize(c, (c.size,1))

    x = np.array([5,6,7,8,9,5,6,7,8,9])
    h = 0.1
    b = 6
    q = []
    counter = 0
    for i in range(c.size):
        counter+=1
        q.append(GradientDescent( x[i],c[i], b,  h))

    print('q',q)
    print(-1/c.size*np.sum(q))

def nonbinary_logistic_regression():
    x = np.array([8, 8, 9])
    x = np.resize(x,(x.size,1))
    b = np.array([6,1,2,3])
    b = np.resize(b,(b.size,1))
    c = np.linspace(0,1,(x.size*b.size)).reshape(x.size,b.size)

    q = []
    d = []
    h = 0.1
    for j in range(b.size):
        for i in range(x.size):
            d.append(x[i]*c[i][j] + b[j])
    a = np.array(d)
    a = np.resize(a,(x.size,b.size))
    for j in range(b.size):
        for i in range(x.size):
            q.append(GradietnDescent2(x[i], c[i][j], b[j], a[i][j], h))
        print(q)
        print(-1/len(q)*(np.sum(q)))
        
def main():
    print("Binary logistic regression:")
    binary_logistic_regression()
    print()
    
    print("Nonbinary logistic regression:")
    nonbinary_logistic_regression()

if __name__ == '__main__':
    main()        