# Methodes de balayage pour chercher le minimum

import random
import numpy as np
import matplotlib.pyplot as plt

#B.1# -----------------

def f(x):
    return pow(x, 3) - 3 * pow(x, 2) + 2 * x + 5

def mf(x):
    return -(pow(x, 3) - 3 * pow(x, 2) + 2 * x + 5)

def chercheMinPasConstant(f, a, b, n):
    dx = abs(b - a) / n
    min = f(a)
    for i in range(n):
        if(f(a + dx * i) < min):
            min = f(a + dx * i)
    return min

def BalAlea(f,a,b,N):
    Val = f(a+ (b-a)*random.random())
    for i in range(N+1):
        N = f(a+(b-a)*random.random())
        if(Val>N):
            Val=N
    return Val

#B.2# -----------------

print(chercheMinPasConstant(f,0,3,1000))
print(BalAlea(f,0,3,1000))    

#B.3# ------------------
#À calculer à la main 
minreelle = (3 + np.sqrt(3)) / 6
n = 1000
a = 0
b = 3

pasConstantErreur = []
aleatoireErreur= []

for i in range(n):
    pasConstantErreur.append(abs(minreelle - chercheMinPasConstant(f, a, b, i + 1)))
    aleatoireErreur.append(abs(minreelle - BalAlea(f, a, b, i + 1)))

X = [k + 1 for k in range(n)]

#plt.plot(np.log(X), np.log(pasConstantErreur), 'r-', label="Erreur pour la methode pas constant")
#plt.plot(np.log(X), np.log(aleatoireErreur), 'b-', label="Erreur pour la methode pas aleatoire")
#plt.ylabel('ln(err)')
#plt.xlabel ('X')
#plt.legend()
#plt.show()


#B.4# -----------------

# pour chercher le maximum de f avec la meme methode
# mf représente -f
print(-chercheMinPasConstant(mf, a, b, n))
print(-BalAlea(mf, a, b, n))

#B.6# -----------------
# méthodes du gradiant 1D

x0 = a + (b - a) * random.random()

u = -0.01

def fder(x):
    return (3 * pow(x, 2) - 6 * x + 2)

for i in range(n):
    if(i == 0):
        xnp1 = x0 + u * fder(x0)
    else:
        xnp1 = xnp1 + u * fder(xnp1)

print(f(xnp1))




#B.7# -----------------
    #B.7.a# -----------------



    #B.7.b# -----------------




#C.1# -----------------