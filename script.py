import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib
matplotlib.use('Qt5Agg')
import math
from matplotlib.animation import FuncAnimation

L = 20
res = 200
x1 = np.linspace(-L,L,res)
x2 = np.linspace(-L,L,res)

dx = x1[1]-x1[0]
dy = x2[1]-x2[0]

X,Y = np.meshgrid(x1,x2)

dt = 0.0005
hbar=2
m = 1

k=2
std = 3
def CI():
    #return np.exp(-((X-8)**2+(Y)**2)/(2*std**2))*np.exp(Y*500j)/(2*np.pi*std)
    return 4/L**2*np.cos(3*np.pi/(2*L)*X)*np.cos(3*np.pi/(2*L)*Y)
psi_t = [CI().astype("complex")]


def Calc(args):
    i,psi_n,psi_t_l,V = args
    psi_na = np.zeros_like(psi_n[0])
    for j in range(res):
        #Condiciones de frontera
        if i==0 or j==0 or i==res-1 or j==res-1:
            psi_na[j]=0
        else:
            #Primero calculamos la segunda derivada con respecto a x y y
            s2psi1 = (psi_t_l[i+1,j]-2*psi_t_l[i,j]+psi_t_l[i-1,j])/dx**2
            s2psi2 = (psi_t_l[i,j+1]-2*psi_t_l[i,j]+psi_t_l[i,j-1])/dy**2
            #Calculamos el siguiente paso de la ecuacion de sch
            psi_na[j] = psi_t_l[i,j]-1j*dt*(-hbar**2/(2*m)*(s2psi1+s2psi2)+V[i,j]*psi_t_l[i,j])
    return psi_na


V = -k/np.sqrt(X**2+Y**2)
#V = k*X+k*Y

def StepCalc(pool):
    psi_n = np.zeros_like(psi_t[-1])

    if res>120:
        args = [(i,psi_n,psi_t[-1],V) for i in range(res)]
        vls = pool.map(Calc,args)
        vls = np.array(vls,"complex")
        psi_n=vls
        #Calc(i,psi_n,psi_t[-1],V)
    else:
        for i in range(res):
            for j in range(res):
                #Condiciones de frontera
                if i==0 or j==0 or i==res-1 or j==res-1:
                    psi_n[i,j]=0
                else:
                    #Primero calculamos la segunda derivada con respecto a x y y
                    s2psi1 = (psi_t[-1][i+1,j]-2*psi_t[-1][i,j]+psi_t[-1][i-1,j])/dx**2
                    s2psi2 = (psi_t[-1][i,j+1]-2*psi_t[-1][i,j]+psi_t[-1][i,j-1])/dy**2
                    #Calculamos el siguiente paso de la ecuacion de sch
                    psi_n[i,j] = psi_t[-1][i,j]-1j*dt*(-hbar**2/(2*m)*(s2psi1+s2psi2)+V[i,j]*psi_t[-1][i,j])
    #Para evitar que la simulacion haga puash
    norm = np.sqrt(np.sum(np.abs(psi_n)**2) * dx * dy)
    psi_n = psi_n/norm
    psi_t.append(psi_n)

steps = 1000
if res>120:
    with Pool() as pool:
        for step in range(steps):
            StepCalc(pool)
            print("Paso {}".format(step+1))
else:
    for step in range(steps):
        StepCalc(None)
        print("Paso {}".format(step+1))

fig = plt.figure()
ax = fig.add_subplot()


def animate(t):
    ax.clear()
    ax.pcolormesh(X,Y,np.abs(psi_t[t])**2,cmap="gist_gray")
    print(t)


ani = FuncAnimation(fig,animate,frames=steps,interval=0.01)
ani.save("animation.mp4","ffmpeg",fps=60)
plt.show()
