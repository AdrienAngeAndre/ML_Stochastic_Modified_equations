import torch
import matplotlib.pyplot as plt
import numpy as np

from Schéma import RK4
from Init import Elipse
from Field import NL_pendule
from Field import H
from Schéma import Meth_Non_Mod
from Schéma import Meth_Mod

if torch.cuda.is_available():
    device = torch.device("cuda")    
else:
    device = torch.device("cpu") 


Type = "Linear"
Meth = "MidPoint"

# Plot l'évolution de la Loss 
def plot_loss(epochs,global_train_loss,global_test_loss):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs,global_train_loss,label="Train Loss")
    plt.plot(epochs,global_test_loss,label="Test Loss")
    plt.savefig(f'train_loss.png')
    plt.yscale("log")
    plt.legend()
    plt.show()

# Fonction qui plot f1, f2 et f3, pas très adapté au cas stochastique, à refaire 
def plot_f(y0_train,models,trunc):
    plot_fi(y0_train,models,1,"a")
    plot_fi(y0_train,models,2,"a")
    plot_fi(y0_train,models,3,"a")
    plt.title(f'fi en fonction de y0 pour un f_app tronquer à f{trunc}')
    plt.xlabel("y0")
    plt.ylabel("fi(y0)")
    plt.legend()

# Fonction qui plot fi si l'entrée est en 1D 
def plot_fi(y0_train,models,i,label):
    l1 = []
    l2 = []
    for y0_batch in y0_train: 
        y0_batch = torch.tensor(y0_batch)
        inp = torch.cat((y0_batch,torch.tensor([0.01]))).to(device)
        l1.append(y0_batch)
        m = models[i-1](inp).to("cpu").detach()
        l2.append(m)
    plt.scatter(l1,l2,label=label)
    return l1

# Fonction qui plot chaque composante de fi si l'entrée est en 2D 
def plot_fi_2D(y0_train,models,i,label):
    l1 = []
    l2 = []
    l3 = []
    for y0_batch in y0_train: 
        y0_batch = torch.tensor(y0_batch)
        inp = torch.cat((y0_batch,torch.tensor([0.01])))
        input21 = inp[[0, 2]]
        input22 = inp[1:3]
        l1.append(y0_batch)
        m = models[i-1][0](input21).detach()
        l2.append(m)
        n = models[i-1][1](input22).detach()
        l3.append(n)
    plt.subplot(2,1,1)
    plt.scatter(l1,l2,label=label)
    plt.subplot(2,1,2)
    plt.scatter(l1,l3,label=label)
    plt.show()

# Fonction qui utilise plot la trajectoire calculé avec RK4 
def plot_traj2D(f,y0,h,T):
    l = RK4(f,y0,h,T)
    l1 = []
    l2 = []
    for i in l:
        l1.append(i[0])
        l2.append(i[1])
    plt.plot(l1,l2)
    plt.show()

# Fonction qui utilise plot la trajectoire calculé avec RK4 en 3D 
def plot_traj3D(f,y0,h,T):
    l = RK4(f,y0,h,T)
    l1 = []
    l2 = []
    l3 = []
    for i in l:
        l1.append(i[0])
        l2.append(i[1])
        l3.append(i[2])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(l1,l2 ,l3 , 'gray')
    plt.show()

# Fonction qui plot la trajectoire calculé 
def plot_NL(models,y0,h,T,trunc,scheme,function):
    t = 0
    l = [y0]
    y = y0
    while(t<T):
        i = torch.cat((y,h))
        Ra = models[trunc](i)
        # On calcul ensuite la valeur en appliquant la méthode d'Euler à la troncature fapp
        f_app = torch.tensor(NL_pendule(t,y))
        for j in range(trunc):
            f_app = f_app + (h**(j+1)*models[j](y)).detach()
        f_app = f_app + h**(trunc+1)*Ra
        if scheme == "Euler_ex":
            y = (y + h*f_app)  
        t += h 
        l.append(y.detach())
    l1 = []
    l2 = []
    for i in l:
        l1.append(i[0])
        l2.append(i[1])
    plt.scatter(l1,l2)


# Fonction qui plot f1 f2 sigma1 sigma2 et ce qui est attendu en linéaire
def plot_fsigma(y0_train,lambd,mu,models):
    plt.subplot(2,2,1)
    l1 = plot_fi(y0_train,models,1,r"f_1 approché par le réseau de neurones")
    l1 = np.array(l1)
    plt.plot(l1,lambd**2/2*l1,label=r"f_1=\frac{\lambda**2}{2}",color="orange")
    plt.title(r'$f_1$ $\lambda = 2$ $\mu = 2 $')
    plt.xlabel("x")
    plt.ylabel(r'$f_1(x) $')
    plt.subplot(2,2,2)
    plot_fi(y0_train,models,2,r"f_2 approché par le réseau de neurones")
    plt.plot(l1,lambd**3/6*l1,label=r"f_1=\frac{\lambda**3}{6}",color="orange")
    plt.title(r'$f_2$ $\lambda = 2$ $\mu = 2 $')
    plt.xlabel("x")
    plt.ylabel(r'$f_2(x) $')
    plt.subplot(2,2,3)
    plot_fi(y0_train,models,4,r"\sigma_1 approché par le réseau de neurones")
    plt.scatter(l1,(mu**4/2+2*lambd*mu**2)*l1**2,label=r"f_1=\frac{\mu**4}{2}+2\lambda\mu^2",color='orange')
    plt.title(r'$\sigma_1$ $\lambda = 2$ $\mu = 2 $')
    plt.xlabel("x")
    plt.ylabel(r'$\sigma_1(x) $')
    plt.subplot(2,2,4)
    plot_fi(y0_train,models,5,r"\sigma_2 approché par le réseau de neurones")
    plt.scatter(l1,(mu**6/6+lambd*mu**4+2*lambd**2*mu**2)*l1**2,label=r"f_1=\frac{\mu**6}{6}+\lambda\mu^4+2\lambda^2\mu^2",color='orange')
    plt.title(r'$\sigma_2$ $\lambda = 2$ $\mu = 2 $')
    plt.xlabel("x")
    plt.ylabel(r'$\sigma_2(x) $')
    plt.legend()
    plt.show()

# Fonction qui plot l'Elipse utilisé pour l'initialisation de y0
def plot_E():
    htheta = np.arange(0,2*np.pi+0.1,0.1)
    q = []
    p = []
    for theta in htheta:
        i = Elipse(theta,0,0)
        q.append(i[0])         
        p.append(i[1])    
    plt.plot(q,p)
    plt.show()


    