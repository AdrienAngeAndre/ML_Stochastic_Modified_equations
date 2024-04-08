import torch
import matplotlib.pyplot as plt
import numpy as np

from Schéma import RK4
from Init import Elipse
from Field import NL_pendule
from Field import H
from Schéma import Meth_Non_Mod
from Schéma import Meth_Mod


# Classe représentant l'application d'un schéma numérique 
class SchémaAppliqué:
    # Fonction d'Initialisation Field est le champs de vecteur utilisé et Scheme le schéma numérique utilisé 
    # models corresponds aux MLPs si le champs de vecteur est modifié sinon None 
    def __init__(self,Field,Scheme,Rand,models):
        self.Field = Field
        self.Scheme = Scheme
        self.Rand = Rand
        self.models = models
        if models == None:
            self.Modified = False
        else:
            self.Modified = True
        self.l = []

    # Applique le schéma au champs de vecteur 
    def Applied(self,y0,h,T,traj):
        if self.Modified:
            self.l = Meth_Mod(self.models,y0,h,T,traj,self.Scheme)[1]
        else:
            self.l = Meth_Non_Mod(y0,h,T,traj,self.Scheme)[1]
    
    #Plot l'évolution de H sur la période sur laquelle le schéma est appliqué 
    def plot_H(self,y0,h):
        v = [H(y0,Type)]
        for i in range(1,len(self.l)):
            v.append((torch.mean(-torch.cos(self.l[i][0])+(self.l[i][1])**2/2)).detach())
        plt.plot([i*h for i in range(len(self.l))],v)

    #Plot l'évolution de la phase sur la période sur laquelle le schéma est appliqué 
    def plot_Phase(self):
        a = []
        b = []
        for i in range(0,len(self.l)):
            a.append(torch.mean(self.l[i][0]).detach())
            b.append(torch.mean(self.l[i][1]).detach())
        plt.plot(a,b)

    def plot_err(self,y0,T,lh,ltrue,traj):
        E1 = []
        E2 = []
        b = torch.tensor([torch.mean(ltrue[:,0]),torch.mean(ltrue[:,1])])
        for h in lh:
            self.Applied(y0,h,T,traj)
            a = torch.tensor([torch.mean(self.l[:,0]),torch.mean(self.l[:,1])])
            E1.append(h)
            E2.append(torch.norm(a-b).detach())
        plt.plot(E1,E2)



    


Type = "Linear"
Meth = "MidPoint"

# Plot l'évolution de la Loss 
def plot_loss(epochs,global_loss):
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Evolution de la perte d'entrainement en fonction des périodes")
    plt.plot(epochs,global_loss)
    plt.savefig(f'training_loss.png')

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
        inp = torch.cat((y0_batch,torch.tensor([0.01])))
        l1.append(y0_batch)
        m = models[i-1](inp).detach()
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


    