import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from Variables import DEVICE


# Fonction calculant xi
def init_rand(type,traj):
    if type == "Gaussienne":
        return torch.randn(traj) 
    elif type == "PF":
        return 2*torch.randint(0, 2, size=(traj,))-1
    elif type == "O2":
        return rand2(traj)

# Type d'initialisation de xi 
def rand2(traj):
    probs = torch.tensor([1/6, 1/6, 2/3])
    distribution = torch.distributions.Categorical(probs)

    samples = distribution.sample((traj,))

    values = torch.zeros(traj)

    # Utilisation de masques pour définir les valeurs en une seule opération
    values[samples == 0] = -torch.sqrt(torch.tensor(3.))
    values[samples == 1] = torch.sqrt(torch.tensor(3.))
    values[samples == 2] = torch.tensor(0)

    return values

# Fonction calculant y1 à l'aide d'un point fixe pour Point Milieu 
def PFixe(q0,h,rand,schéma):
    with torch.no_grad():
        q = q0
        aux = q0
        r = rand.to(DEVICE)
        h = h.to(DEVICE)
        q = q.to(DEVICE)
        aux = aux.to(DEVICE)
        traj = np.shape(rand)
        save = aux+1
        N = 0
        while N<10 and torch.max(torch.abs(aux-save))>10**(-7):
            save = aux
            if schéma.f.Field == schéma.sigma.Field :
                aux = q + (h+torch.sqrt(h)*r)*schéma.f.evaluate((aux+q)/2,h)
            else:
                aux = q + (h*schéma.f.evaluate((aux+q)/2,h)+torch.sqrt(h)*schéma.sigma.evaluate((aux+q)/2,h)*r)
            N += 1
    return aux

# Classe représentant l'application d'un schéma numérique modifié
class Schéma:
    # Initialisation d'un schéma modifié, on donne l'intégrateur (EM,Midpoint,...)
    # les termes des drift et diffusions et la manière dont on calcul xi  
    def __init__(self,Scheme,f,sigma,Rand):
        self.Scheme = Scheme 
        self.f = f
        self.sigma = sigma 
        self.Rand = Rand
        # Ce paramètre sauve l'application du schéma afin d'éviter de devoir reapliquer
        # le schéma si on plot plusieurs metrics 
        self.l = []

    # Applique un pas du schéma  
    def step(self,y0,h,rand):
        r = rand.unsqueeze(-1).to(DEVICE)
        # Euler Maruyama Basique 
        if self.Scheme == "EMaruyama":
            if(np.shape(y)[0] == 1):
                y = y0.repeat(np.shape(r)[0],1) + h.repeat(np.shape(r)[0],1)*self.f.evaluate(y0,h) + torch.sqrt(h)*self.sigma.evaluate(y0,h)*r
            else:
                y = y0 + h*self.f.evaluate(y0,h) + torch.sqrt(h)*self.sigma.evaluate(y0,h)*r
            return y 
        # Euler Maruyama réécrit avec sigma tilde sous la racine 
        elif self.Scheme == "EMaruyamaLinear":
            if(np.shape(y0)[0] == 1):
                y = y0.repeat(np.shape(r)[0],1) + h.repeat(np.shape(r)[0],1)*self.f.evaluate(y0,h) + torch.sqrt(h)*torch.sqrt(torch.abs(self.sigma.evaluate(y0,h)))*r
            else:
                y = y0 + h*self.f.evaluate(y0,h) + torch.sqrt(h)*torch.sqrt(torch.abs(self.sigma.evaluate(y0,h)))*r
            return y
        # Point milieu stochastique 
        elif self.Scheme == "MidPoint":
            y0 = y0.to(DEVICE)
            h = h.to(DEVICE)
            y1 = PFixe(y0,h,r,self)
            if self.f.Field == self.sigma.Field:
                y = y0 + (h + torch.sqrt(h)*r)*self.f.evaluate((y0+y1)/2,h)
            else:
                y = y0 + h*self.f.evaluate((y0+y1)/2,h) + torch.sqrt(h)*self.sigma.evaluate((y0+y1)/2,h)*r
            return y

    # Applique le schéma jusqu'a un temps T 
    def Applied(self,y0,h,T,traj,save_l):
        y = y0
        l = [y0]
        h_comp = h
        with torch.no_grad():
            for t in np.arange(h,T+h,h):
                rand = init_rand(self.Rand,traj)
                if t == h_comp:
                    y = y.repeat(traj,1)
                    h = h.repeat(traj,1)
                l.append(torch.mean(y,dim=0))
                y = self.step(y,h,rand)
        if save_l:
            l.append(torch.mean(y,dim=0))
            self.l = l 
        return y 
    
    # Plot l'erreur faible pour le schéma 
    def Weakerr(self,ytrue,lh,y0,T,nb_traj):
        E1 = []
        # On parcoure la plage des h 
        for h in lh :
            y1 = torch.mean(self.Applied(y0,torch.tensor([h]),T,nb_traj,False),dim=0)
            E1.append(torch.norm(y1-ytrue).to("cpu"))
        return E1 

    #Plot l'évolution de H sur la période sur laquelle le schéma est appliqué 
    def plot_H(self,h):
        v = []
        for i in range(len(self.l)):
            v.append(-torch.cos(self.l[i][0])+(self.l[i][1])**2/2).detach()
        plt.plot([i*h for i in range(len(self.l))],v)

    # Plot l'erreur sur entre H(y) et H(y0) sur laquelle le schéma est appliqué 
    def plot_ErrH(self,h,label):
        v = []
        H0 = (torch.mean(-torch.cos(self.l[i][0])+(self.l[i][1])**2/2))
        for i in range(0,len(self.l)):
            v.append(torch.abs((torch.mean(-torch.cos(self.l[i][0])+(self.l[i][1])**2/2))-H0))
        plt.plot([i*h for i in range(len(self.l))],v,label=label)

    #Plot l'évolution de la phase sur la période sur laquelle le schéma est appliqué 
    def plot_Phase(self,label):
        a = []
        b = []
        for i in range(0,len(self.l)):
                a.append(self.l[i][0])
                b.append(self.l[i][1])
        plt.plot(a,b,label=label)

    

    





