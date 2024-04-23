import torch
import argparse
import numpy as np
import scipy
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
from model import create_models
from model import train_models
from Init import create_dataset
from Init import create_opt
from Schéma import Meth_Mod
from Schéma import Meth_Non_Mod
from Schéma import Test
from Field import H
from Plot import plot_fsigma
from Plot import plot_loss

np.random.seed(42)
func = "Pendule"
Scheme = "MidPoint"
Rand1 = "PF"
Rand2 = "Ordre2"
Rand3 = "Gaussienne"
NB_POINT_TRAIN = 90
NB_POINT_TEST = 30

if torch.cuda.is_available():
    device = torch.device("cuda")    
    print("Utilisation du GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu") 
    print("utilisation du CPU")

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
        self.y = 0

    # Applique le schéma au champs de vecteur 
    def Applied(self,y0,h,T,traj,save_l,Grad):
        if self.Modified:
            y,l = Meth_Mod(self.models,y0,torch.tensor([h]),T,traj,self.Scheme,self.Rand,self.Field,Grad)
            self.y = y 
            if(save_l):
                self.l = l
        else:
            y,l = Meth_Non_Mod(y0,torch.tensor([h]),T,traj,self.Scheme,self.Rand,self.Field)
            self.y = y 
            if(save_l):
                self.l = l
    
    #Plot l'évolution de H sur la période sur laquelle le schéma est appliqué 
    def plot_H(self,y0,h):
        v = [H(y0,self.Field)]
        for i in range(1,len(self.l)):
            v.append((torch.mean(-torch.cos(self.l[i][0])+(self.l[i][1])**2/2)).detach())
        plt.plot([i*h for i in range(len(self.l))],v)

    def plot_ErrH(self,y0,h,label):
        v = [0]
        for i in range(1,len(self.l)):
            if self.Modified:
                v.append(torch.abs((torch.mean(-torch.cos(self.l[i][0][0])+(self.l[i][0][1])**2/2)).detach()-H(y0,self.Field)))
            else:
                v.append(torch.abs((torch.mean(-torch.cos(self.l[i][0])+(self.l[i][1])**2/2)).detach()-H(y0,self.Field)))
        plt.plot([i*h for i in range(len(self.l))],v,label=label)

    #Plot l'évolution de la phase sur la période sur laquelle le schéma est appliqué 
    def plot_Phase(self,y0,label):
        a = [y0[0]]
        b = [y0[1]]
        for i in range(1,len(self.l)):
            if self.Modified:
                a.append(self.l[i][0][0].detach())
                b.append(self.l[i][0][1].detach())
            else:
                a.append(self.l[i][0].detach())
                b.append(self.l[i][1].detach())
        plt.plot(a,b,label=label)

    def plot_traj(self,y0_batch):
        a = []
        b = []
        c = []
        d = []
        for y0 in y0_batch:
            d.append(self.models[0](y0[0].to(device))[0][0].to("cpu").detach())
            c.append(-y0[0][0][1]*torch.cos(y0[0][0][0])/4)
            a.append(y0[0][0][0])
            b.append(y0[0][0][1])
        plt.scatter(a,c,label=r'$q->f_{1app}^0(q,p)$')
        plt.scatter(a,d,label=r'$q->f_1^0(q,p)$')
        plt.legend()
        plt.savefig("q->f1-0.png")
        plt.show()
        plt.scatter(b,c,label=r'$p->f_{1app}^0(q,p)$')
        plt.scatter(b,d,label=r'$p->f_1^0(q,p)$')
        plt.legend()
        plt.savefig("p->f1-0.png")
        plt.show()

            

    def plot_err(self,y0,T,lh,ltrue,traj,label):
        if self.Field == "Linear":
            lambd = 2
            mu = 2
            b = torch.exp(torch.Tensor([lambd]))*y0
            E1 = []
            E2 = []
            for h in lh:
                self.Applied(y0,h,T,traj,False,False)
                a = torch.tensor([torch.mean(self.y[:,0])])
                E1.append(h)
                E2.append(torch.abs(a-b).detach())
            plt.plot(E1,E2,label=label)
        else:
            E1 = []
            E2 = []
            if(np.shape(ltrue)[-1] == 1):
                b = torch.tensor([torch.mean(ltrue[:,0])])
            else:
                b = torch.tensor([torch.mean(ltrue[:,0]),torch.mean(ltrue[:,1])])
            for h in lh:
                self.Applied(y0,h,T,traj,False,False)
                if(np.shape(ltrue)[-1] == 1):
                    a = torch.tensor([torch.mean(self.y[:,0])])
                else:
                    a = torch.tensor([torch.mean(self.y[:,0]),torch.mean(self.y[:,1])])
                    print(a)
                E1.append(h)
                if(h == 0.1):
                    print(torch.norm(a-b).detach())
                E2.append(torch.norm(a-b).detach())
            plt.plot(E1,E2,label=label)

def plot_err_linear(models):
    lambd = 1
    nb_traj = 10000000
    ytrue = torch.ones(10,2)*torch.tensor([0,1])*np.exp(lambd*1)*torch.tensor([1])
    h = [0.01,0.05,0.1,0.2,0.5]
    A1 = SchémaAppliqué(func,Scheme,Rand3,models)
    A2 = SchémaAppliqué(func,Scheme,Rand3,None)
    A1.plot_err(torch.tensor([0,1]),1,h,ytrue,nb_traj,r'$h -> |E[(\phi_h^{f_{app},\sigma_{app}})^{n}(y_0)] - E[y_L]|$')
    A2.plot_err(torch.tensor([0,1]),1,h,ytrue,nb_traj,r'$h -> |E[(\phi_h^{f,\sigma})^{n}(y_0)] - E[y_L]|$')
    plt.plot(h,h,label=r'$h->h$')
    plt.plot(h,[hi**2 for hi in h],label=r'$h->h^2$')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("WeakErr_linear.png")
    plt.show()

def plot_err_pendulum(models):
    nb_traj = 10000000
    h = [0.1,0.2,0.333333334]
    Atrue = SchémaAppliqué(func,Scheme,Rand3,None)
    Atrue.Applied(torch.tensor([0,1]),0.001,1,1000000,False,False)
    ytrue = Atrue.y
    A1 = SchémaAppliqué(func,Scheme,Rand3,models)
    A2 = SchémaAppliqué(func,Scheme,Rand3,None)
    A1.plot_err(torch.tensor([0,1]),1,h,ytrue,nb_traj,r'$h -> |E[(\phi_h^{f_{app},\sigma_{app}})^{n}(y_0)] - E[y_L]|$')
    A2.plot_err(torch.tensor([0,1]),1,h,ytrue,nb_traj,r'$h -> |E[(\phi_h^{f,\sigma})^{n}(y_0)] - E[y_L]|$')
    plt.plot(h,h,label=r'$h->h$')
    plt.plot(h,[hi**2 for hi in h],label=r'$h->h^2$')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("WeakErr_Pendulum.png")
    plt.show()



############# Main #############

def main():
    # On crée les 2 modèles pour l'instant un MLP pour f1 et un MLP pour Ra 
    models = create_models(2,func)

    # Pour l'instant on chosis un opt standard SGD et un learning rate moyen
    # Sujet à approfondir, influence du learning rate er de l'optimizer 

    y0_train, h_train, input2_train, Ey_train, Vy_train = create_dataset(NB_POINT_TRAIN,func,Rand3,1)
    training_set = [y0_train,h_train,input2_train, Ey_train, Vy_train]

    y0_test, h_test, input2_test, Ey_test, Vy_test = create_dataset(NB_POINT_TEST,func,Rand3,1)
    testing_set = [y0_test,h_test,input2_test, Ey_test, Vy_test]

    optimizer,all_parameters = create_opt(models,func)

    models, global_train_loss, global_test_loss = train_models(models,training_set,testing_set,optimizer,func,Rand3)

    epochs = torch.arange(0, 2)

    plot_loss(epochs,global_train_loss,global_test_loss)

    plot_err_pendulum(models)

    A = SchémaAppliqué(func,Scheme,Rand3,models)
    A2 = SchémaAppliqué(func,Scheme,Rand3,None)
    A.plot_traj(y0_train)
    A.Applied(torch.tensor([0,1]),0.05,5,1,True,False)
    A2.Applied(torch.tensor([0,1]),0.05,5,1,True,False)
    A.plot_Phase([0,1],r'Phase MidPoint Modifié')
    A2.plot_Phase([0,1],r'Phase MidPoint')
    plt.legend()
    plt.savefig("phase.png")
    plt.show()
    A.plot_ErrH(torch.tensor([0,1]),0.05,r'|H_app(y_t)-H(y_0)|')
    A2.plot_ErrH(torch.tensor([0,1]),0.05,r'|H(y_t)-H(y_0)|')
    plt.yscale("log")
    plt.legend()
    plt.savefig("ErrH.png")
    plt.show()





if __name__ == '__main__':
    main()
