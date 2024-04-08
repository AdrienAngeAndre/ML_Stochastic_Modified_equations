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
from Field import H

np.random.seed(42)
func = "Pendule"
Scheme = "MidPoint"
Rand1 = "PF"
Rand2 = "Ordre2"

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
    def Applied(self,y0,h,T,traj):
        if self.Modified:
            y,l = Meth_Mod(self.models,y0,torch.tensor([h]),T,traj,self.Scheme,self.Rand)
            self.y = y 
            self.l = l
        else:
            y,l = Meth_Non_Mod(y0,torch.tensor([h]),T,traj,self.Scheme,self.Rand)
            self.y = y 
            self.l = l
    
    #Plot l'évolution de H sur la période sur laquelle le schéma est appliqué 
    def plot_H(self,y0,h):
        v = [H(y0,self.Field)]
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
            a = torch.tensor([torch.mean(self.y[:,0]),torch.mean(self.y[:,1])])
            E1.append(h)
            E2.append(torch.norm(a-b).detach())
        plt.plot(E1,E2)


############# Main #############

def main():

    # lambda²/ 2      (2lamb*mu² + mu^2/2)
    Atrue = SchémaAppliqué(func,Scheme,Rand2,None)
    A1 = SchémaAppliqué(func,Scheme,Rand1,None)
    A2 = SchémaAppliqué(func,Scheme,Rand2,None)
    nb_traj = 50000
    h = [0.05,0.1,0.2,0.5]
    Atrue.Applied(torch.tensor([0,1]),0.001,1,nb_traj)
    ytrue = Atrue.y
    A1.plot_err(torch.tensor([0,1]),1,h,ytrue,nb_traj)
    A2.plot_err(torch.tensor([0,1]),1,h,ytrue,nb_traj)
    plt.plot(h,h)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    # On crée les 2 modèles pour l'instant un MLP pour f1 et un MLP pour Ra 
    # models = create_models(2,func)

    # Pour l'instant on chosis un opt standard SGD et un learning rate moyen
    # Sujet à approfondir, influence du learning rate er de l'optimizer 


    # y0_train, h_train, input2_train, Ey_train, Vy_train = create_dataset(Rand1)

    # optimizer,all_parameters = create_opt(models,func)
    
    # training_set = [y0_train,h_train,input2_train, Ey_train, Vy_train]

    # models, global_loss = train_models(models,training_set,optimizer,func,Rand1)


    # h = 0.05
    # l = Esymp_mod3(models,torch.tensor([0,1]),torch.tensor([h]),10,1)[1]
    # l2 = ESymp(torch.tensor([0,1]),10,torch.tensor([h]),1)[1]
    # v = [0]
    # v2 = [0]
    # v3 = [0]
    # w = [-1/2]
    # a = [0]
    # b = [1]
    # for i in range(1,len(l)):
    #     a.append(torch.mean(l[i][:,0]).detach())
    #     b.append(torch.mean(l[i][:,1]).detach())
    # plt.plot(a,b)
    # plt.show()
    # for i in range(1,len(l)):
    #     v.append(torch.abs((torch.mean(-torch.cos(l[i][:,0])+(l[i][:,1])**2/2)).detach()+1/2))
    #     v2.append(torch.abs((torch.mean(-torch.cos(l2[i][0])+(l2[i][1])**2/2)).detach()+1/2))
    #     # w.append((torch.mean(w[0]+h/2*l[i][:,1]*torch.sin(l[i][:,0]))).detach())
    # plt.plot([i*h for i in range(len(l))],v,label="SSV - f_app")
    # plt.plot([i*h for i in range(len(l))],v2,label="SSV - f")
    # plt.yscale("log")
    # plt.legend()
    # # plt.plot([i*h for i in range(len(l))],w)
    # plt.show()
    
    # if function == 'Non_linear_Pendulum':
    #     y0 = torch.tensor([-1.5,0])
    #     h = torch.tensor([0.1])
    #     T = 10
    #     plot_NL(models,y0,h,T,trunc,scheme,function)
    #     plot_traj2D(NL_pendule,y0,h,T)
    # elif function == 'Rigid_Body':
    #     y0 = torch.tensor([0.15,1,0.85])
    #     h = torch.tensor([0.01])
    #     T = 12
    #     plot_traj3D(RBody,y0,h,T)

    # t = torch.tensor([0.00])
    
    # y = torch.tensor([1])

    

    # m = [y]
    # n = [0]
    # time = [0]
    # N = 2**8
    # h = 1/N
    # dW = np.sqrt(h) * np.random.randn(N)
    # W = np.cumsum(dW)
    # TF = 1
    # t = np.arange(0, TF, h)
    # R = 1
    # Dt = R * h
    # L = int(N / R)
    # Xtrue = y * np.exp((lambd - 0.5 * mu**2) * np.arange(0, TF, h) + mu * W)
    # Xem = np.zeros(L)
    # Xem2 = np.zeros(L)
    # # Applique Euler Maruyama à notre fonction modifiée 
    # Xtemp = y
    # for j in range(L):
    #     i = torch.cat((Xtemp,torch.tensor([h])))
    #     Winc = np.sum(dW[R*(j-1):R*j])
    #     Xtemp = Xtemp + Dt*(Xtemp+Dt*models[0](i))*lambd + mu*(Xtemp+Dt*models[1](i))*Winc
    #     Xem[j] = Xtemp
    # t = np.arange(0, TF, h)
    # # Applique Euler Maruyama à notre fonction de base 
    # Xtemp2 = y
    # for j in range(L):
    #     i = torch.cat((Xtemp2,torch.tensor([h])))
    #     Winc = np.sum(dW[R*(j-1):R*j])
    #     Xtemp2 = Xtemp2 + Dt*(Xtemp2)*lambd + mu*(Xtemp2)*Winc
    #     Xem2[j] = Xtemp2


    # plt.plot(t,Xtrue, label = "XTrue calulé par la formule connu dans ce cas")
    # plt.plot(t,Xem2, label = "X simulé en appliquant Euler aux fonction de base ")
    # plt.plot(t,Xem, label = "X simulé en appliquant Euler aux fonction modifiés ")
    # plt.xlabel("t")
    # plt.ylabel("X")
    # plt.show()
    

    # M = 300000

    # TF = 0.25
    # Xem = np.zeros(5)
    # for p in range(1, 6):
    #     Dt = 2**(p-10)
    #     L = int(TF/Dt)
    #     Xtemp = y * np.ones(M)

    #     for j in range(L):
    #         Winc = np.sqrt(Dt) * np.random.randn(M)
    #         Xtemp = Xtemp + Dt*(Xtemp)*lambd + mu*(Xtemp)*Winc

    #     Xem[p-1] = torch.mean(Xtemp)

    # Xerr = np.abs(Xem - np.exp(lambd*TF))
    # Dtvals = 2.0**np.arange(1, 6) / 2**10

    # A = np.column_stack((np.ones(5), np.log(Dtvals)))
    # rhs = np.log(Xerr)
    # sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    # q = sol[1]
    # resid = np.linalg.norm(A.dot(sol) - rhs)
    # print(f"q = {q}")
    # print(f"residual = {resid}")
    # plt.loglog(Dtvals, Xerr, 'b*-', label="erreur pour la fonction de base")
    
    # # Calcul l'esperance E(X_L) - E(X(T)) pour 
    # Xem = np.zeros(5)
    # for p in range(1, 6):
    #     Dt = 2**(p-10)
    #     L = int(TF/Dt)
    #     Xtemp = y * np.ones((M, 1))
    #     print(p)
    #     print(L)
    #     for j in range(L):
    #         print(j)
    #         i = torch.cat((Xtemp, Dt * torch.ones((M, 1))), axis=1).to(torch.float32)
    #         Winc = np.sqrt(Dt) * np.random.randn(M, 1)
    #         Xtemp = Xtemp + Dt * (lambd*Xtemp + Dt * models[0](i).detach()+ Dt**2 * models[1](i).detach())  + (mu*Xtemp + Dt * models[2](i).detach() + Dt**2 * models[3](i).detach()) * Winc
    #     Xem[p-1] = torch.mean(Xtemp)
    
    # Xerr = np.abs(Xem - np.exp(lambd*TF))
    # Dtvals = 2.0**np.arange(1, 6) / 2**10
    # A = np.column_stack((np.ones(5), np.log(Dtvals)))
    # rhs = np.log(Xerr)
    # sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    # q = sol[1]
    # resid = np.linalg.norm(A.dot(sol) - rhs)
    # print(f"q = {q}")
    # print(f"residual = {resid}")
    # plt.loglog(Dtvals, Xerr, 'g*-', label="erreur pour la fonction modifiée")

    # plt.loglog(Dtvals, Dtvals, 'r--', label="fonction identité")   
    # plt.xlabel(r'$\Delta t$')
    # plt.ylabel(r'$| E(X(T)) - \text{Sample average of } X_L |$')
    # plt.legend()



    # while (t<TF):
    #     i = torch.cat((y,h))
    #     x = y*torch.ones(1000000) + h*(y+h*models[0](i))*torch.ones(1000000)+torch.sqrt(h)*(y+h*models[1](i))*torch.randn(1000000)
    #     Vy = torch.var(x).detach().unsqueeze(0)
    #     Ey = torch.mean(x).detach().unsqueeze(0)
    #     m.append(Ey)
    #     n.append(Vy)
    #     t += h
    #     time.append(t.item())
    # plt.plot(time,n)
    # l1 = []
    # l2 = []
    # for i in range(len(time)):
    #     l1.append(np.exp(time[i]*lambd))
    #     l2.append(np.exp(2*lambd*time[i])*(np.exp(mu**2*time[i])-1))
    # # plt.plot(time,l1)
    # # plt.plot(time,l2)
    # plt.show()


if __name__ == '__main__':
    main()
