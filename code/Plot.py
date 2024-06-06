import torch
import matplotlib.pyplot as plt
import numpy as np
from Schéma import Schéma
from Field import Field,ModifiedField,AnalyticModField
from Variables import NB_POINT_TRAIN,NB_POINT_TEST,EPOCHS
from Init import create_dataset,create_opt,Tubes
from model import create_models,train_models

if torch.cuda.is_available():
    device = torch.device("cuda")    
else:
    device = torch.device("cpu") 

# Plot l'évolution de la Loss 
def plot_loss(epochs,global_train_loss,global_test_loss,test_without):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs,global_train_loss,label="Train Loss")
    plt.plot(epochs,global_test_loss,label="Test Loss")
    if test_without != None:
        plt.plot(epochs,test_without,label="Base Loss")
    plt.yscale("log")
    plt.legend()

# Fonction qui plot fi si l'entrée est en 1D 
def plot_fi(y0_train,models,i,label):
    l1 = []
    l2 = []
    for y0_batch in y0_train: 
        y0_batch = torch.tensor(y0_batch)
        #inp = torch.cat((y0_batch,torch.tensor([0.1]))).to(device)
        l1.append(y0_batch)
        m = models[i-1](y0_batch.to(device)).to("cpu").detach()
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


# Fonction qui plot f1 f2 sigma1 sigma2 et ce qui est attendu en linéaire
def plot_fsigma(y0_train,models1,models2,Sys):
    # On determine jusqu'à quelle ordre plot 
    trunc = len(models1)-1
    # On crée le terme de Drift et Difussion modifiés avec les modèles entrainés
    f = ModifiedField(Sys.func1,models1)
    sigma = ModifiedField(Sys.func2,models2)
    # On crée les termes de Drift et Diffusion modifiés calculés analytiquement 
    f2 = AnalyticModField(Sys.func1,Sys.trunc)
    sigma2 = AnalyticModField(Sys.func2,Sys.trunc)
    # On compare f1 modifié par Machine learning et analytiquement
    plt.figure()
    f.print_modfield(1,y0_train)
    f2.print_ana(1,y0_train)
    plt.title(rf'$f_1$ $\lambda = {Sys.Lambd}$ $\mu = {Sys.Mu} $')
    plt.xlabel("x")
    plt.ylabel(r'$f_1(x) $')
    plt.legend()
    plt.savefig("f1.png")
    # On compare sigma1 modifié par Machine learning et analytiquement
    plt.figure()
    sigma.print_modfield(1,y0_train)
    sigma2.print_ana(1,y0_train)
    plt.title(rf'$\sigma_1$ $\lambda = {Sys.Lambd}$ $\mu = {Sys.Mu} $')
    plt.xlabel("x")
    plt.ylabel(r'$\sigma_1(x) $')
    plt.legend()
    plt.savefig("sigma1.png")
    if (trunc>1):
        # On compare f2 modifié par Machine learning et analytiquement
        plt.figure()
        f.print_modfield(2,y0_train)
        f2.print_ana(2,y0_train)
        plt.title(rf'$f_2$ $\lambda = {Sys.Lambd}$ $\mu = {Sys.Mu} $')
        plt.xlabel("x")
        plt.ylabel(r'$f_2(x) $')
        plt.legend()
        plt.savefig("f2.png")
        # On compare sigma2 modifié par Machine learning et analytiquement
        plt.figure()
        sigma.print_modfield(2,y0_train)
        sigma2.print_ana(2,y0_train)
        plt.title(rf'$\sigma_2$ $\lambda = {Sys.Lambd}$ $\mu = {Sys.Mu} $')
        plt.xlabel("x")
        plt.ylabel(r'$\sigma_2(x) $')
        plt.legend()
        plt.savefig("sigma2.png")


def Esperance(Systeme,nb_traj):
    if Systeme.func1 == "Linearf":
        ytrue = torch.ones(10,1)*torch.tensor(1)*np.exp(Systeme.Lambd*1)*torch.tensor(1)
    else:
        f = Field(Systeme.func1)
        sigma = Field(Systeme.func2)
        Atrue = Schéma(Systeme.Scheme,f,sigma,Systeme.Rand)
        # On calcul la référence avec un pas très petit 
        ytrue = torch.mean(Atrue.Applied(Systeme.y0,torch.tensor([0.0001]),Systeme.T,nb_traj,False),dim=0)
    return ytrue

# Plot l'erreur faible pour l'application d'un schéma simple et 
# d'un Schéma Modifié  
def plot_Weakerr(Systeme,models1,models2,nb_traj):
    lh = Systeme.LH
    y0 = Systeme.y0 
    T = Systeme.T 
    # On définit un schéma de référence Point Milieu simple
    ytrue = Esperance()
    # On définit Point milieu modifié en utilisant les models entrainés 
    fmod = ModifiedField(Systeme.func1,models1)
    sigmamod = ModifiedField(Systeme.func2,models2)
    A1 = Schéma(Systeme.Scheme,fmod,sigmamod,Systeme.Rand) 
    # On définit Point Milieu simple 
    f = Field(Systeme.func1)
    sigma = Field(Systeme.func2)
    A2 = Schéma(Systeme.Scheme,f,sigma,Systeme.Rand) 
    # On calcule l'erreur faible sur la plage de h 
    E2 = A1.Weakerr(ytrue,lh,y0,T,nb_traj)
    E3 = A2.Weakerr(ytrue,lh,y0,T,nb_traj)
    # On plot 
    plt.plot(lh,E2,label=r'$h -> |E[(\phi_h^{f_{app},\sigma_{app}})^{n}(y_0)] - E[y_L]|$')
    plt.plot(lh,E3,label=r'$h -> |E[(\phi_h^{f,\sigma})^{n}(y_0)] - E[y_L]|$')
    plt.plot(lh,lh,label=r'$h->h$')
    plt.plot(lh,[hi**2 for hi in lh],label=r'$h->h^2$')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("h")
    plt.ylabel("erreur faible")
    plt.legend()


# Fonction qui effectue un GridSchearch afin de determiner le LR qui donne la perte d'entrainement la plus basse 
def plot_lr(Sys):
    # Plage de LR qui sert au GridSchearch
    plage_lr = [1e-6,1e-5,1e-4,1e-3,1e-2]
    # Liste qui stocke la valeur de Loss pour chaque LR 
    loss_lr = []
    # Création du Training set, on garde le même pour chaque LR pour la consistence des résultats 
    y0_train, h_train, input2_train, Ey_train, Vy_train = create_dataset(NB_POINT_TRAIN,Sys,Sys.Dim)
    training_set = [y0_train,h_train,input2_train, Ey_train, Vy_train]
    # Création du Testing set, on garde le même pour chaque LR pour la consistence des résultats 
    y0_test, h_test, input2_test, Ey_test, Vy_test = create_dataset(NB_POINT_TEST,Sys,Sys.Dim)
    testing_set = [y0_test,h_test,input2_test, Ey_test, Vy_test]
    epochs = torch.arange(0, EPOCHS)
    # Pour chaque LR, on entraine un nouveau modèle sur un optimizer avec ce LR et on observe la Loss 
    for lr in plage_lr:
        models1 = create_models(Sys.Dim,Sys.func1,Sys.Trunc)
        models2 = create_models(Sys.Dim,Sys.func2,Sys.Trunc)
        optimizer,all_parameters = create_opt(models1,models2,lr)
        models1, models2, global_train_loss, global_test_loss, best_loss = train_models(models1, models2, training_set, testing_set, optimizer, Sys)
        loss_lr.append(best_loss)
    # On plot la Loss en fonction du LR 
    plt.plot(plage_lr,loss_lr)
    plt.xlabel("Learning Rate")
    plt.ylabel("Perte")
    plt.xscale("log")
    plt.title(r"Valeur de la perte en fonction de $\eta$")
    plt.savefig("LR.png")

# Fonction qui plot l'Elipse utilisé pour l'initialisation de y0
def plot_E():
    htheta = np.arange(0,2*np.pi+0.1,0.1)
    q = []
    p = []
    for theta in htheta:
        i = Tubes(theta,0,0)
        q.append(i[0])         
        p.append(i[1])    
    plt.plot(q,p)
    plt.show()

def plot_Phase(Systeme,models1,models2):
    fmod = ModifiedField(Systeme.func1,models1)
    sigmamod = ModifiedField(Systeme.func2,models2)
    A1 = Schéma(Systeme.Scheme,fmod,sigmamod,Systeme.Rand) 
    # On définit Point Milieu simple 
    f = Field(Systeme.func1)
    sigma = Field(Systeme.func2)
    A2 = Schéma(Systeme.Scheme,f,sigma,Systeme.Rand) 

    A1.Applied(torch.tensor([0,1]),0.1,20,1,True)
    A2.Applied(torch.tensor([0,1]),0.1,20,1,True)
    A1.plot_Phase(r'Phase Schéma Modifié')
    A2.plot_Phase(r'Phase Schéma Simple')
    plt.legend()


def plot_ErrH(Systeme,models1,models2):
    fmod = ModifiedField(Systeme.func1,models1)
    sigmamod = ModifiedField(Systeme.func2,models2)
    A1 = Schéma(Systeme.Scheme,fmod,sigmamod,Systeme.Rand) 
    # On définit Point Milieu simple 
    f = Field(Systeme.func1)
    sigma = Field(Systeme.func2)
    A2 = Schéma(Systeme.Scheme,f,sigma,Systeme.Rand) 

    A1.Applied(torch.tensor([0,1]),0.1,20,1,True)
    A2.Applied(torch.tensor([0,1]),0.1,20,1,True)
    A1.plot_ErrH(0.1,r'$|H_app(y_t)-H(y_0)|$')
    A2.plot_ErrH(0.1,r'$|H(y_t)-H(y_0)|$')
    plt.legend()
