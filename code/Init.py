import torch
import time
import numpy as np
import torch.utils.data as data
from Schéma import Schéma
from Field import Field
from Variables import BS,MU,LAMBDA,FUNC1,FUNC2,DECAY,NB_TRAJ_INIT,DEVICE


# Fonction créant l'optimiser Adam 
def create_opt(models1,models2,lr):
    all_parameters = []
    # On parcoure les MLPs des termes de drifts et diffusions et on les mets sur GPU + 
    # on donne à l'optimiser les paramètres 
    for model in models1:
        model.to(DEVICE)
        all_parameters += list(model.parameters())
    for model in models2:
        model.to(DEVICE)
        all_parameters += list(model.parameters())
    # Création de l'optimiser Adam 
    optimizer = torch.optim.Adam(all_parameters, lr=lr,weight_decay=DECAY) 
    return optimizer,all_parameters


# Fonction d'initialisation pour le cas Linéaire
def init_Linear(nb_point):
    # On choisis de manière uniforme les y0 et h 
    y0 = torch.rand(nb_point, 1)*2
    log_h = torch.rand(nb_point, 1)*3-3
    h = torch.exp(log_h)
    input2 = torch.cat((y0,h),1)
    # On utilise les expressions exact de l'esperance et de la variance 
    Ey = torch.exp(LAMBDA*h)*y0/h**2
    Vy = torch.exp(LAMBDA*h*2)*(torch.exp(h*MU**2)-1)*y0**2/ h**2
    return y0,h,input2,Ey,Vy

# Renvoie la position dans le Tube autour de H(y) = H(y0) 
def Tubes(theta,ep1,ep2):
    q = np.pi/3*np.cos(theta)+ep1
    p = np.sin(theta)+ep2
    return ([q,p])

# Fonction d'initialisation pour le pendule 
def init_Pend(nb_point,h_fixé,Tube,Rand):
    # 2 types d'initialisations pour y0
    if Tube:
        # Si on ressere le domaine, on genere nb_point dans le tube autour de la variété 
        y0 = []
        for i in range(0,nb_point):
            theta = torch.rand(1)*2*np.pi
            ep1 = torch.rand(1)*0.4-0.2
            ep2 = torch.rand(1)*0.4-0.2
            y0.append(Tubes(theta,ep1,ep2))
        y0 = torch.tensor(y0)
    else:
        # Si initialisation basique on genre nb_point dans un compact 
        y0 = torch.rand(nb_point, 1) *2*np.pi-np.pi
        y1 = torch.rand(nb_point, 1) *3-1.5
        y0 = torch.cat((y0,y1),dim=1)
    # Soit on choisit h dans un compact soit on fixe h 
    if h_fixé:
        #h = torch.ones(nb_point,1)*0.1
        log_h = torch.rand(nb_point, 1)*3-3
        h = torch.exp(log_h)
    else:
        h = torch.ones(int(nb_point/3),1)*0.05
        h = torch.cat((h,torch.ones(int(nb_point/3),1)*0.1,torch.ones(int(nb_point/3),1)*0.2)) 
    input2 = torch.cat((y0,h),1)
    # On calcule une approximation de la solution exact y1 en utilisant le schéma non modifié avec un pas plus petit
    x = []
    f = Field("Pendule")
    s = Schéma("MidPoint",f,f,Rand)
    for i in range(0,nb_point):
        print("  {} % \r".format(str(int(i)/10).rjust(3)), end="")
        x.append(s.Applied(y0[i],h[i]/10,h[i],NB_TRAJ_INIT,False))
    # On approxime l'espérance et la covariance par des estiamteurs standards (Monte-Carlo)
    Ex = []
    for i in range(0,nb_point):
        Ex.append(torch.mean(x[i],dim=0)/h[i][0]**2)
    Ex = torch.stack(Ex)
    Vx = []
    for i in range(0,nb_point):
        Vx.append(torch.cov(x[i].T)/ h[i][0]**2)
    Vx = torch.stack(Vx)

    return y0,h,input2,Ex,Vx

# Initilisation pour le bruit Additif 
def init_Additif(nb_point,hf):
    y0 = torch.randn(nb_point,1)/(1-hf/2)
    h = torch.ones(nb_point,1)*hf
    input2 = torch.cat((y0,h),1)
    Ex = torch.zeros(nb_point)
    Vx = torch.ones(nb_point)
    return y0,h,input2,Ex,Vx

# Fonction créant les données et les mettant sous la forme de générateurs
def create_dataset(nb_point,Type,Rand,dim):
    # On crée les ensembles de données en fonction de notre problème 
    if FUNC1 == "Linearf" and FUNC2 == "LinearSigma":
        y0,h,input2,Ey,Vy = init_Linear(nb_point,dim)
    elif FUNC1 == "Pendule":
        y0,h,input2,Ey,Vy = init_Pend(nb_point,True,True,Rand)
    elif FUNC1 == "Additif":
        y0,h,input2,Ey,Vy = init_Additif(nb_point,0.1)

    # On transforme nos ensembles de données en générateur avec une structure de données de pytorch 
    # Le générateur va envoyer les données par lot de taille BS 
    train_dataset = data.TensorDataset(y0)
    train_dataloader = data.DataLoader(
            train_dataset, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    train_dataset2 = data.TensorDataset(h)
    train_dataloader2 = data.DataLoader(
            train_dataset2, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    train_dataset3 = data.TensorDataset(input2)
    train_dataloader3 = data.DataLoader(
            train_dataset3, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))

    val_dataset = data.TensorDataset(Ey)
    val_dataloader = data.DataLoader(
            val_dataset, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    val_dataset2 = data.TensorDataset(Vy)
    val_dataloader2 = data.DataLoader(
            val_dataset2, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    return train_dataloader,train_dataloader2,train_dataloader3,val_dataloader,val_dataloader2
