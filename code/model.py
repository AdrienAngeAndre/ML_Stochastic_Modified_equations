import torch
import argparse
import numpy as np
import scipy
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
from Init import param
from Schéma import init_rand
from Schéma import PFixe
from Field import JNablaH

BS = 1
EPOCHS = 5
if torch.cuda.is_available():
    device = torch.device("cuda")    
else:
    device = torch.device("cpu") 

################################### Fonction relevant de la création du modèle ###################################


# MLP simple avec pour l'instant 2 hidden layer 
class MLP(nn.Module):
    # on choisis comme fonction d'activation Tanh pour l'instant 
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.hid = nn.Linear(hidden_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.acti = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)
        self.init_W(0,0)
    
    def init_W(self,weight,bias):
        if weight != None:
            nn.init.constant_(self.layer1.weight,weight)
            nn.init.constant_(self.layer1.bias,bias)
            nn.init.constant_(self.hid.weight,weight)
            nn.init.constant_(self.hid.bias,bias)
            nn.init.constant_(self.layer2.weight,weight)
            nn.init.constant_(self.layer2.bias,bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.acti(x)
        x = self.dropout(x)
        x = self.hid(x)
        x = self.acti(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


# Fonction qui permet de créer 
def create_models(y_dim,func):
    models = []
    if func == "Linear":
        models.append(MLP(y_dim+1, 1000, y_dim))
        models.append(MLP(y_dim+1, 1000, y_dim))
        models.append(MLP(y_dim+1, 1000, y_dim))
        models.append(MLP(y_dim+1, 1000, y_dim))
        models.append(MLP(y_dim+1, 1000, y_dim))
        models.append(MLP(y_dim+1, 1000, y_dim))
    elif func == "Other":
        models.append([MLP(1+1, 1000, 1),MLP(1+1, 1000, 1)])
        models.append([MLP(1+1, 1000, 1),MLP(1+1, 1000, 1)])
        models.append([MLP(1+1, 1000, 1),MLP(1+1, 1000, 1)])
        models.append([MLP(1+1, 1000, 1),MLP(1+1, 1000, 1)])
    else:
        models.append(MLP(y_dim, 100, y_dim))
        models.append(MLP(y_dim+1, 100, y_dim))
    return models


################################### Fonction relevant de l'entrainement du modèle ###################################


def X_Linear(train_batch,models,traj,Rand):
    y0_batch = train_batch[0]
    h_batch = train_batch[1]
    input2_batch = train_batch[2]
    lambd, mu = param("Linear")
    rand = init_rand(Rand)
    f_mod = y0_batch * lambd + h_batch * models[0](input2_batch) + h_batch**2 * models[1](input2_batch) + h_batch**3 * models[2](input2_batch) 
    sigma_mod =  torch.sqrt(torch.abs(y0_batch**2 * mu**2 + h_batch * models[3](input2_batch) + h_batch**2 * models[4](input2_batch) + h_batch**3 * models[5](input2_batch)))
    x = y0_batch * torch.ones(traj) + h_batch * f_mod * torch.ones(traj) + torch.sqrt(h_batch) * rand * sigma_mod
    return x 

def X_SSV(train_batch,models,traj,Rand):
    y0_batch = train_batch[0]
    h_batch = train_batch[1]
    input2_batch = train_batch[2]
    rand = init_rand(Rand)
    p1 = y0_batch[:,1].unsqueeze(-1) * torch.ones(traj)  + h_batch*(-torch.sin(y0_batch[:,0].unsqueeze(-1))/2 + h_batch * models[0][1](input2_batch) + h_batch**2 * models[1][1](input2_batch))*torch.ones(traj) + torch.sqrt(h_batch) * (-torch.sin(y0_batch[:,0].unsqueeze(-1))/2 + h_batch * models[2][1](input2_batch) + h_batch**2 * models[3][1](input2_batch)) * rand
    i = torch.cat((p1.unsqueeze(-1),input2_batch[:,],(h_batch*torch.ones(traj)).unsqueeze(-1)),dim=2)
    q = y0_batch[:,0].unsqueeze(-1) * torch.ones(traj) + h_batch*(p1 + h_batch* models[0][0](i).squeeze(2) + h_batch**2 * models[1][0](i).squeeze(2)) + torch.sqrt(h_batch) * (p1 + h_batch * models[2][0](i).squeeze(2) + h_batch**2 * models[3][0](i).squeeze(2)) * rand
    i = torch.cat((q.unsqueeze(-1),(h_batch*torch.ones(traj)).unsqueeze(-1)),dim=2)
    p = p1 + h_batch*(-torch.sin(q)/2 + h_batch * models[0][1](i).squeeze(2) + h_batch**2 * models[1][1](i).squeeze(2))*torch.ones(traj) + torch.sqrt(h_batch) * (-torch.sin(q)/2 + h_batch * models[2][1](i).squeeze(2) + h_batch**2 * models[3][1](i).squeeze(2)) * rand
    x = torch.stack((q, p))
    return x

def X_MidPoint(train_batch,models,traj,Rand):
    y0_batch = train_batch[0]
    h_batch = train_batch[1]
    # rand = (2*torch.randint(0, 2, size=(nb_trajectories,))-1)
    rand = init_rand(Rand)
    # rand = torch.randn(nb_trajectories)
    y1 = PFixe(y0_batch,h_batch,rand,models,traj,True)
    y = ((y0_batch.unsqueeze(-1) * torch.ones(traj)).mT)
    h = ((h_batch.unsqueeze(-1) * torch.ones(traj)).mT)
    r = rand.unsqueeze(-1)
    #input21 = torch.cat((y,y1,h),dim=2).to(device)
    input21 = ((y + y1)/2).to(device)
    input22 = torch.cat(((y+y1)/2,h),dim=2).to(device)
    x = y + (h+np.sqrt(h)*r) * (JNablaH((y + y1)/2) + h*models[0](input21).to("cpu") + h**2*models[1](input22).to("cpu"))
    return x


def compute_MSE(train_batch,models,function,Rand):
    # on calcul d'abords f1 et Ra grâce à une première passe sur les 2 MLP, f1 prends en entré y0 et Ra y0 et h
    # On calcul ensuite la valeur en appliquant la méthode d'EUler à la troncature fapp
    # On en déduit la MSE
    h_batch = train_batch[2]
    Ey = train_batch[3]
    Vy = train_batch[4]
    nb_trajectories = 100000
    if function == "Linear":
        x = X_Linear(train_batch,models,nb_trajectories,Rand)
    elif function == "Other":
        x = X_SSV(train_batch,models,nb_trajectories,Rand)
    else:
        x = X_MidPoint(train_batch,models,nb_trajectories,Rand)
    # Calcul des moyennes et des variances
    y_hat1 = torch.mean(x,dim=1) /h_batch**2
    y_hat2 = torch.cov(x[0].T) / h_batch**2
    loss = torch.mean(torch.mean(torch.abs(y_hat1 - Ey), dim=1)) + torch.mean(torch.sum(torch.abs(y_hat2 - Vy)))
    return loss

def train_models(models, training_set, optimizer, function, Rand):
    epochs = torch.arange(0, EPOCHS)
    # Pour chaque période, on parcoure les entrées et on update les poids par descente de gradient
    global_loss = []
    best_models = models
    best_loss = 1000
    for ii in epochs:
        ppp = 0
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        for y0_batch, h_batch, input2_batch, Ey_batch, Vy_batch in zip(training_set[0],training_set[1],training_set[2],training_set[3],training_set[4]):
            print("  {} % \r".format(str(int(1000 * (ppp + 1)*BS / 100) / 100).rjust(3)), end="")
            y0_batch = y0_batch[0]
            h_batch = h_batch[0]
            input2_batch = input2_batch[0]
            Ey_batch = Ey_batch[0]
            Vy_batch = Vy_batch[0]
            # Appel à la fonction qui calcul la perte 
            training_batch = [y0_batch,h_batch,input2_batch,Ey_batch,Vy_batch]
            loss = train_batch(training_batch, models,optimizer, function)
            epoch_train_losses.append(loss)
            ppp += 1
        # On renvoie la moyenne des erreurs des échantillons d'entrainement
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        if(epoch_train_loss<best_loss):
            best_loss = epoch_train_loss
            best_models = models
        global_loss.append(epoch_train_loss)
        print(epoch_train_loss)
    return best_models,global_loss

# y0 est l'entrée initale , h le pas de temps , y la valeur exacte du flow au temps h avec en entrée y0
def train_batch(training_batch, models, optimizer, function,Rand):
    optimizer.zero_grad()
    loss = compute_MSE(training_batch, models, function,Rand)
    loss.backward()
    optimizer.step()
    return loss.item()