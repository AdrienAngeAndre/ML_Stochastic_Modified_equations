import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from Schéma import init_rand
from Schéma import Schéma
from Field import ModifiedField,Field
from Variables import BS,EPOCHS,NB_HIDDEN,HIDDEN_SIZE,NB_TRAJ_TRAIN,NB_POINT_TRAIN,DEVICE

################################### Fonction relevant de la création du modèle ###################################


# Classe répretant un MLP 
class MLP(nn.Module):
    # Classe initialisant les différentes couches d'un MLP 
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, HIDDEN_SIZE)
        self.hid = nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, output_dim)
        self.acti = nn.ReLU()
        self.dropout = nn.Dropout(p=0)
    
    # Fonction qui permet d'initialiser à la main les poids et les biais 
    def init_W(self,weight,bias):
        if weight != None:
            nn.init.constant_(self.layer1.weight,weight)
            nn.init.constant_(self.layer1.bias,bias)
            nn.init.constant_(self.hid.weight,weight)
            nn.init.constant_(self.hid.bias,bias)
            nn.init.constant_(self.layer2.weight,weight)
            nn.init.constant_(self.layer2.bias,bias)

    # Fonction qui fait la forward 
    def forward(self, x):
        x = self.layer1(x)
        x = self.acti(x)
        x = self.dropout(x)
        for _ in range(NB_HIDDEN):
            x = self.hid(x)
            x = self.acti(x)
            x = self.dropout(x)
        x = self.layer2(x)
        return x

# Fonction qui permet de créer les différents MLP pour un champs de vecteur 
# et les stockes dans une liste
def create_models(y_dim,func,trunc):
    models = []
    if func == "Additif":
        # Dans le cadre du bruit additif simple on a besoin d'un seul MLP 
        models.append(MLP(y_dim+1, y_dim))
    else:
        # On a besoin de N MLPs en fonction de l'ordre auquel on tronque la série 
        for i in range(trunc):
            models.append(MLP(y_dim, y_dim))
        # On besoin d'un MLP en plus pour le reste, la dimension d'entrée prends en compte y et h 
        models.append(MLP(y_dim+1, y_dim))
    return models


################################### Fonction relevant de l'entrainement du modèle ###################################


# Fonction qui calcul N trajectoires d'un pas du schéma modifié avec l'état actuelle 
# des MLPs 
def X(y0_batch,h_batch,Rand,Schéma):
    # On crée le xi 
    rand = init_rand(Rand,NB_TRAJ_TRAIN)
    # On applique le schéma numérique modifié 
    x = Schéma.step(y0_batch,h_batch,rand)
    return x 

# Fonction qui calcul N trajectoires pour le bruit additif
def X_addi(y0_batch,h_batch,Schéma,Rand,seed):
    models = Schéma.f.mod
    torch.random.manual_seed(seed)
    rand = init_rand(Rand,NB_TRAJ_TRAIN)
    r = rand.unsqueeze(-1)
    input2_batch = torch.cat((y0_batch,h_batch.repeat(NB_TRAJ_TRAIN,1)),1)
    input22 = input2_batch.to(DEVICE)
    x = y0_batch-models[0](input22).to("cpu")*h_batch*y0_batch + torch.sqrt(2*h_batch)*r
    return x

# Calcule l'erreur entre les valeurs prédites et calculés avec notre 
# fonction de perte  
def compute_loss(train_batch,Schéma,Sys,seed):
    # Tous les lots 
    y0_batch = train_batch[0]
    h_batch = train_batch[1]
    Ey_batch = train_batch[3]
    Vy_batch = train_batch[4]
    loss = 0
    # On gère le ici le mini batching cad traiter un lot de plusieurs données avant la backpropagation plutot qu'une seule 
    # Normalement on vectorise sur pytorch mais ici rien que la génération de trajectoires sur une seule donnée est couteux 
    # en mémoire ainsi on ne peux pas vectoriser et traiter d'un seule cout la génération de trajectoires sur un lot entier de données 
    # Donc on traite les données du lot 1 par 1 à la différence que l'on fait la backpropagation seulement après avoir traiter le lot entier
    for i in range(BS):
        # Si on traite par lot les tenseurs ont une dimension suplémentaire 
        if BS == 1:
            y0 = y0_batch[0]
            h = h_batch[0]
            Ey = Ey_batch[0]
            Vy = Vy_batch[0]
        else:
            y0 = y0_batch[i]
            h = y0_batch[i]
            Ey = Ey_batch[i]
            Vy = Vy_batch[i]
        y0 = y0.to(DEVICE)
        h = h.to(DEVICE)
        # On calcul un N trajectoires d'un pas du schéma numérique  
        if Sys.Scheme == "Additif":
            x = X_addi(y0,h,Schéma,Sys.Rand,seed)
            y_hat1 = torch.mean(x).unsqueeze(-1).unsqueeze(-1)
        else:
            x = X(y0,h,Sys.Rand,Schéma)
            y_hat1 = torch.mean(x.to("cpu"),dim=0)/ h_batch[0]**2
        # On va distinguer les cas de la dimension 1 ou plus à cause des fonctions 
        # internes à pytorch qui sont différentes 
        if np.shape(Vy)[-1] == 1:
            if Sys.Scheme == "Additif":
                y_hat2 = torch.var(x)
            else:
                y_hat2 = torch.var(x.to("cpu")) / h_batch[0]**2
        else:
            y_hat2 = torch.cov(x.T.to("cpu")) / h_batch**2
        Ey = Ey.to("cpu")
        Vy = Vy.to("cpu")
        # Fonction de perte, la normalisation est directement dans les y_hat et Ey, Vy 
        loss += torch.mean(torch.abs(y_hat1 - Ey)) + torch.mean(torch.abs(y_hat2 - Vy))
    return loss/BS

# Fonction d'entrainement globale du réseaux de neurones 
def train_models(models1, models2, training_set, testing_set, optimizer, Sys):
    epochs = torch.arange(0, EPOCHS)
    global_train_loss = []
    global_test_loss = []
    best_models1 = models1
    best_models2 = models2
    best_loss = 1000
    # On parcoure les epochs 
    for ii in epochs:
        ppp = 0
        seed = 1
        # On crée un schéma modifié avec les modèles courant 
        f = ModifiedField(Sys.func1,models1)
        sigma = ModifiedField(Sys.func2,models2)
        Sché = Schéma(Sys.Scheme,f,sigma,Sys.Rand)
        print('Training epoch {}'.format(ii))
        # On Calcule la perte test sur des données à part 
        with torch.no_grad():
            epoch_test_losses = []
            for y0_batch, h_batch, input2_batch, Ey_batch, Vy_batch in zip(testing_set[0],testing_set[1],testing_set[2],testing_set[3],testing_set[4]):
                y0_batch = y0_batch[0]
                h_batch = h_batch[0]
                input2_batch = input2_batch[0]
                Ey_batch = Ey_batch[0]
                Vy_batch = Vy_batch[0]
                # Appel à la fonction qui calcul la perte 
                testing_batch = [y0_batch,h_batch,input2_batch,Ey_batch,Vy_batch]
                loss_test = compute_loss(testing_batch,Sché,Sys,seed)
                epoch_test_losses.append(loss_test)
            epoch_test_loss = torch.tensor(epoch_test_losses).mean().item()
        # On entraine notre modèle sur toutes les données restantes 
        epoch_train_losses = []
        for y0_batch, h_batch, input2_batch, Ey_batch, Vy_batch in zip(training_set[0],training_set[1],training_set[2],training_set[3],training_set[4]):
            print("  {} % \r".format(str(int((ppp + 1)*BS)*100 / NB_POINT_TRAIN).rjust(3)), end="")
            y0_batch = y0_batch[0]
            h_batch = h_batch[0]
            input2_batch = input2_batch[0]
            Ey_batch = Ey_batch[0]
            Vy_batch = Vy_batch[0]
            # Appel à la fonction qui calcul la perte 
            training_batch = [y0_batch,h_batch,input2_batch,Ey_batch,Vy_batch]
            loss_train = train_batch(training_batch,Sché,optimizer,Sys,seed)
            epoch_train_losses.append(loss_train)
            ppp += 1
            seed += 1
        # On renvoie la moyenne des erreurs des échantillons d'entrainement
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        # On test si la backpropagation a bien baissé la perte 
        if(epoch_train_loss<best_loss):
            best_loss = epoch_train_loss
            best_models1 = f.mod
            best_models2 = sigma.mod
        global_train_loss.append(epoch_train_loss)
        global_test_loss.append(epoch_test_loss)
        print(epoch_train_loss)
        print(epoch_test_loss)
    return best_models1,best_models2,global_train_loss,global_test_loss,best_loss

# Entrainement d'un lot 
def train_batch(training_batch, Schéma, optimizer,Sys,seed):
    # On calcul la perte puis on fait la backpropagation  
    optimizer.zero_grad()
    loss = compute_loss(training_batch, Schéma,Sys,seed)
    loss.backward()
    optimizer.step()
    return loss.item()

# Calcul la perte de la fonction non modifié 
def loss_without_train(training_set,Sys):
    epoch_losses = []
    seed = 1 
    f = Field(Sys.func1)
    sigma = Field(Sys.func2)
    Sché = Schéma(Sys.Scheme,f,sigma,Sys.Rand)
    for y0_batch, h_batch, input2_batch, Ey_batch, Vy_batch in zip(training_set[0],training_set[1],training_set[2],training_set[3],training_set[4]):
        y0_batch = y0_batch[0]
        h_batch = h_batch[0]
        input2_batch = input2_batch[0]
        Ey_batch = Ey_batch[0]
        Vy_batch = Vy_batch[0]
        training_batch = [y0_batch,h_batch,input2_batch,Ey_batch,Vy_batch]
        loss = compute_loss(training_batch,Sché,Sys,seed)
        epoch_losses.append(loss)
        seed += 1
    return torch.tensor(epoch_losses).mean().item()
