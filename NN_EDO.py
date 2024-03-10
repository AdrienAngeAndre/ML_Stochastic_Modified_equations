import torch
import numpy as np
import scipy
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

########### Modèles Générales ###########

# MLP simple avec pour l'instant 2 hidden layer 
class MLP(nn.Module):
    # on choisis comme fonction d'activation Tanh pour l'instant 
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.hid = nn.Linear(hidden_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.acti = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.acti(x)
        x = self.hid(x)
        x = self.acti(x)
        x = self.layer2(x)
        return x
    
def create_models(y_dim,trunc):
    models = []
    for i in range(trunc):
        models.append(MLP(y_dim, 1000, y_dim))
    models.append(MLP(y_dim+1, 1000, y_dim))
    return models

########### Fonction d'entrainement du modèle ######## 

# y0 est l'entrée initale , h le pas de temps , y la valeur exacte du flow au temps h avec en entrée y0
def train_batch(y0, h, input2, y, l, models, optimizer, criterion,trunc, **kwargs):
    optimizer.zero_grad()
    loss = compute_MSE(y0,h,input2, y,l,models,criterion,trunc)
    loss.backward()
    optimizer.step()
    return loss.item()

def compute_MSE(y0,h, input2, y,l, models, criterion, trunc):
    # on calcul d'abords f1 et Ra grâce à une première passe sur les 2 MLP, f1 prends en entré y0 et Ra y0 et h
    Ra = models[trunc](input2)
    # On calcul ensuite la valeur en appliquant la méthode d'EUler à la troncature fapp
    f_app = NL_pendule2(0,y0)
    for j in range(trunc):
        f_app += h**(j+1)*models[j](y0)
    f_app += h**(trunc+1)*Ra
    y_hat = (y0 + h*f_app)/h**2
    # On en déduit la MSE 
    loss = criterion(y_hat, y)
    return loss


########### Fonctions pour créer les datasets d'exemples ######## 

def create_dataset_NL():
    y0 = torch.rand(100, 2)*4-2
    train_dataset = data.TensorDataset(y0)
    train_dataloader = data.DataLoader(
            train_dataset, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    # Choisir log h dans un intervalle
    h = torch.rand(100, 1)
    train_dataset2 = data.TensorDataset(h)
    train_dataloader2 = data.DataLoader(
            train_dataset2, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    input2 = torch.cat((y0,h),1)
    train_dataset3 = data.TensorDataset(input2)
    train_dataloader3 = data.DataLoader(
            train_dataset3, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    y = []
    for i in range(0,100):
        y.append((RK4(NL_pendule,y0[i],h[i]/10,h[i])[10])/h[i]**2)
    y = torch.stack(y)
    val_dataset = data.TensorDataset(y)
    val_dataloader = data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    return train_dataloader,train_dataloader2,train_dataloader3,val_dataloader

############# Fonction qui permette l'affichage #########    

def plot_fi(y0_train,models,i):
    l1 = []
    l2 = []
    for y0_batch in y0_train: 
        y0_batch = torch.tensor(y0_batch)
        l1.append(y0_batch)
        m = models[i-1](y0_batch).detach()
        l2.append(m)
    plt.scatter(l1,l2,label=f'y =f{i}(y0)')

#Permet d'afficher les différents f 
def plot_f(epochs,global_loss,y0_train,models,trunc,l):
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Evolution de la perte d'entrainement en fonction des périodes")
    plt.plot(epochs,global_loss)
    plt.savefig(f'training_loss.png')
    plot_fi(y0_train,models,1)
    plot_fi(y0_train,models,2)
    plot_fi(y0_train,models,3)
    plt.title(f'fi en fonction de y0 pour un f_app tronquer à f{trunc} et lambda={l}')
    plt.xlabel("y0")
    plt.ylabel("fi(y0)")
    plt.legend()
    plt.show()

def plot_traj(f,y0,h,T):
    l = RK4(f,y0,h,T)
    l1 = []
    l2 = []
    for i in l:
        l1.append(i[0])
        l2.append(i[1])
    plt.plot(l1,l2)
    plt.show()


########### Fonctions auxiliares (méthodes numériques ou champs de vecteurs) ###########

def RK4(f,y0,h,T):
    l = [y0]
    t = 0
    y = y0
    while(t<T):
        k1 = f(t,y)
        k2 = f(t+h/2,y+h/2*k1)
        k3 = f(t+h/2,y+h/2*k2)
        k4 = f(t+h,y+h*k3)
        y = y + h/6*(k1+2*k2+2*k3+k4)
        t += h 
        l.append(y)  
    return l

def NL_pendule(t,y):
    if(torch.is_tensor(y)):
        y = y.detach()
    return np.array([-np.sin(y[1]),y[0]])

def NL_pendule2(t,y):
    return torch.tensor([[-np.sin(y[0][1]),y[0][0]]])

############# Main #############

def main():
    # valeur de lambda dans L'EDO y' = lambda * y 
    l = -2
    y_dim = 2
    trunc = 3
    # On crée les 2 modèles pour l'instant un MLP pour f1 et un MLP pour Ra 
    models = create_models(y_dim, trunc)
    # La fonction de perte est une MSE 
    criterion = nn.MSELoss()
    # Pour l'instant on chosis un opt standard SGD et un learning rate moyen
    # Sujet à approfondir, influence du learning rate er de l'optimizer 
    all_parameters = []
    for model in models:
        all_parameters += list(model.parameters())
    optimizer = torch.optim.SGD(all_parameters, lr=0.001)
    
    # On créer un dataset d'entrainement 80% des données (nombre de données à augmenter plus tard)
    # y0_train, h_train, y_train = create_datasets(800,l)
    y0_train, h_train, input2_train, y_train = create_dataset_NL()
    # Pour l'instant 100 périodes, 
    epochs = torch.arange(0, 30)
    # Pour chaque période, on parcoure les entrées et on update les poids par descente de gradient
    global_loss = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        for y0_batch, h_batch, input2_batch, y_batch in zip(y0_train,h_train,input2_train,y_train):
            y0_batch = y0_batch[0]
            h_batch = h_batch[0]
            input2_batch = input2_batch[0]
            y_batch = y_batch[0]
            # Appel à la fonction qui calcul la perte 
            loss = train_batch(
                y0_batch, h_batch, input2_batch, y_batch, l, models, optimizer, criterion,trunc)
            epoch_train_losses.append(loss)
        # On renvoie la moyenne des erreurs des échantillons d'entrainement
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        global_loss.append(epoch_train_loss)
        print(epoch_train_loss)
    y0 = torch.tensor([-1.5,0])
    h = torch.tensor([0.01])
    t = 0
    T = 10
    l = [y0]
    y = y0
    while(t<T):
        i = torch.cat((y,h))
        Ra = models[trunc](i)
        # On calcul ensuite la valeur en appliquant la méthode d'EUler à la troncature fapp
        f_app = torch.tensor(NL_pendule(t,y))
        for j in range(trunc):
            f_app = f_app + (h**(j+1)*models[j](y)).detach()
        f_app = f_app + h**(trunc+1)*Ra
        y = (y + h*f_app)  
        t += h 
        l.append(y.detach())
    l1 = []
    l2 = []
    for i in l:
        l1.append(i[0])
        l2.append(i[1])
    plt.scatter(l1,l2)
    plot_traj(NL_pendule,y0,h,T)
    plt.show()


if __name__ == '__main__':
    main()
