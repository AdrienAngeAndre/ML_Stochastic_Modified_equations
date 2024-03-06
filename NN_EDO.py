import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

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

# y0 est l'entrée initale , h le pas de temps , y la valeur exacte du flow au temps h avec en entrée y0
def train_batch(y0, h, y, l, model1, model2, optimizer, criterion, **kwargs):
    optimizer.zero_grad()
    loss = compute_MSE(y0,h,y,l,model1,model2,criterion)
    loss.backward()
    optimizer.step()
    return loss.item()

def compute_MSE(y0,h,y,l, model1, model2, criterion):
    # on calcul d'abords f1 et Ra grâce à une première passe sur les 2 MLP, f1 prends en entré y0 et Ra y0 et h
    f1 = model1(y0)
    input2 = torch.cat((y0, h), 0)
    Ra = model2(input2)
    # On calcul ensuite la valeur en appliquant la méthode d'EUler à la troncature fapp
    y_hat = (y0 + h*y0*l + h**2*f1 + h**3*Ra)/h**2 
    # On en déduit la MSE 
    loss = criterion(y_hat, y)
    return loss

def create_datasets(n,l):
    # Création du dataset initial de y0, pour l'instant y0 1D chosis aléatoirement entre 0 et 1  
    y0 = torch.rand(n,1)
    train_dataset = data.TensorDataset(y0)
    train_dataloader = data.DataLoader(
            train_dataset, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))

    # Création du dataset initial de h, pour l'instant h chosis aléatoirement entre 0 et 1  
    h = torch.rand(n,1)
    train_dataset2 = data.TensorDataset(h)
    train_dataloader2 = data.DataLoader(
            train_dataset2, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))

    # Création des valeurs attendus pour y1 ici y(h) = exp(lambda*h)*y0  
    y = torch.exp(l*h)*y0/h**2
    val_dataset = data.TensorDataset(y)
    val_dataloader = data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))

    return train_dataloader,train_dataloader2,val_dataloader
    



def main():
    # valeur de lambda dans L'EDO y' = lambda * y 
    l = 3 
    y_dim = 1
    # On crée les 2 modèles pour l'instant un MLP pour f1 et un MLP pour Ra 
    model_1 = MLP(y_dim, 1000, y_dim)
    model_2 = MLP(y_dim+1, 1000, y_dim)
    # La fonction de perte est une MSE 
    criterion = nn.MSELoss()
    # Pour l'instant on chosis un opt standard SGD et un learning rate moyen
    # Sujet à approfondir, influence du learning rate er de l'optimizer 
    optimizer = torch.optim.SGD(list(model_1.parameters())+list(model_2.parameters()), lr=0.001)
    
    # On créer un dataset d'entrainement 80% des données (nombre de données à augmenter plus tard)
    y0_train, h_train, y_train = create_datasets(80,l)
    # Les 20% restant servent à tester sur des données autres 
    y0_test, h_test, y_test = create_datasets(4,l)

    # Pour l'instant 30 périodes, 
    epochs = torch.arange(0, 30)

    # Pour chaque période, on parcoure les entrées et on update les poids par descente de gradient
    global_loss = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        for y0_batch, h_batch, y_batch in zip(y0_train,h_train,y_train):
            # zip marche sur n'importe quel iterable mais ne renvoie pas des tenseurs donc
            # il faut les retransformés  
            y0_batch = torch.tensor(y0_batch)
            h_batch = torch.tensor(h_batch)
            y_batch = torch.tensor(y_batch)
            # Appel à la fonction qui calcul la perte 
            loss = train_batch(
                y0_batch, h_batch, y_batch, l, model_1,model_2, optimizer, criterion)
            epoch_train_losses.append(loss)
        # On renvoie la moyenne des erreurs des échantillons d'entrainement
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        global_loss.append(epoch_train_loss)
        print(epoch_train_loss)

    # afficher les 
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Evolution de la perte d'entrainement en fonction des périodes")
    plt.plot(epochs,global_loss)
    plt.savefig(f'training_loss.png')
    plt.show()

if __name__ == '__main__':
    main()