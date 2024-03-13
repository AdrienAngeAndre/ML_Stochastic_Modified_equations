import torch
import argparse
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
    
def create_models(y_dim,q):
    models = []
    models.append(MLP(y_dim+1, 1000, y_dim))
    models.append(MLP(y_dim+1, 1000, y_dim*q))
    return models

########### Fonction d'entrainement du modèle ######## 

# y0 est l'entrée initale , h le pas de temps , y la valeur exacte du flow au temps h avec en entrée y0
def train_batch(y0, h, input2, Ey, Vy, models, scheme, optimizer, criterion,trunc, function):
    optimizer.zero_grad()
    loss = compute_MSE(y0,h,input2, Ey, Vy, models, scheme, criterion,trunc, function)
    loss.backward()
    optimizer.step()
    return loss.item()


def compute_MSE(y0, h, input2, Ey, Vy, models,scheme, criterion, trunc, function):
    # on calcul d'abords f1 et Ra grâce à une première passe sur les 2 MLP, f1 prends en entré y0 et Ra y0 et h
    # On calcul ensuite la valeur en appliquant la méthode d'EUler à la troncature fapp
    # On en déduit la MSE
    x = y0*torch.ones(1000000) + h*(y0+h*models[0](input2))*torch.ones(1000000)+torch.sqrt(h)*(y0+h*models[1](input2))*torch.randn(1000000)
    y_hat1 = torch.mean(x)
    y_hat2 = torch.var(x)
    loss = criterion(y_hat1, Ey[0][0]) + criterion(y_hat2, Vy[0][0])
    return loss


def train_models(models, scheme, training_set, optimizer, criterion,trunc, function, nb_epochs):
    epochs = torch.arange(0, 30)
    # Pour chaque période, on parcoure les entrées et on update les poids par descente de gradient
    global_loss = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        epoch_train_losses = []
        for y0_batch, h_batch, input2_batch, Ey_batch, Vy_batch in zip(training_set[0],training_set[1],training_set[2],training_set[3],training_set[4]):
            y0_batch = y0_batch[0]
            h_batch = h_batch[0]
            input2_batch = input2_batch[0]
            Ey_batch = Ey_batch[0]
            Vy_batch = Vy_batch[0]
            # Appel à la fonction qui calcul la perte 
            loss = train_batch(
                y0_batch, h_batch, input2_batch, Ey_batch,Vy_batch, models,scheme, optimizer, criterion,trunc, function)
            epoch_train_losses.append(loss)
        # On renvoie la moyenne des erreurs des échantillons d'entrainement
        epoch_train_loss = torch.tensor(epoch_train_losses).mean().item()
        global_loss.append(epoch_train_loss)
        print(epoch_train_loss)
    return models,global_loss
    


########### Fonctions pour créer les datasets d'exemples ######## 

def create_dataset_NL(lambd,mu):
    y0 = torch.ones(100, 1)
    train_dataset = data.TensorDataset(y0)
    train_dataloader = data.DataLoader(
            train_dataset, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    # Choisir log h dans un intervalle
    log_h = torch.rand(100, 1)*3-3
    h = torch.exp(log_h)
    train_dataset2 = data.TensorDataset(h)
    train_dataloader2 = data.DataLoader(
            train_dataset2, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    input2 = torch.cat((y0,h),1)
    train_dataset3 = data.TensorDataset(input2)
    train_dataloader3 = data.DataLoader(
            train_dataset3, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    Ey = torch.exp(lambd*h)*y0
    val_dataset = data.TensorDataset(Ey)
    val_dataloader = data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    Vy = torch.exp(lambd*h*2)*(torch.exp(h*mu**2)-1)*y0**2
    val_dataset2 = data.TensorDataset(Vy)
    val_dataloader2 = data.DataLoader(
            val_dataset2, batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(42))

    return train_dataloader,train_dataloader2,train_dataloader3,val_dataloader,val_dataloader2

############# Fonction qui permette l'affichage #########    


#Permet d'afficher les différents f 
def plot_loss(epochs,global_loss):
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Evolution de la perte d'entrainement en fonction des périodes")
    plt.plot(epochs,global_loss)
    plt.savefig(f'training_loss.png')

def plot_f(y0_train,models,trunc):
    plot_fi(y0_train,models,1)
    plot_fi(y0_train,models,2)
    plot_fi(y0_train,models,3)
    plt.title(f'fi en fonction de y0 pour un f_app tronquer à f{trunc}')
    plt.xlabel("y0")
    plt.ylabel("fi(y0)")
    plt.legend()

def plot_fi(y0_train,models,i):
    l1 = []
    l2 = []
    for y0_batch in y0_train: 
        y0_batch = torch.tensor(y0_batch)
        l1.append(y0_batch)
        m = models[i-1](y0_batch).detach()
        l2.append(m)
    plt.scatter(l1,l2,label=f'y =f{i}(y0)')

def plot_traj2D(f,y0,h,T):
    l = RK4(f,y0,h,T)
    l1 = []
    l2 = []
    for i in l:
        l1.append(i[0])
        l2.append(i[1])
    plt.plot(l1,l2)
    plt.show()

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

def RBody(t,y):
    I1 = 1
    I2 = 2
    I3 = 3
    if(torch.is_tensor(y)):
        y = y.detach()
    return np.array([(1/I3-1/I2)*y[1]*y[2],(1/I1-1/I3)*y[0]*y[2],(1/I2-1/I1)*y[0]*y[1]])

def RBody2(t,y):
    I1 = 1
    I2 = 2
    I3 = 3
    return torch.tensor([[(1/I3-1/I2)*y[0][1]*y[0][2],(1/I1-1/I3)*y[0][0]*y[0][2],(1/I2-1/I1)*y[0][0]*y[0][1]]])



############# Main #############

def main():
    y_dim = 1
    trunc = 1
    q = 1
    lambd = 2
    mu = 0.1
    scheme = ""
    function = ""
    # On crée les 2 modèles pour l'instant un MLP pour f1 et un MLP pour Ra 
    models = create_models(y_dim, q)
    # La fonction de perte est une MSE 
    criterion = nn.L1Loss()
    # Pour l'instant on chosis un opt standard SGD et un learning rate moyen
    # Sujet à approfondir, influence du learning rate er de l'optimizer 
    all_parameters = []
    for model in models:
        all_parameters += list(model.parameters())
    optimizer = torch.optim.SGD(all_parameters, lr=0.0001)
    
    # On créer un dataset d'entrainement 80% des données (nombre de données à augmenter plus tard)
    y0_train, h_train, input2_train, Ey_train, Vy_train = create_dataset_NL(lambd,mu)

    training_set = [y0_train,h_train,input2_train, Ey_train, Vy_train]

    models, global_loss = train_models(models,scheme,training_set,optimizer,criterion,trunc,function,30)

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
    y = torch.tensor([1])
    m = [y]
    n = [0]
    time = [0]
    N = 2**8
    h = 1/N
    dW = np.sqrt(h) * np.random.randn(N)
    W = np.cumsum(dW)
    TF = 1
    t = np.arange(0, TF, h)
    R = 1
    Dt = R * h
    L = int(N / R)
    Xtrue = y * np.exp((lambd - 0.5 * mu**2) * np.arange(0, TF, h) + mu * W)
    Xem = np.zeros(L)
    Xem2 = np.zeros(L)
    # Applique Euler Maruyama à notre fonction modifiée 
    Xtemp = y
    for j in range(L):
        i = torch.cat((Xtemp,torch.tensor([h])))
        Winc = np.sum(dW[R*(j-1):R*j])
        Xtemp = Xtemp + Dt*(Xtemp+Dt*models[0](i))*lambd + mu*(Xtemp+Dt*models[1](i))*Winc
        Xem[j] = Xtemp
    t = np.arange(0, TF, h)
    # Applique Euler Maruyama à notre fonction de base 
    Xtemp2 = y
    for j in range(L):
        i = torch.cat((Xtemp2,torch.tensor([h])))
        Winc = np.sum(dW[R*(j-1):R*j])
        Xtemp2 = Xtemp2 + Dt*(Xtemp2)*lambd + mu*(Xtemp2)*Winc
        Xem2[j] = Xtemp2


    plt.plot(t,Xtrue, label = "XTrue calulé par la formule connu dans ce cas")
    plt.plot(t,Xem2, label = "X simulé en appliquant Euler aux fonction de base ")
    plt.plot(t,Xem, label = "X simulé en appliquant Euler aux fonction modifiés ")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.show()


    M = 50000


    Xem = np.zeros(5)
    for p in range(1, 6):
        Dt = 2**(p-10)
        L = int(TF/Dt)
        Xtemp = y * np.ones(M)

        for j in range(L):
            Winc = np.sqrt(Dt) * np.random.randn(M)
            Xtemp = Xtemp + Dt*(Xtemp)*lambd + mu*(Xtemp)*Winc

        Xem[p-1] = torch.mean(Xtemp)

    Xerr = np.abs(Xem - np.exp(lambd*TF))
    Dtvals = 2.0**np.arange(1, 6) / 2**10

    A = np.column_stack((np.ones(5), np.log(Dtvals)))
    rhs = np.log(Xerr)
    sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    q = sol[1]
    resid = np.linalg.norm(A.dot(sol) - rhs)
    print(f"q = {q}")
    print(f"residual = {resid}")
    plt.loglog(Dtvals, Xerr, 'b*-', label="erreur pour la fonction de base")
    
    # Calcul l'esperance E(X_L) - E(X(T)) pour 
    Xem = np.zeros(5)
    for p in range(1, 6):
        Dt = 2**(p-10)
        L = int(TF/Dt)
        Xtemp = y * np.ones(M)

        for j in range(L):
            Winc = np.sqrt(Dt) * np.random.randn(M)
            Xtemp = Xtemp + Dt*(Xtemp+Dt*models[0](i).detach())*lambd + mu*(Xtemp+Dt*models[1](i).detach())*Winc

        Xem[p-1] = torch.mean(Xtemp)
    
    Xerr = np.abs(Xem - np.exp(lambd*TF))
    Dtvals = 2.0**np.arange(1, 6) / 2**10
    A = np.column_stack((np.ones(5), np.log(Dtvals)))
    rhs = np.log(Xerr)
    sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    q = sol[1]
    resid = np.linalg.norm(A.dot(sol) - rhs)
    print(f"q = {q}")
    print(f"residual = {resid}")
    plt.loglog(Dtvals, Xerr, 'g*-', label="erreur pour la fonction modifiée")

    plt.loglog(Dtvals, Dtvals, 'r--', label="fonction identité")   
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$| E(X(T)) - \text{Sample average of } X_L |$')
    plt.legend()
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
    plt.show()


if __name__ == '__main__':
    main()