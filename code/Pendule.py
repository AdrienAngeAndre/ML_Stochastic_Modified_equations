import torch
import matplotlib.pyplot as plt
from model import create_models, train_models, loss_without_train
from Init import create_dataset,create_opt
from Plot import plot_Weakerr,plot_loss,plot_Phase,plot_ErrH
from Variables import LR,Systeme,EPOCHS

NB_POINT_TRAIN = 300
NB_POINT_TEST = 100 
FUNC1 = "Pendule"
FUNC2 = "Pendule"
RAND = "Gaussienne"
SCHEME = "MidPoint"
DIM = 2
TRUNC = 1
Y0 = torch.tensor([0,1])
T = 1 
LH = [0.1,0.2,0.3333333]


# Fonction qui test l'entrainement pour le pendule et trace les courbes attendus 
def main():
    Sys = Systeme(DIM,TRUNC,FUNC1,FUNC2,SCHEME,RAND)
    Sys.init_param_weak_err(Y0,T,LH)
    # On crée le modèle pour fapp 
    models1 = create_models(Sys.Dim,Sys.func1,Sys.trunc)
    
    y0_train, h_train, input2_train, Ey_train, Vy_train = create_dataset(NB_POINT_TRAIN,Sys,Sys.Dim)
    training_set = [y0_train,h_train,input2_train, Ey_train, Vy_train]

    y0_test, h_test, input2_test, Ey_test, Vy_test = create_dataset(NB_POINT_TEST,Sys,Sys.Dim)
    testing_set = [y0_test,h_test,input2_test, Ey_test, Vy_test]

    optimizer,all_parameters = create_opt(models1,[],LR)

    models1, models2, global_train_loss, global_test_loss, best_loss = train_models(models1, models1, training_set, testing_set, optimizer, Sys)

    epochs = torch.arange(0, EPOCHS)

    l = loss_without_train(training_set,Sys)
    loss_without = torch.ones(EPOCHS)*l

    plt.figure()

    plot_loss(epochs,global_train_loss,global_test_loss,loss_without)

    plt.title(f"Evolution de la perte pour le système Pendule")
    plt.savefig("Loss_Pendulum.png")

    plt.figure()

    plot_Weakerr(Sys,models1,models2,1000000)
    
    plt.title(f"Erreur faible pour le Pendule")
    plt.savefig("WeakErr_Pendulum.png")

    plt.figure()

    plot_Phase(Sys,models1,models2)

    plt.title(f"Evolution de la phase pour le pendule")
    plt.savefig("Phase_Pendulum.png")

    plt.figure()

    plot_ErrH(Sys,models1,models2)

    plt.title(f"Erreur sur l'Hamiltonien pour le pendule")
    plt.savefig("HamilErr_Pendulum.png")

if __name__ == '__main__':
    main()