import torch
import matplotlib.pyplot as plt
from model import create_models, train_models,loss_without_train
from Init import create_dataset,create_opt
from Plot import plot_Weakerr,plot_loss,plot_fsigma
from Variables import LR,Systeme,print_parameters,EPOCHS

NB_POINT_TRAIN = 300
NB_POINT_TEST = 100 
FUNC1 = "Linearf"
FUNC2 = "LinearSigma"
RAND = "Gausienne"
SCHEME = "EMaruyamaLinear"
DIM = 1
TRUNC = 2
Y0 = torch.tensor([1])
T = 1 
LH = [0.01,0.05,0.1,0.2,0.5]


# Fonction qui test l'entrainement pour le système linéaire et trace les courbes attendus 
def main():
    Sys = Systeme(DIM,TRUNC,FUNC1,FUNC2,SCHEME,RAND)
    Sys.init_param_weak_err(Y0,T,LH)
    print_parameters(Sys)
    models1 = create_models(Sys.DIM,Sys.func1,Sys.TRUNC)
    models2 = create_models(Sys.DIM,Sys.func2,Sys.TRUNC)


    y0_train, h_train, input2_train, Ey_train, Vy_train = create_dataset(NB_POINT_TRAIN,Sys,Sys.Dim)
    training_set = [y0_train,h_train,input2_train, Ey_train, Vy_train]

    y0_test, h_test, input2_test, Ey_test, Vy_test = create_dataset(NB_POINT_TEST,Sys,Sys.Dim)
    testing_set = [y0_test,h_test,input2_test, Ey_test, Vy_test]

    optimizer,all_parameters = create_opt(models1,models2,LR)

    models1, models2, global_train_loss, global_test_loss, best_loss = train_models(models1, models2, training_set, testing_set, optimizer, Sys)

    epochs = torch.arange(0, EPOCHS)

    l = loss_without_train(training_set,Sys)
    loss_without = torch.ones(EPOCHS)*l

    plt.figure()

    plot_loss(epochs,global_train_loss,global_test_loss,loss_without)

    plt.title(f"Evolution de la perte pour le système Linéaire")
    plt.savefig("Loss_Linear.png")

    plt.figure()

    plot_Weakerr(Sys,models1,models2,1000000)
    
    plt.title(f"Erreur faible pour le Pendule")
    plt.savefig("WeakErr_Pendulum.png")

    plot_fsigma(y0_train,models1,models2,Sys)

if __name__ == '__main__':
    main()