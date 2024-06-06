
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import create_models, train_models,loss_without_train
from Init import create_dataset,create_opt
from Plot import plot_loss,plot_fi
from Variables import LR,Systeme,print_parameters,EPOCHS


np.random.seed(42)
NB_POINT_TRAIN = 300
NB_POINT_TEST = 100 
FUNC1 = "Additif"
FUNC2 = "Additif"
RAND = "Gausienne"
SCHEME = "Additif"
DIM = 1
TRUNC = 1

# Fonction qui test l'entrainement pour le bruit additif et trace les courbes attendus 
def test_addi():
    Sys = Systeme(DIM,TRUNC,FUNC1,FUNC2,SCHEME,RAND)
    print_parameters(Sys)
    models1 = create_models(Sys.Dim,Sys.func1,Sys.Trunc)

    y0_train, h_train, input2_train, Ey_train, Vy_train = create_dataset(NB_POINT_TRAIN,Sys,Sys.Dim)
    training_set = [y0_train,h_train,input2_train, Ey_train, Vy_train]

    y0_test, h_test, input2_test, Ey_test, Vy_test = create_dataset(NB_POINT_TEST,Sys,Sys.Dim)
    testing_set = [y0_test,h_test,input2_test, Ey_test, Vy_test]

    optimizer,all_parameters = create_opt(models1,models1,LR)

    models1, models2, global_train_loss, global_test_loss, best_loss = train_models(models1, models1, training_set, testing_set, optimizer, Sys)

    epochs = torch.arange(0, EPOCHS)

    l = loss_without_train(training_set,Sys)
    loss_without = torch.ones(EPOCHS)*l

    plot_loss(epochs,global_train_loss,global_test_loss,loss_without)
    torch.random.manual_seed(1)
    y0_train = torch.randn(100000)
    y = []
    for y0 in y0_train:
        y.append(torch.tensor(y0).unsqueeze(-1)) 
    plot_fi(y,models1,1,"lambda")

    plt.show()