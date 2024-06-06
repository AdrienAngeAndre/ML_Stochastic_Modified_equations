import torch
import argparse
import numpy as np
import scipy
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
from model import create_models
from model import train_models,loss_without_train
from Init import create_dataset
from Init import create_opt
from Schéma import Schéma
from Field import H,ModifiedField,AnalyticModField,Field
from Plot import plot_fsigma,plot_fi
from Plot import plot_loss
from Variables import FUNC1,FUNC2,SCHEME,RAND,DIM,NB_POINT_TRAIN,NB_POINT_TEST,EPOCHS,LAMBDA,print_parameters,LR,TRUNC,Systeme
np.random.seed(42)


# Fonction qui test l'entrainement en utilisant les variables spécifiés dans le fichier de commande(Variables)  
# et trace les courbes attendus 
def main():
    Sys = Systeme(DIM,TRUNC,FUNC1,FUNC2,SCHEME,RAND)
    print_parameters(Sys)
    models1 = create_models(Sys.DIM,Sys.func1,Sys.TRUNC)
    models2 = create_models(Sys.DIM,Sys.func2,Sys.TRUNC)


    y0_train, h_train, input2_train, Ey_train, Vy_train = create_dataset(NB_POINT_TRAIN,Sys.func1,Sys.Rand,Sys.Dim)
    training_set = [y0_train,h_train,input2_train, Ey_train, Vy_train]

    y0_test, h_test, input2_test, Ey_test, Vy_test = create_dataset(NB_POINT_TEST,Sys.func2,Sys.Rand,Sys.Dim)
    testing_set = [y0_test,h_test,input2_test, Ey_test, Vy_test]

    optimizer,all_parameters = create_opt(models1,models2,LR)

    models1, models2, global_train_loss, global_test_loss, best_loss = train_models(models1, models2, training_set, testing_set, optimizer, Sys)

    epochs = torch.arange(0, EPOCHS)

    l = loss_without_train(training_set,Sys)
    loss_without = torch.ones(EPOCHS)*l

    plt.figure()

    plot_loss(epochs,global_train_loss,global_test_loss,loss_without)

    plt.title(f"Evolution de la perte pour le système")
    plt.savefig("Loss.png")


if __name__ == '__main__':
    main()
