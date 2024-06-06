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
from Plot import plot_loss
from Variables import SYS,NB_POINT_TRAIN,NB_POINT_TEST,EPOCHS,print_parameters,LR
np.random.seed(42)


# Fonction qui test l'entrainement en utilisant les variables spécifiés dans le fichier de commande(Variables)  
# et trace les courbes attendus 
def main():
    print_parameters(SYS)
    models1 = create_models(SYS.Dim,SYS.func1,SYS.trunc)
    models2 = create_models(SYS.Dim,SYS.func2,SYS.trunc)


    y0_train, h_train, input2_train, Ey_train, Vy_train = create_dataset(NB_POINT_TRAIN,SYS,SYS.Dim)
    training_set = [y0_train,h_train,input2_train, Ey_train, Vy_train]

    y0_test, h_test, input2_test, Ey_test, Vy_test = create_dataset(NB_POINT_TEST,SYS,SYS.Dim)
    testing_set = [y0_test,h_test,input2_test, Ey_test, Vy_test]

    optimizer,all_parameters = create_opt(models1,models2,LR)

    models1, models2, global_train_loss, global_test_loss, best_loss = train_models(models1, models2, training_set, testing_set, optimizer, SYS)

    epochs = torch.arange(0, EPOCHS)

    l = loss_without_train(training_set,SYS)
    loss_without = torch.ones(EPOCHS)*l

    plt.figure()

    plot_loss(epochs,global_train_loss,global_test_loss,loss_without)

    plt.title(f"Evolution de la perte pour le système")
    plt.savefig("Loss.png")


if __name__ == '__main__':
    main()
