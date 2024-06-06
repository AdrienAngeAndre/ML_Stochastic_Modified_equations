import torch

# Ce module est un module de commande 

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")    
else:
    DEVICE = torch.device("cpu")

# Classe qui permet de stocker les Variables inhérentes à un systeme 
class Systeme:
    # Initialisation de base d'un Système 
    def __init__(self,dim,trunc,func1,func2,Scheme,Rand):
        self.Dim = dim
        self.trunc = trunc 
        self.func1 = func1 
        self.func2 = func2
        self.Scheme = Scheme 
        self.Rand = Rand

    # Si le système est linéaire initialise les paramètres 
    def init_param_linear(self,lambd,mu):
        self.lambd = lambd 
        self.mu = mu 

    # Initialisation des paramètres pour plot l'erreur faible si besoin  
    def init_param_weak_err(self,y0,T,LH):
        self.y0 = y0 
        self.T = T 
        self.LH = LH

    def print_param(self):
        print("PARAMETRES :")
        if self.func1 == "Linearf":
            print(f"Terme de Drift : {self.func1} avec lambda = {self.lambd}")
            print(f"Terme de Diffusion : {self.func2} avec mu = {self.mu}")
        else:
            print(f"Terme de Drift : {self.func1}")
            print(f"Terme de Diffusion : {self.func2}")
        print(f"Schéma : {self.Scheme}")
        print(f"Dimension du système : {self.Dim}")
        print(f"Initialisation de xi : {self.Rand}")



DIM = 1
TRUNC = 2
FUNC1 = "Linearf"
FUNC2 = "Linears"
SCHEME = "EMaruyamaLinear"
RAND = "Gaussienne"
LAMBDA = 1
MU = 0.1
Y0 = torch.tensor([1])
T = 1
LH = [0.01,0.05,0.1,0.5]

SYS = Systeme(DIM,TRUNC,FUNC1,FUNC2,SCHEME,RAND)
SYS.init_param_linear(LAMBDA,MU)
SYS.init_param_weak_err(Y0,T,LH)

# Paramètres du réseaux de neurones et l'appentissage 

EPOCHS = 1
NB_POINT_TRAIN = 300
NB_POINT_TEST = 10
NB_TRAJ_TRAIN = 1000000
NB_TRAJ_INIT = 1000000
NB_HIDDEN = 2
HIDDEN_SIZE = 100
LR = 1e-4
DECAY = 1e-9
DROPOUT = 0.0
BS = 1

def print_param_res():
    print(f"Nombres de Point Train : {NB_POINT_TRAIN}")
    print(f"Nombres de Point Test : {NB_POINT_TEST}")
    print(f"Nombres de Trajectoire pour calculer la solution exact : {NB_TRAJ_INIT}")
    print(f"Nombres de Trajectoire pour l'entrainement : {NB_TRAJ_TRAIN}")
    print(f"Nombres de couches cachées : {NB_HIDDEN}")
    print(f"Nombres de neuronnes par couche cachée : {HIDDEN_SIZE}")
    print(f"Optimiser : AdamW")
    print(f"Learning rate : {LR}")
    print(f"Weight Decay : {DECAY}")
    print(f"Dropout : {DROPOUT}")
    print(f"Nombres d'epochs : {EPOCHS}")

def print_parameters(Systeme):
    print("PARAMETRES DU SYSTEME :")
    Systeme.print_param()
    print("PARAMETRES DU RESEAU DE NEURONES :")
    print_param_res()
    
