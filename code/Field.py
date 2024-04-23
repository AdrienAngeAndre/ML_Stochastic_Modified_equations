import torch
import numpy as np



# Fonction qui renvoie le champs de vecteur d'un pendule classique 
def NL_pendule(t,y):
    if(torch.is_tensor(y)):
        y = y.detach()
    return np.array([-np.sin(y[1]),y[0]])

# Même fonction mais qui renvoie dans un format différent 
def NL_pendule2(t,y):
    return torch.tensor([[-np.sin(y[0][1]),y[0][0]]])

# Fonction qui renvoie le champs de vecteur d'un Rigid Body classique 
def RBody(t,y):
    I1 = 1
    I2 = 2
    I3 = 3
    if(torch.is_tensor(y)):
        y = y.detach()
    return np.array([(1/I3-1/I2)*y[1]*y[2],(1/I1-1/I3)*y[0]*y[2],(1/I2-1/I1)*y[0]*y[1]])

# Même fonction mais qui renvoie dans un format différent 
def RBody2(t,y):
    I1 = 1
    I2 = 2
    I3 = 3
    return torch.tensor([[(1/I3-1/I2)*y[0][1]*y[0][2],(1/I1-1/I3)*y[0][0]*y[0][2],(1/I2-1/I1)*y[0][0]*y[0][1]]])

# Fonction qui renvoie le champs de vecteur d'un pendule stochastique associé à H = P**2/2 -Cos(Q)
def JNablaH(y):
    if len(y.shape) > 1:
        result = torch.empty_like(y)
        result[..., 0] = y[..., 1]
        result[..., 1] = -torch.sin(y[..., 0])
    else:
        result = torch.tensor([y[1], -torch.sin(y[0])])
    return result

# Fonction qui renvoie le champs de vecteur associé à H = (P**2+Q**2)/2
def JNablaH2(y):
    if len(y.shape) > 1:
        result = torch.empty_like(y)
        result[..., 0] = y[..., 1]
        result[..., 1] = -y[..., 0]
    else:
        result = torch.tensor([y[1], -y[0]])
    return result

def H(y0,Type):
    if Type == "Linear":
        return H_Linear(y0)
    else:
        return H_pendule(y0)

# Renvoie H(y0) avec H = P**2/2 -Cos(Q)
def H_pendule(y0):
    return y0[1]**2/2 - np.cos(y0[0]) 

# Renvoie H(y0) avec H =(P**2+Q**2)/2
def H_Linear(y0):   
    return (y0[0]**2+y0[1]**2)/2



# Fonction qui renvoie l'expression exact de F1 pour le pendule stochastique 
def f1(y):
    if len(y.shape) > 1:
        result = torch.empty_like(y)
        result[..., 1] = (-torch.sin(y[...,0])*y[...,1]**2+2*torch.cos(y[...,0])*torch.sin(y[...,0]))/24
        result[..., 0] = -2*y[...,1]*torch.cos(y[...,0])/24
    else:
        result = torch.tensor([-2*y[1]*torch.cos(y[0])/24 ,(-torch.sin(y[0])*y[...,1]**2+2*torch.cos(y[0])*torch.sin(y[0]))/24])
    return result