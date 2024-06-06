import torch
import matplotlib.pyplot as plt
import numpy as np
from Variables import MU,LAMBDA,DEVICE

# Classe réprésentant un champs de vecteur 
class Field:
    # Fonction d'initialisation du champs de vecteur 
    def __init__(self,field):
        # Chaine de caractère indiquant le champs de vecteur 
        self.Field = field
        # Si le champs de vecteur correspond au terme de Drift d'une EDS 
        # on précise la valeur de LAMBDA 
        if field == "Linearf":
            self.lambd = LAMBDA
            self.mu = MU
        # Si le champs de vecteur correspond au terme de Diffusion d'une EDS 
        # on précise la valeur de MU 
        elif field == "LinearSigma":
            self.lambd = LAMBDA
            self.mu = MU 
        
    # Fonction renvoyant la valeur du champs de vecteur appliqué à y
    def evaluate(self,y,h):
        if self.Field == "Linearf":
            return self.lambd*y
        elif self.Field == "LinearSigma":
            return self.mu**2*y**2
        elif self.Field == "Pendule":
            return JNablaH(y)

# Classe représentant un champs de vecteur modifié par machine learning, cette classe est donc une 
# extension de la classe Field  
class ModifiedField(Field):
    # Fonction d'initialisation du champs modifié, on étends la classe correpondant 
    # à un champs de vecteur 
    def __init__(self, field, mod):
        super().__init__(field)
        self.mod = mod 
        self.trunc = len(mod)-1

    # Evalue la valeur du champs modifié en se servant de la valeur du champs de base 
    def evaluate(self,y,h):
        y = y.to(DEVICE)
        h = h.to(DEVICE)
        if len(np.shape(h)) == 1 and len(np.shape(y)) != 1:
            h = h.repeat(np.shape(y)[0],1)
        input2 = torch.cat((y,h),len(np.shape(y))-1).to(DEVICE)
        fx = super().evaluate(y,h).to(DEVICE)
        for i in range(self.trunc):
            fx += h**(i+1)*self.mod[i](y)
        fx += h**(self.trunc+1)*self.mod[self.trunc](input2)
        return fx
    
    # Plot les valeurs du terme i du champs perturbé par Machine learning sur y0 
    def print_modfield(self,i,y0):
        l1 = []
        l2 = []
        for y in y0:
            l1.append(y[0][0][0])
            l2.append(self.mod[i-1](y[0].to(DEVICE)).to("cpu").detach()[0][0])
        plt.scatter(l1,l2,label=self.label(i),color="blue")

    # Renvoie le label correspondant 
    def label(self,i):
        if self.Field == "Linearf":
            if i == 1:
                return r"$f_1$ approché par le réseau de neurones"
            elif i == 2:
                return r"$f_2$ approché par le réseau de neurones"
        elif self.Field == "Linearsigma":
            if i == 1:
                return r"$\sigma_1$ approché par le réseau de neurones"
            elif i == 2:
                return r"$\sigma_2$ approché par le réseau de neurones"
    
# classe représentant un champs de vecteur modifié avec le champs modifié 
# calculé analitiquement 
class AnalyticModField(Field):
    # Initialisation du champs modifié calculé analytiquement  
    def __init__(self,field,trunc):
        super().__init__(field)
        self.trunc = trunc
    
    # Evalue la valeur du champs modifié en se servant de la valeur calculé analytiquement 
    def evaluate(self,y,h):
        fx = super().evaluate(y,h)
        if self.trunc == 1:
            fx += h*self.o1(y)
        elif self.trunc == 2:
            fx += h**2*self.o2(y)
        return fx

    # Renvoie le premier terme de la pertrubation du champs modifié calculé 
    def o1(self,y):
        if self.Field == "Linearf":
            return self.lambd**2*y/2 
        elif self.Field == "LinearSigma":
            return y**2*(self.mu**4/2+2*self.lambd*self.mu**2)
        elif self.Field == "Pendule":
            return f1(y)
        
    # Renvoie le second terme de la perturbation du champs modifié calculé 
    def o2(self,y):
        if self.Field == "Linearf":
            return self.lambd**3*y/6
        elif self.Field == "LinearSigma":
            return y**2*(self.mu**6/6+self.lambd*self.mu**4+2*self.lambd**2*self.mu**2)
        elif self.Field == "Pendule":
            print(r"Expression analytique pour le pendule non calculé pour $\tilde{f}_2$, l'expression est donc renvoyé à l'ordre 1")
            return 0
        
    # Renvoie le label correspondant 
    def label(self,i):
        if self.Field == "Linearf":
            if i == 1:
                return r"$f_1=\frac{\lambda^2}{2}x$"
            elif i == 2:
                return r"$f_1=\frac{\lambda^3}{6}$x"
        elif self.Field == "LinearSigma":
            if i == 1:
                return r"$f_1=(\frac{\mu^4}{2}+2\lambda\mu^2)x^2$"
            elif i == 2:
                return r"$f_1=(\frac{\mu**6}{6}+\lambda\mu^4+2\lambda^2\mu^2)x^2$"
        
    # Plot les valeurs du terme i du champs perturbé sur y0 
    def print_ana(self,i,y0):
        l1 = []
        l2 = []
        for y in y0:
            l1.append(y[0][0][0])
            if i == 1:
                l2.append(self.o1(y[0])[0][0])
            elif i == 2:
                l2.append(self.o2(y[0])[0][0])
        plt.scatter(l1,l2,label=self.label(i),color="orange")

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
        result[..., 1] = -(-torch.sin(y[...,0])*y[...,1]**2+2*torch.cos(y[...,0])*torch.sin(y[...,0]))/8
        result[..., 0] = 2*y[...,1]*torch.cos(y[...,0])/8
    else:
        result = -torch.tensor([-2*y[1]*torch.cos(y[0])/8 ,-(-torch.sin(y[0])*y[...,1]**2+2*torch.cos(y[0])*torch.sin(y[0]))/8])
    return result
