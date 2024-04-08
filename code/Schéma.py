import torch
from Field import JNablaH
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")    
else:
    device = torch.device("cpu") 

def randO2(traj):
    probs = torch.tensor([1/6, 1/6, 2/3])
    distribution = torch.distributions.Categorical(probs)

    samples = distribution.sample((traj,))

    values = torch.zeros(traj)

    # Utilisation de masques pour définir les valeurs en une seule opération
    values[samples == 0] = -torch.sqrt(torch.tensor(3.))
    values[samples == 1] = torch.sqrt(torch.tensor(3.))
    values[samples == 2] = torch.tensor(0)

    return values

def init_rand(type,traj):
    if type == "Gaussienne":
        return torch.randn(traj) 
    elif type == "PF":
        return 2*torch.randint(0, 2, size=(traj,))-1
    elif type == "Ordre2":
        return randO2(traj)

def PFixe(q0,h,rand,models,traj,train):
    if(len(np.shape(q0)) == 1):
        q = (q0.unsqueeze(-1) * torch.ones(traj)).mT
        aux = (q0.unsqueeze(-1) * torch.ones(traj)).mT
    elif(train):
        q = (q0.unsqueeze(-1) * torch.ones(traj)).mT
        aux = (q0.unsqueeze(-1) * torch.ones(traj)).mT
    else:
        q = q0
        aux = q0
    h = (h.unsqueeze(-1) * torch.ones(traj)).mT
    r = rand.unsqueeze(-1)
    traj = np.shape(rand)
    save = aux+1
    N = 0
    while N<10 and torch.max(torch.abs(aux-save))>10**(-7):
        if(models != None):
            if(len(np.shape(q)) == 2):
                if (len(np.shape(h)) == 3):
                    h = h[0]
                input21 = ((q+aux)/2).to(device)
                input22 = torch.cat(((q+aux)/2,h),dim=1).to(device)
            else:
                input21 = ((q+aux)/2).to(device)
                input22 = torch.cat(((q+aux)/2,h),dim=2).to(device)
        save = aux
        if(models == None):
            aux = (q + (h+np.sqrt(h)*r)*(JNablaH((aux+q)/2)))
        else:
            mod = models[0](input21)
            mod2 = models[1](input22)
            aux = (q + (h + torch.sqrt(h) * r) * (JNablaH((aux + q) / 2) + h * mod.to("cpu") + h**2 * mod2.to("cpu")))
        N += 1
    return aux

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

def ESymp(y,h,traj,Rand):
    rand = init_rand(Rand,traj)
    p = y[1]*torch.ones(traj) - np.sin(y[0])*(h*torch.ones(traj)+np.sqrt(h)*rand)
    q = y[0]*torch.ones(traj) + p*(h*torch.ones(traj)+np.sqrt(h)*rand) 
    return ([q,p])

def PRK(y,h,traj,Rand):
    rand = init_rand(Rand,traj)
    p = (y[1]*torch.ones(traj) + -h*np.sin(y[0])*torch.ones(traj) - np.sqrt(h)*rand*np.sin(y[0]))/(1+np.cos(y[0])*h/2)
    q = y[0]*torch.ones(traj) + h*p + (+np.sin(y[0])*h*torch.ones(traj))/2 + p*np.sqrt(h)*rand
    return ([q,p])

def SSV(y,h,traj,Rand):
    traj = traj
    rand = init_rand(Rand,traj)
    p1 = y[1]*torch.ones(traj) - np.sin(y[0])/2*(h*torch.ones(traj)+np.sqrt(h)*rand)
    q = y[0]*torch.ones(traj) + p1*(h*torch.ones(traj)+np.sqrt(h)*rand)
    p = p1 - torch.sin(q)/2*(h*torch.ones(traj)+np.sqrt(h)*rand)
    return ([q,p])

def MidPoint(y,h,traj,Rand):
    traj = traj
    # rand = 2*torch.randint(0, 2, size=(traj,))-1
    rand = init_rand(Rand,traj)
    #rand = torch.random(traj)
    y1 = PFixe(y,h,rand,None,traj,False)
    p = y*torch.ones(traj,2) + (h+np.sqrt(h)*rand).unsqueeze(-1)*(JNablaH((y+y1)/2))
    return p

def EMAruyama(y0,h,traj,Rand):
    y = y0 
    l = [y0]
    rand = init_rand(Rand,traj)
    y =  y*torch.ones(traj,2) + (h+np.sqrt(h)*rand).unsqueeze(-1)*(JNablaH((y)))
    return y

def Meth_Non_Mod(y0,h,T,traj,type,Rand):
    y = y0
    l = [y0]
    for t in np.arange(h,T,h):
        if type == "MidPoint":
            y = MidPoint(y,h,traj,Rand)
        elif type == "EMaruyama":
            y = EMAruyama(y,h,traj,Rand)
        elif type == "SSV":
            y = SSV(y,h,traj,Rand)
        l.append([torch.mean(y[:,0]),torch.mean(y[:,1])])
    return y,l

def SSV_Mod(models,y0,h,T,traj,Rand):
    y = y0
    l = [y0]
    for t in np.arange(h,T,h):
        if t == h:
            input2 = torch.cat((torch.tensor([y]),torch.tensor([[h]])),dim=1)
            input21 = input2[0,[0, 2]]
            input22 = input2[0,1:3]
            rand = init_rand(Rand,traj)
            p = y[1]*torch.ones(traj) + h*(-np.sin(y[0])+h*models[0][1](input22)+h**2*models[1][1](input22))*torch.ones(traj)+np.sqrt(h)*(-np.sin(y[0])+h*models[2][1](input22)+h**2*models[3][1](input22))*rand
            q = y[0]*torch.ones(traj) + h*(p+h*models[0][0](input21)+h**2*models[1][0](input21))*torch.ones(traj)+np.sqrt(h)*(p+h*models[2][0](input21)+h**2*models[3][0](input21))*rand
            y = torch.stack((q,p)).mT
        else:
            input2 = torch.cat((y,(h*torch.ones(traj)).unsqueeze(1)),dim=1)
            input21 = input2[:, [0, 2]]
            input22 = input2[:, 1:3]
            rand = init_rand(Rand,traj)
            p = y[:,1].unsqueeze(-1) + h*(-torch.sin(y[:,0].unsqueeze(-1))+h*models[0][1](input22)+h**2*models[1][1](input22))+np.sqrt(h)*(-torch.sin(y[:,0].unsqueeze(-1))+h*models[2][1](input22)+h**2*models[3][1](input22))*rand.unsqueeze(-1)
            q = y[:,0].unsqueeze(-1)+ h*(p+h*models[0][0](input21)+h**2*models[1][0](input21))+np.sqrt(h)*(p+h*models[2][0](input21)+h**2*models[3][0](input21))*rand.unsqueeze(-1)
            y = torch.cat((q, p), dim=1)
        l.append(y)
    return y,l

def MidPoint_Mod(models,y0,h,T,traj,Rand):
    y = y0
    l = [y0]
    h_comp = h
    for t in np.arange(h,T,h):
        rand = init_rand(Rand,traj)
        y1 = PFixe(y,h,rand,models,traj,False)
        if t == h_comp:
            y = (y.unsqueeze(-1) * torch.ones(traj)).mT
            h = (h.unsqueeze(-1) * torch.ones(traj)).mT
            r = rand.unsqueeze(-1)
            input21 = ((y+y1)/2).to(device)
            input22 = torch.cat(((y+y1)/2,h),dim=1).to(device)
            y = y*torch.ones(traj,2) + (h+np.sqrt(h)*r) * (JNablaH((y + y1)/2)  + h*models[0](input21).to("cpu") + h**2*models[1](input22).to("cpu"))
        else:
            r = rand.unsqueeze(-1)
            input21 = ((y+y1)/2).to(device)
            input22 = torch.cat(((y+y1)/2,h),dim=1).to(device)
            y = y + (h+np.sqrt(h)*r) * (JNablaH((y + y1)/2) + h*models[0](input21).to("cpu") + h**2*models[1](input22).to("cpu"))
        l.append(y)
    return y,l

def Meth_Mod(models,y0,h,T,traj,type,Rand):
    y = y0
    l = [y0]
    for t in np.arange(h,T,h):
        if type == "MidPoint":
            y = MidPoint_Mod(models,y,h,T,traj,Rand)
        elif type == "SSV":
            y = SSV_Mod(models,y,h,T,traj,Rand)
        l.append([torch.mean(y[:,0]),torch.mean(y[:,1])])
    return y,l

