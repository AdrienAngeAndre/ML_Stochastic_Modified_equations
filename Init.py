import torch
import numpy as np
import torch.utils.data as data
from Schéma import Meth_Non_Mod

NB_POINT = 100
BS = 1
if torch.cuda.is_available():
    device = torch.device("cuda")    
else:
    device = torch.device("cpu") 

def create_opt(models,func):
    all_parameters = []

    if func == "Other":
        for model in models:
            for m in model:
                all_parameters += list(m.parameters())
    else:
        for model in models:
            model.to(device)
            all_parameters += list(model.parameters())
    
    optimizer = torch.optim.Adam(all_parameters, lr=0.001)
    return optimizer,all_parameters

def param(type):
    lambd = 2 
    mu = 2 
    return lambd, mu 

def Elipse(theta,ep1,ep2):
    q = np.pi/3*np.cos(theta)+ep1
    p = np.sin(theta)+ep2
    return ([q,p])


def init_Linear(nb_point):
    lambd, mu = param("Linear")
    y0 = torch.rand(nb_point, 2)*2
    log_h = torch.rand(nb_point, 1)*3-3
    h = torch.exp(log_h)
    input2 = torch.cat((y0,h),1)
    Ey = torch.exp(lambd*h)*y0/h**2
    Vy = torch.exp(lambd*h*2)*(torch.exp(h*mu**2)-1)*y0**2/h**2
    return y0,h,input2,Ey,Vy

def init_Pend(nb_point,h_fixé,Tube,Rand):
    if Tube:
        y0 = []
        for i in range(0,nb_point):
            theta = torch.rand(1)*2*np.pi
            ep1 = torch.rand(1)*0.4-0.2
            ep2 = torch.rand(1)*0.4-0.2
            y0.append(Elipse(theta,ep1,ep2))
            y0 = torch.tensor(y0)
    else:
        y0 = torch.rand(nb_point, 1) *2*np.pi-np.pi
        y1 = torch.rand(nb_point, 1)  *3-1.5
        y0 = torch.cat((y0,y1),dim=1)

    if h_fixé:
        h = torch.ones(nb_point,1)*0.1
    else:
        h = torch.ones(nb_point/4)*0.05
        h = torch.cat((h,torch.ones(nb_point/4)*0.1,torch.ones(nb_point/4)*0.2,torch.ones(nb_point/4)*0.5)) 
    h = h.unsqueeze(-1)

    input2 = torch.cat((y0,h),1)

    x = []
    for i in range(0,nb_point):
        print("  {} % \r".format(str(int(i)/10).rjust(3)), end="")
        x.append(Meth_Non_Mod(y0[i],h[i]/10,h[i],10000,"MidPoint",Rand)[0])

    Ex = []
    for i in range(0,nb_point):
        Ex.append([torch.mean(x[i][:,0])/h[i][0]**2,torch.mean(x[i][:,1])/h[i][0]**2])
    Ex = torch.stack([torch.stack(t) for t in Ex])

    Vx = []
    for i in range(0,nb_point):
        Vx.append(torch.cov(x[i].T)/h[i][0]**2)
    Vx = torch.stack(Vx)

    return y0,h,input2,Ex,Vx


def create_dataset(type,Rand):

    if type == "Linear":
        y0,h,input2,Ey,Vy = init_Linear(NB_POINT)
    else:
        y0,h,input2,Ey,Vy = init_Pend(NB_POINT,True,False,Rand)

    train_dataset = data.TensorDataset(y0)
    train_dataloader = data.DataLoader(
            train_dataset, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    train_dataset2 = data.TensorDataset(h)
    train_dataloader2 = data.DataLoader(
            train_dataset2, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    train_dataset3 = data.TensorDataset(input2)
    train_dataloader3 = data.DataLoader(
            train_dataset3, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))

    val_dataset = data.TensorDataset(Ey)
    val_dataloader = data.DataLoader(
            val_dataset, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))
    
    val_dataset2 = data.TensorDataset(Vy)
    val_dataloader2 = data.DataLoader(
            val_dataset2, batch_size=BS, shuffle=False, generator=torch.Generator().manual_seed(42))

    return train_dataloader,train_dataloader2,train_dataloader3,val_dataloader,val_dataloader2
