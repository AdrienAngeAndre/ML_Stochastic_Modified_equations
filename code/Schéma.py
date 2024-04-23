import torch
from Field import JNablaH
from Field import f1
import numpy as np
import matplotlib.pyplot as plt

MidPoint2 = True

if torch.cuda.is_available():
    device = torch.device("cuda")    
else:
    device = torch.device("cpu") 

def param(type):
    lambd = 2
    mu = 2
    return lambd, mu 

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

def PFixe(q0,h,rand,models,traj,train,field,Grad):
    lambd, mu = param(field)
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
            if field == "Linear":
                aux = (q + (h*lambd*(aux+q)/2)+np.sqrt(h)*r*(mu*(aux+q)/2))
            else:
                aux = (q + (h+np.sqrt(h)*r)*(JNablaH((aux+q)/2)))
        else:
            if field == "Linear":
                with torch.no_grad():
                    mod = models[0](input22)
                    mod2 = models[1](input22)
                    mod3 = models[2](input22)
                    mod4 = models[3](input22)
                    mod5 = models[4](input22)
                    mod6 = models[5](input22)
                    aux = (q + (h*(lambd*(aux+q)/2+h*mod+h**2*mod2+h**3*mod3)+np.sqrt(h)*r*(mu*(aux+q)/2+h*mod4+h**2*mod5+h**3*mod6)))
            else:
                if Grad:
                    mod = models[0](input21)
                    mod2 = models[1](input22)
                    aux = (q + (h + torch.sqrt(h) * r) * (JNablaH((aux + q) / 2) + h * mod.to("cpu") + h**2 * mod2.to("cpu")))
                    # aux = (q + (h + torch.sqrt(h) * r) * (JNablaH((aux + q) / 2) + h * f1((aux+q)/2)))
                else:
                    with torch.no_grad():
                        mod = models[0](input21)
                        mod2 = models[1](input22)
                        aux = (q + (h + torch.sqrt(h) * r) * (JNablaH((aux + q) / 2) + h * mod.to("cpu") + h**2 * mod2.to("cpu")))
                        #aux = (q + (h + torch.sqrt(h) * r) * (JNablaH((aux + q) / 2) + h * f1((aux+q)/2)))
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

def ESymp(y,h,traj,Rand,field):
    rand = init_rand(Rand,traj)
    p = y[1]*torch.ones(traj) - np.sin(y[0])*(h*torch.ones(traj)+np.sqrt(h)*rand)
    q = y[0]*torch.ones(traj) + p*(h*torch.ones(traj)+np.sqrt(h)*rand) 
    return ([q,p])

def PRK(y,h,traj,Rand,field):
    rand = init_rand(Rand,traj)
    p = (y[1]*torch.ones(traj) + -h*np.sin(y[0])*torch.ones(traj) - np.sqrt(h)*rand*np.sin(y[0]))/(1+np.cos(y[0])*h/2)
    q = y[0]*torch.ones(traj) + h*p + (+np.sin(y[0])*h*torch.ones(traj))/2 + p*np.sqrt(h)*rand
    return ([q,p])

def SSV(y,h,traj,Rand,field):
    traj = traj
    rand = init_rand(Rand,traj)
    p1 = y[1]*torch.ones(traj) - np.sin(y[0])/2*(h*torch.ones(traj)+np.sqrt(h)*rand)
    q = y[0]*torch.ones(traj) + p1*(h*torch.ones(traj)+np.sqrt(h)*rand)
    p = p1 - torch.sin(q)/2*(h*torch.ones(traj)+np.sqrt(h)*rand)
    return ([q,p])

def MidPoint(y,h,traj,Rand,field):
    traj = traj
    dim = np.shape(y)[-1]
    rand = init_rand(Rand,traj)
    y1 = PFixe(y,h,rand,None,traj,False,field,False)
    if field == "Linear":
        lambd, mu = param(field)
        p = y*torch.ones(traj,dim) + (h*torch.ones(traj).unsqueeze(-1))*(lambd*(y+y1)/2)+np.sqrt(h)*rand.unsqueeze(-1)*(mu*(y+y1)/2)
    else:
        p = y*torch.ones(traj,dim) + (h+np.sqrt(h)*rand).unsqueeze(-1)*(JNablaH((y+y1)/2))
    return p

def EMAruyama(y0,h,traj,Rand,field):
    y = y0 
    l = [y0]
    rand = init_rand(Rand,traj)
    if field == "Linear":
        lambd, mu = param(field)
        y = y*torch.ones(traj,np.shape(y)[-1]) + (h*torch.ones(traj).unsqueeze(-1))*(lambd*y)+np.sqrt(h)*rand.unsqueeze(-1)*(mu*y)
    else:
        y =  y*torch.ones(traj,np.shape(y)[-1]) + (h+np.sqrt(h)*rand).unsqueeze(-1)*(JNablaH((y)))
    return y

def Meth_Non_Mod(y0,h,T,traj,type,Rand,field):
    y = y0
    l = [y0]
    for t in np.arange(h,T+0.1*h,h):
        if type == "MidPoint":
            y = MidPoint(y,h,traj,Rand,field)
        elif type == "EMaruyama":
            y = EMAruyama(y,h,traj,Rand,field)
        elif type == "SSV":
            y = SSV(y,h,traj,Rand,field)
        if(np.shape(y)[-1] == 1):
            l.append([torch.mean(y[:,0])])
        else:
            l.append([torch.mean(y[:,0]),torch.mean(y[:,1])])
    return y,l

def Test(y0,h,T,traj,Rand,field):
    h = torch.tensor([h])
    y = y0
    y1 = y0
    l = [y0[0]]
    l2 = [y0[0]]
    l3 = [y0[1]]
    l4 = [y0[1]]
    for t in np.arange(h,T+0.1*h,h):
        y = MidPoint(y,h,traj,Rand,field)
        y1 = EMAruyama(y1,h,traj,Rand,field)
        l.append(torch.mean(y[:,0]))
        l2.append(torch.mean(y1[:,0]))
        l3.append(torch.mean(y[:,1]))
        l4.append(torch.mean(y1[:,1]))
    plt.plot(l,l3)
    plt.plot(l2,l4)
    plt.show()

def SSV_Mod(models,y0,h,T,traj,Rand,field):
    y = y0
    l = [y0]
    for t in np.arange(h,T+h,h):
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

def MidPoint_Mod(models,y0,h,T,traj,Rand,field,Grad):
    y = y0
    l = [y0]
    h_comp = h
    dim = np.shape(y)[-1]
    lambd, mu = param("Linear")
    for t in np.arange(h,T+h,h):
        rand = init_rand(Rand,traj)
        if field == "Linear":
            if t == h_comp:
                input2 = torch.cat((y,h)).to(device)
                f_mod = y * lambd + h * models[0](input2).to("cpu") + h**2 * models[1](input2).to("cpu") + h**3 * models[2](input2).to("cpu") 
                sigma_mod =  torch.sqrt(torch.abs(y**2 * mu**2 + h * models[3](input2).to("cpu") + h**2 * models[4](input2).to("cpu") + h**3 * models[5](input2).to("cpu")))
                y = y * torch.ones(traj,dim) + h * (y * lambd + h * models[0](input2).to("cpu") + h**2 * models[1](input2).to("cpu") + h**3 * models[2](input2).to("cpu") ) * torch.ones(traj,dim) + torch.sqrt(h) * rand.unsqueeze(-1) * torch.sqrt(torch.abs(y**2 * mu**2 + h * models[3](input2).to("cpu") + h**2 * models[4](input2).to("cpu") + h**3 * models[5](input2).to("cpu")))
            else:
                with torch.no_grad():
                    input2 = torch.cat((y,h*torch.ones(traj).unsqueeze(-1)),1).to(device)
                    f_mod = y * lambd + h * models[0](input2).to("cpu") + h**2 * models[1](input2).to("cpu") + h**3 * models[2](input2).to("cpu") 
                    sigma_mod =  torch.sqrt(torch.abs(y**2 * mu**2 + h * models[3](input2).to("cpu") + h**2 * models[4](input2).to("cpu") + h**3 * models[5](input2).to("cpu")))
                    y = y + h * ( y * lambd + h * models[0](input2).to("cpu") + h**2 * models[1](input2).to("cpu") + h**3 * models[2](input2).to("cpu") )  + torch.sqrt(h) * rand.unsqueeze(-1) * torch.sqrt(torch.abs(y**2 * mu**2 + h * models[3](input2).to("cpu") + h**2 * models[4](input2).to("cpu") + h**3 * models[5](input2).to("cpu")))
        else:
            if MidPoint2:
                y1 = PFixe(y,h_comp,rand,models,traj,False,field,Grad)
                if t == h_comp:
                    with torch.no_grad():
                        y = (y.unsqueeze(-1) * torch.ones(traj)).mT
                        h = (h.unsqueeze(-1) * torch.ones(traj)).mT
                        r = rand.unsqueeze(-1)
                        input21 = ((y+y1)/2).to(device)
                        input22 = torch.cat(((y+y1)/2,h),dim=1).to(device)
                        y = y*torch.ones(traj,2) + (h+np.sqrt(h)*r) * (JNablaH((y + y1)/2)  + h*models[0](input21).to("cpu") + h**2*models[1](input22).to("cpu"))
                        #y = y*torch.ones(traj,2) + (h+np.sqrt(h)*r) * (JNablaH((y + y1)/2)  + h*f1((y + y1)/2))
                else:
                    with torch.no_grad():
                        r = rand.unsqueeze(-1)
                        input21 = ((y+y1)/2).to(device)
                        input22 = torch.cat(((y+y1)/2,h),dim=1).to(device)
                        y = y + (h+np.sqrt(h)*r) * (JNablaH((y + y1)/2) + h*models[0](input21).to("cpu") + h**2*models[1](input22).to("cpu"))
                        #y = y + (h+np.sqrt(h)*r) * (JNablaH((y + y1)/2) + h*f1((y + y1)/2))
                l.append(y)
            else:
                if t == h_comp:
                    with torch.no_grad():
                        r = rand.unsqueeze(-1)
                        input21 = torch.tensor(y,dtype=torch.float32).to(device)
                        input22 = torch.cat((y,h),dim=0).to(device)
                        y = y + (h+np.sqrt(h)*r) * (JNablaH(y) + h*models[0](input21).to("cpu") + h**2*models[1](input22).to("cpu"))
                else:
                    with torch.no_grad():
                        r = rand.unsqueeze(-1)
                        input21 = y.to(device)
                        input22 = torch.cat((y,h*torch.ones(traj,1)),dim=1).to(device)
                        y = y + (h+np.sqrt(h)*r) * (JNablaH(y) + h*models[0](input21).to("cpu") + h**2*models[1](input22).to("cpu"))
                l.append(y)
    return y,l

def Meth_Mod(models,y0,h,T,traj,type,Rand,field,Grad):
    y = y0
    l = [y0]
    if type == "MidPoint" or type == "EMaruyama":
        y,l = MidPoint_Mod(models,y,h,T,traj,Rand,field,Grad)
    elif type == "SSV":
        y,l = SSV_Mod(models,y,h,T,traj,Rand,field)
    return y,l

