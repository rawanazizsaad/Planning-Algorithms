import random
import numpy as np
import matplotlib.pyplot as plt

ub=5
lb=-5
food_source = 50   
D = 2  
N=int(food_source/D)
max_opt_iter = 50
limit=1

trial=np.zeros((N,1))
prob=[]
Fx=np.zeros((N))
x1=np.zeros((N,D))
x2=np.zeros((N,D))
fx=np.array((N,1))
fit=np.zeros((N,1))
pos=np.ones((N,D))
Xp=np.zeros((N,D))
xbest=np.zeros((max_opt_iter,D))
fxbest=np.zeros((max_opt_iter,1))
Xpos=np.zeros((N,D))
xnew=np.zeros((N,D))

def obj_fun(X):
    x1=X[:,0]
    x2=X[:,1]
    
    fx=np.power(x1,2)-x1*x2+np.power(x2,2)+2*x1+4*x2+3
    return fx

def fitness_fun(X):
    x1=X[:,0]
    x2=X[:,1]
    
    fx=np.power(x1,2)-x1*x2+np.power(x2,2)+2*x1+4*x2+3
    
    for i in range(len(fx)):
        if fx[i]>=0:
            fit[i]=1/(1+fx[i])
        else:
            fit[i]=1+abs(fx[i])
    return fit


for i in range(N):
    for j in range(D):
        pos[i,j]= lb+random.random()*(ub-lb)


 
Fx=obj_fun(pos)

Fitness=fitness_fun(pos)


for opt_iter in range(max_opt_iter):
  
    
    for i in range(N):
        xnew[i,:]=pos[i,:]
        neigh_index = random.randint(0, D-1) 
        partner=int(np.ceil(np.random.random()*N))-1
        while partner == i:
            partner = int(np.ceil(np.random.random()*N))-1
        Xpos=pos[i,neigh_index]
        Xp=pos[partner,neigh_index]
        xnew[i,neigh_index]=Xpos + 2*(random.random()-0.5)*(Xpos-Xp)
        if xnew[i,neigh_index]>ub:
            xnew[i,neigh_index] = ub
        elif xnew[i,neigh_index] < lb:
            xnew[i,neigh_index] = lb
        objnew=obj_fun(xnew)
        if objnew[i]<Fx[i]:
            pos[i,:]=xnew[i]
            Fx[i]=objnew[i]
            trial[i]=0
        else:
            trial[i]=trial[i]+1
            
    summ=np.sum(Fitness)
    for i in range(N):
        prob.append(Fitness[i]/summ)
    for i in range(N):
        r=np.random.random()
        if r<prob[i]:
            xnew[i,:]=pos[i,:]
            neigh_index = random.randint(0, D-1)
            partner=int(np.ceil(np.random.random()*N))-1
            while partner == i:
                partner = int(np.ceil(np.random.random()*N))-1
            Xpos=pos[i,neigh_index]
            Xp=pos[partner,neigh_index]
            xnew[i,neigh_index]=Xpos + 2*(random.random()-0.5)*(Xpos-Xp)
            if xnew[i,neigh_index]>ub:
                xnew[i,neigh_index] = ub
            elif xnew[i,neigh_index] < lb:
                xnew[i,neigh_index] = lb
            objnew=obj_fun(xnew)
            if objnew[i]<Fx[i]:
                pos[i,:]=xnew[i]
                Fx[i]=objnew[i]
                trial[i]=0
            else:
                trial[i]=trial[i]+1
        
    H=[]
    for i in range(N):
        if trial[i]>limit:
            H.append(i)
    for i in range(len(H)):
        for j in range(D):
            pos[i,j]=lb+2*(random.random()-0.5)*(ub-lb)
   
    Fx=obj_fun(pos)
    fxval=min(Fx) 
    fxindex=np.argmin(Fx)
    Gbest=pos[fxindex,:]
   
    xbest[opt_iter,:]=Gbest
    fxbest[opt_iter]=fxval
    print("iteration",opt_iter,"  best cost is ",fxbest[opt_iter])
x = range(max_opt_iter)
plt.plot(x, fxbest)
plt.title('iteration & best object function')
plt.xlabel('iteration ')
plt.ylabel('best object function')
plt.show()