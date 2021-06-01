#%%
import numpy as np
import matplotlib.pyplot as plt
import gym.envs.classic_control.pendulum as pendulum
import torch
from utils import *

#%%
alpha          = 1
beta           = 5.5
num_sims       = 500
num_time_steps = 10
x_vec,xdot_vec,deltax,deltaxd,u_vec = generate_data(num_sims,
                                                    num_time_steps,
                                                    control_method="generate data",
                                                    noise_level = 0,
                                                    return_training_data=True,
                                                    alpha=alpha,
                                                    beta=beta)
#%% now let's try learning the optimal control, not just the nonlinearity
xx          = torch.Tensor(np.transpose(np.array([x_vec,xdot_vec])))
yy          = torch.Tensor(np.transpose(np.array([deltax,deltaxd])))
l = 1
g = 10
m = 1
# NN parameters
D_in  = 2 # x1, x2
H     = 100
D_out = 2 # delta x, delta x dot


model = torch.nn.Sequential(
torch.nn.Linear(D_in, H),
torch.nn.ReLU(),
torch.nn.Linear(H, H),
torch.nn.ReLU(),
torch.nn.Linear(H, D_out)
)

# pick data
train_count = int(0.90 * len(x_vec)) 
train_idx   = np.random.choice(len(x_vec),size=train_count,replace=False)
mask        = np.zeros(len(x_vec),dtype=bool)
mask[train_idx] = True
x_train     = xx[mask,:]
x_test      = xx[~mask,:]
y_train     = yy[mask,:]
y_test      = yy[~mask,:]

# train network
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
epochs = 1000
loss_train = np.zeros(epochs)
loss_test  = np.zeros(epochs)
for t in range(epochs):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x_train)

    # Compute and save loss.
    loss = loss_fn(y_pred, y_train)
    loss_train[t] = loss.item()

    # compute testing loss
    test_loss = loss_fn(model(x_test),y_test)
    loss_test[t] = test_loss.item()

    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()


plt.figure()
plt.semilogy(loss_train,label="Train")
plt.semilogy(loss_test,label='Test')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE Loss")
plt.title("Trained on "+str(train_count)+" samples")
plt.show()

plt.figure()
plt.plot(x_train[:,0].detach().numpy(),model(x_train)[:,0].detach().numpy(),'.')
plt.plot(x_train[:,0].detach().numpy(),y_train[:,0].detach().numpy(),'.')
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\Delta \dot{\theta}$")
plt.title("Model results with "+str(train_count)+" samples")

#%% now train with quadratic cost function

xx   = torch.Tensor(np.transpose(np.array([x_vec,xdot_vec])))
dt = 0.1
D_out = 5
model2 = torch.nn.Sequential(
torch.nn.Linear(D_in, H),
torch.nn.ReLU(),
torch.nn.Linear(H, H),
torch.nn.ReLU(),
torch.nn.Linear(H, H),
torch.nn.ReLU(),
torch.nn.Linear(H, D_out)
)

# pick data
train_count = int(0.90 * len(x_vec)) 
train_idx   = np.random.choice(len(x_vec),size=train_count,replace=False)
mask        = np.zeros(len(x_vec),dtype=bool)
mask[train_idx] = True
x_train     = xx[mask,:]
x_test      = xx[~mask,:]
y_train     = torch.zeros(train_count)
y_test      = torch.zeros(len(x_vec)-train_count)

x1 = x_train[:,0]
x2 = x_train[:,1]
x1_test = x_test[:,0]
x2_test = x_test[:,1]
print("x1,x2",x1.shape,x2.shape)

# train network
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.RMSprop(model2.parameters(), lr=learning_rate)
epochs = 2000
loss_train = np.zeros(epochs)
loss_test  = np.zeros(epochs)
costw = torch.Tensor([[1,0.1]])
for t in range(epochs):
    # Forward pass: compute predicted y by passing x to the model.
    u = model2(x_train).squeeze()
    #print("size of u",u.shape)
    # Compute and save loss. 
    nextx1 = x1.detach().clone()
    nextx2 = x2.detach().clone()
    stacked = torch.stack([nextx1,nextx2])
    cost_value = torch.zeros(1,train_count)
    
    # try computing single u, do this m times and iterate x, then backprop

    for ii in range(D_out):
        trans_input = torch.transpose(stacked,0,1)
        nextx2 = nextx2.add(3*( g/(2*l)*nextx1 + u[:,ii]/(m*l**2) )*dt).add(model(trans_input)[:,0])
        nextx1 = nextx1 + dt*nextx2
        stacked = torch.stack([nextx1,nextx2])
        cost_value += torch.square(u[:,ii]*0.01)+torch.matmul(costw,torch.square(stacked))
        #print(nextx2.shape)
        #print(nextx1.shape)
    # x2_expected = x2.add(3*( g/(2*l)*x1 + u/(m*l**2) )*dt).add(model(x_train)[:,0])
    # x1_expected = x1 + dt*x2_expected

    # next_x      = torch.stack([x1_expected,x2_expected])
    # cost_value  = torch.square(u)+ \
    #               torch.matmul(costw,torch.square(next_x)*10)
    #               #torch.sum(torch.square(next_x)*100,dim=0)

    loss = loss_fn(cost_value.squeeze(), y_train)
    loss_train[t] = loss.item()

    # compute testing loss
    #u_test       = model2(x_test).squeeze()
    #x2_ex_test   = x2_test.add(3*( g/(2*l)*x1_test + u_test/(m*l**2) )*dt).add(model(x_test)[:,0])
    #x1_ex_test   = x1_test + dt*x2_ex_test
    #next_x_test  = torch.stack([x1_ex_test,x2_ex_test])
    #cost_valuet  = torch.square(u_test)+torch.matmul(costw,torch.square(next_x_test)*10)
    #losst        = loss_fn(cost_valuet.squeeze(), y_test)
    #loss_test[t] = losst.item()
    #test_loss = loss_fn(model2(x_test),y_test)
    #loss_test[t] = test_loss.item()

    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its parameters
    optimizer.step()


plt.figure()
plt.semilogy(loss_train,label="Train")
#plt.semilogy(loss_test,label='Test')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("MSE Loss")
plt.title("Trained on "+str(train_count)+" samples")
plt.show()







#%%
env = pendulum.PendulumEnv(max_torque = 50,max_speed=50)
env.seed(14)
obs = env.reset()
dt=0.1
env.dt=dt
u_vec = []
x_vec = []
xdot_vec=[]
m = env.m
l = env.l
g = env.g
n = 100
######### feedback linearization with nn model #########
for ii in range(n):
    # feedback linearization
    x1,x2 = env.state # theta, theta-dot
    #print("x1,x2",x1,x2)
    # cancel out the nonlinear term
    x = torch.Tensor([[x1,x2]])
    #print(x.shape)
    #print(torch.Tensor([[x1,x2]]).shape)
    u = model2(x).detach().numpy()[0][0] #.float()
    #print("u",u)
    #nonlinear_cancellation = 3*g/(2*l)*x1 + float(y[1].detach().numpy())*m*l**2/3
    #nonlinear_cancellation = -g/2*( 20*float(y[0].detach().numpy())/(3*g*env.dt) +x1 )
    #print(x1,nonlinear_cancellation,m*l*g/2 * np.sin(x1 + np.pi))
    # -m*l*g/2 * np.sin(x1 + np.pi)
    #print(2*np.pi-nonlinear_cancellation)
    #u = -alpha*x2-beta*x1+nonlinear_cancellation

    # apply action
    obs,_,_,_ = env.step([u]) # take an action
    x_vec.append(x1)
    u_vec.append(u)
    xdot_vec.append(x2)

    if np.abs(x1) <= 0.001 and np.abs(x2) <=0.001:
        print(ii)
        break

# plot results
t = env.dt*np.arange(len(x_vec))
plt.figure()
plt.subplot(211)
plt.plot(t,x_vec,label=r"$\theta$")
plt.plot(t,xdot_vec,label=r"$\dot{\theta}$")
plt.ylabel(r"$\circ \ \ \ \ \circ/s$")
plt.legend()
plt.title("feedback linearization with NN")
plt.subplot(212)
plt.plot(t,u_vec,label="u")
plt.xlabel('time (s)')
plt.ylabel(r"$N \cdot m$")
plt.legend()
plt.show() 
# %%
