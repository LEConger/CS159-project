# %%
# ablation study - cutting out organ - 
# what if I have less training data, 
# less epochs, 
# how does model perform? which parameters impact network?
# 
# see if I can get it to overfit
# seeing if excessive architecture can prevent overfitting
# if so, then use lipschitz coeff
# also try using less data

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gym.envs.classic_control.pendulum as pendulum
import torch
import pickle
# load nn
# pickle.load('nn_model.pkl' , 'rb' )


#%% ablation

env = pendulum.PendulumEnv(max_torque = 50,max_speed=50)
env.seed(14)
obs = env.reset()
env.dt=0.1
alpha = 1
beta  = 1
u_vec = []
x_vec = []
xdot_vec=[]
deltax = [] # expected - actual for x1
deltaxd = [] # expected - actual for x2
m = env.m
l = env.l
g = env.g
N = 10
M = 100 # number of initial conditions


#%% generate training data

for jj in range(M):
    
    # make new environment with random initial condition
    env.close()
    env = pendulum.PendulumEnv(max_torque = 50,max_speed=50)
    env.reset()
    env.dt=0.1

    for ii in range(N):
        # feedback linearization
        x1,x2 = env.state # theta, theta-dot
        input_term = np.random.uniform(-6,6)

        # add random input
        u = -alpha*x2 - beta*x1 + input_term + np.random.randn()

        # apply action
        env.step([u]) # take an action
        x_vec.append(x1)
        xdot_vec.append(x2)
        u_vec.append(u)

        # sample state; compute difference between expected and actual
        # we are assuming x_dot = Ax + Bu + f(x) where f(x) is a nonlinearity
        x1_new,x2_new = env.state # theta, theta-dot
        x2_expected = x2 + 3*( g/(2*l)*x1 + u/(m*l**2) )*env.dt
        #x2_expected = x2 + 3*( -g/(2*l)*np.sin(x1+np.pi) + u/(m*l**2) )*env.dt
        x1_expected = x1 + env.dt*x2_expected
        # compute and store differences
        deltax.append(x1_new-x1_expected)
        deltaxd.append(x2_new-x2_expected)

#%% view training data
plt.figure(figsize=(15,15))
plt.subplot(4,1,1)
plt.plot(x_vec,deltax,'.')
plt.ylabel(r"$\Delta x$")
plt.xlabel("x")

plt.subplot(4,1,2)
plt.plot(xdot_vec,deltax,'.')
plt.ylabel(r"$\Delta x$")
plt.xlabel(r"$\dot{x}$")

plt.subplot(4,1,3)
plt.plot(x_vec,deltaxd,'.')
plt.ylabel(r"$\Delta \dot{x}$")
plt.xlabel("x")

plt.subplot(4,1,4)
plt.plot(xdot_vec,deltaxd,'.')
plt.ylabel(r"$\Delta \dot{x}$")
plt.xlabel(r"$\dot{x}$")

plt.figure()
plt.plot(x_vec,'.')
plt.plot(xdot_vec,'.')
plt.plot(u_vec,'.')
# %% train on varying amounts of data on an excessively large NN

data_amount = np.arange(0.01,0.2,0.02) # in increments of 5%
xx          = torch.Tensor(np.transpose(np.array([x_vec,xdot_vec])))
yy          = torch.Tensor(np.transpose(np.array([deltax,deltaxd])))
N           = len(x_vec)

# NN parameters
D_in  = 2 # x1, x2
H     = 100
D_out = 2 # delta x, delta x dot

for ii in range(len(data_amount)):
    model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(), #Tanh
    torch.nn.Linear(H, H),
    torch.nn.ReLU(), #Sigmoid
    torch.nn.Linear(H, H),
    torch.nn.ReLU(), #Tanh
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
    )

    # pick data
    keep_count  = int(data_amount[ii]*N)
    train_count = int(0.80 * keep_count)
    keep_idx    = np.random.choice(N,size=keep_count,replace=False)
    train_idx   = np.random.choice(keep_count,size=train_count,replace=False)
    mask        = np.zeros(N,dtype=bool)
    train_mask  = np.zeros(keep_count,dtype=bool)
    train_mask[train_idx] = True
    mask[keep_idx[train_mask]] = True
    x_train     = xx[mask,:]
    y_train     = yy[mask,:]
    mask[keep_idx[train_mask]] = False
    mask[keep_idx[~train_mask]] = True
    x_test      = xx[mask,:]
    y_test      = yy[mask,:]

    # train network
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-4
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    epochs = 5000
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


#%%

# %%
