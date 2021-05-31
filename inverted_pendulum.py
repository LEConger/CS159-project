#%%
# import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# import gym.envs.classic_control.pendulum as pendulum
import torch
from utils import *
%load_ext autoreload
%autoreload 2


#%% ############# feedback linearization ########################

num_sims       = 1
num_time_steps = 100

generate_data(num_sims,num_time_steps)


#%% ############# linearization approximation #####################

alpha = 1
beta  = 5.5
generate_data(num_sims,
                num_time_steps,
                control_method="linearized approx",
                alpha=alpha,
                beta=beta)

#%% ############### Learn nonlinearity with neural network; include noise #############
num_sims       = 500
num_time_steps = 10
x_vec,xdot_vec,deltax,deltaxd,u_vec = generate_data(num_sims,
                                                    num_time_steps,
                                                    control_method="generate data",
                                                    noise_level = 1,
                                                    return_training_data=True,
                                                    alpha=alpha,
                                                    beta=beta)

#%% ############# train model ####################

xx = torch.Tensor(np.transpose(np.array([x_vec,xdot_vec])))
yy = torch.Tensor(np.transpose(np.array([deltax,deltaxd])))
D_in = 2 # x1, x2
H = 100
D_out = 2 # delta x, delta x dot

train_count = int(0.90 * len(x_vec)) 
train_idx   = np.random.choice(len(x_vec),size=train_count,replace=False)
mask        = np.zeros(len(x_vec),dtype=bool)
mask[train_idx] = True
x_train     = xx[mask,:]
x_test      = xx[~mask,:]
y_train     = yy[mask,:]
y_test      = yy[~mask,:]


model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
epochs = 4000
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
plt.show()


#%% ############ check NN #################
dt = 0.1
g  = 10
plt.figure()
plt.plot(x_train[:,0].detach().numpy(),model(x_train)[:,0].detach().numpy(),'.')
plt.plot(x_train[:,0].detach().numpy(),y_train[:,0].detach().numpy(),'.')
plt.plot(x_train[:,0].detach().numpy(),dt*3*g/20*(np.sin(x_train[:,0].detach().numpy())-
      x_train[:,0].detach().numpy()),'.')
plt.plot(x_test[:,0].detach().numpy(),dt*3*g/20*(np.sin(x_test[:,0].detach().numpy())-
      x_test[:,0].detach().numpy()),'.')



#%% run simulation with model

num_sims       = 1
num_time_steps = 100
beta = 1
generate_data(num_sims,
                  num_time_steps,
                  control_method="model",
                  return_training_data=False,
                  noise_level=0,
                  model=model,
                  alpha=alpha,
                  beta=beta)


# %% todo if time
# ablation study - cutting out organ - 
# what if I have less training data, 
# less epochs, 
# how does model perform? which parameters impact network?
# 
# see if I can get it to overfit
# seeing if excessive architecture can prevent overfitting
# if so, then use lipschitz coeff
# also try using less data