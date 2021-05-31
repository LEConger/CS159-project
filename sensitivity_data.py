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

import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import *


#%% ablation
alpha = 1
beta  = 5.5
num_sims       = 500
num_time_steps = 10
x_vec,xdot_vec,deltax,deltaxd,u_vec = generate_data(num_sims,
                                                    num_time_steps,
                                                    control_method="generate data",
                                                    noise_level = 1,
                                                    return_training_data=True,
                                                    alpha = alpha,
                                                    beta = beta)
# %% train on varying amounts of data on an excessively large NN

data_amount = np.arange(0.01,0.1,0.01) # in increments of 5%
xx          = torch.Tensor(np.transpose(np.array([x_vec,xdot_vec])))
yy          = torch.Tensor(np.transpose(np.array([deltax,deltaxd])))
r           = len(x_vec)

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
    keep_count  = int(data_amount[ii]*r)
    train_count = int(0.80 * keep_count)
    keep_idx    = np.random.choice(r,size=keep_count,replace=False)
    train_idx   = np.random.choice(keep_count,size=train_count,replace=False)
    mask        = np.zeros(r,dtype=bool)
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

    generate_data(1,
                  100,
                  control_method="model",
                  return_training_data=False,
                  noise_level=0,
                  model=model)



# %%
