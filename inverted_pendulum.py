#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gym.envs.classic_control.pendulum as pendulum
import torch
from utils import *
%reload_ext autoreload
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

#%% ############### Generate data for local regression; include noise #############
num_sims       = 1
num_time_steps = 50
x_vec,xdot_vec,deltax,deltaxd,_ = generate_data(num_sims,
                                                num_time_steps,
                                                control_method="generate data",
                                                noise_level = 1,
                                                return_training_data=True,
                                                alpha=alpha,
                                                beta=beta)





#%% ############ train local regression ##############

# random 90%/10% split
def train_test_split(xx, yy):
    size = xx.shape[0]
    
    # random train/test split --> indices
    train_count = int(0.90 * size) 
    train_idx   = np.random.choice(size, size=train_count,replace=False)
    mask        = np.zeros(size, dtype=bool)
    mask[train_idx] = True
    
    # selecting instances for splits by index
    x_train     = xx[mask,:]
    x_test      = xx[~mask,:]
    y_train     = yy[mask,:]
    y_test      = yy[~mask,:]
    
    return (x_train, y_train), (x_test, y_test)


# make training data for local regression
def make_training_data_for_LR(num_sims, num_time_steps):
    x_vec,xdot_vec,deltax,deltaxd,_ = generate_data(num_sims,
                                                num_time_steps,
                                                control_method="generate data",
                                                noise_level = 1,
                                                return_training_data=True,
                                                alpha=alpha,
                                                beta=beta)
    
    # shape our training data
    xx = np.transpose(np.array([x_vec, xdot_vec]))  # do we needd a bias term?
    yy = np.transpose(np.array([deltax, deltaxd]))

    # train/test split
    (xx_train, yy_train), (xx_test, yy_test) = train_test_split(xx, yy)
    x_train, xdot_train = xx_train[:, 0], xx_train[:, 1]
    y_train, ydot_train = yy_train[:, 0], yy_train[:, 1]
    x_test, xdot_test = xx_test[:, 0], xx_test[:, 1]
    y_test, ydot_test = yy_test[:, 0], yy_test[:, 1]

    return x_train, y_train, ydot_train


# LOCAL REGRESSION: calculate W weight diagonal Matric used in calculation of predictions
def get_LOWES_weights(x_query, x_train, bandwidth):
    # M is the No of training examples
    M = x_train.shape[0]
    # Initialising W with identity matrix
    W = np.mat(np.eye(M))
    # calculating weights for query points
    for i in range(M):
        xi = x_train[i]
        denominator = (-2 * bandwidth * bandwidth)
        W[i, i] = np.exp(np.dot((xi-x_query), (xi-x_query).T)/denominator)
    return W

# LOCAL REGRESSION: make prediction for one query point x
def predict_one(x_train, y_train, x_test, bandwidth):
    num_samples = x_train.shape[0]
    
    # for 1-dim data, reshape from (n,) to (n,1)
    x_ = np.expand_dims(x_train, axis=1)
    y_ = np.expand_dims(y_train, axis=1)
    
    # stack the input with a column of 1 for bias
    bias = np.ones((num_samples, 1))
    x_ = np.hstack((x_, bias))
    
    # stack a given query point (test) with 1 for bias
    x_query = np.hstack((x_test, 1))
    
    # compute LOWES weight matrix
    W = get_LOWES_weights(x_query, x_, bandwidth)
    
    # calculating parameter theta
    theta = np.linalg.pinv(x_.T*(W * x_))*(x_.T*(W * y_))
    
    # calculating predictions
    pred = np.dot(x_query, theta)
    return theta, pred

# LOCAL REGRESSION: predict over dataset
# instead of pre-training, fit a local regression at inference on the x of choice, with all training data supplemental
# prediction on-the-fly
def local_regression(training_data, x_query):
    x_train, y_train, ydot_train = training_data
    y_pred = []
    ydot_pred = []
    _, y_pred = predict_one(x_train, y_train, x_query, bandwidth=0.1)
    _, ydot_pred = predict_one(x_train, ydot_train, x_query, bandwidth=0.1)

    return [y_pred, ydot_pred]

#%% ############## Test Local Regression on new data
# test on new data
num_sims       = 1
num_time_steps = 100
beta = 1
training_data = make_training_data_for_LR(num_sims=10, num_time_steps=10)
generate_data(num_sims,
              num_time_steps,
              control_method="local regression model",
              return_training_data=False,
              noise_level=0,
              model=[local_regression,training_data],
              alpha=alpha,
              beta=beta)


#%% ############### Lgenerate data for neural network; include noise #############
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

# random train/test split --> indices
train_count = int(0.90 * len(x_vec)) 
train_idx   = np.random.choice(len(x_vec),size=train_count,replace=False)
mask        = np.zeros(len(x_vec),dtype=bool)
mask[train_idx] = True
# selecting instances for splits by index
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
              control_method="nn model",
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