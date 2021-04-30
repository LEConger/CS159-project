import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gym.envs.classic_control.pendulum as pendulum
import torch

#%%
env = pendulum.PendulumEnv(max_torque = 50,max_speed=50)
env.seed(14)
obs = env.reset()
dt = 0.1
env.dt=dt
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
N = 5
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

#%% now let's try learning the optimal control, not just the nonlinearity
xx          = torch.Tensor(np.transpose(np.array([x_vec,xdot_vec])))
yy          = torch.Tensor(np.transpose(np.array([deltax,deltaxd])))

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

xx          = torch.Tensor(np.transpose(np.array([x_vec,xdot_vec])))

D_out = 1
model2 = torch.nn.Sequential(
torch.nn.Linear(D_in, H),
torch.nn.ReLU(),
torch.nn.Linear(H, H),
torch.nn.Tanh(),
torch.nn.Linear(H, H),
torch.nn.Sigmoid(),
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
print("x1,x2",x1.shape,x2.shape)

# train network
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.RMSprop(model2.parameters(), lr=learning_rate)
epochs = 1000
loss_train = np.zeros(epochs)
loss_test  = np.zeros(epochs)
for t in range(epochs):
    # Forward pass: compute predicted y by passing x to the model.
    u = model2(x_train).squeeze()
    
    # Compute and save loss. 
    x2_expected = x2.add(3*( g/(2*l)*x1 + u/(m*l**2) )*dt).add(model(x_train)[:,0])
    x1_expected = x1 + dt*x2_expected

    next_x      = torch.stack([x1_expected,x2_expected])
    #print("next_x",next_x.shape)
    cost_value  = torch.square(u)+torch.square(next_x)*10

    loss = loss_fn(cost_value, y_train)
    loss_train[t] = loss.item()

    # compute testing loss
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
env.dt=0.1
alpha = 1
beta  = 1
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