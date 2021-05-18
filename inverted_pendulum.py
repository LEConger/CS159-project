#%%
#from IPython import display
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#matplotlib inline

env = gym.make('Pendulum-v0')

env.dt=1e-7
print(env.dt)
obs = env.reset()
print(obs)
alpha = 1
beta  = 1
u_vec = []
x_vec = []
xdot_vec=[]
m = env.m
l = env.l
g = env.g

for ii in range(200):
    # compute action based on linearized system
    #x1 = np.sqrt(obs[0]**2+obs[1]**2)
    #x2 = obs[2]
    #u = -alpha*x1 - beta*x2 

    # adjust based on NN


    # feedback linearization
    #x1 = np.arctan2(obs[1],obs[0]) # fix this - use tangent inverse
    x1,x2 = env.state
    #x2 = obs[2]
    nonlinear_cancellation = m*l*g/2 * np.sin(x1 + np.pi)
    #u = -beta*x2-alpha*x1 - nonlinear_cancellation #-alpha*x1 - beta*x2 - obs[1] # here, obs[1] is the nonlinearity
    u = -beta*(x1-np.pi/2)-alpha*x2 #-beta*(x1-np.pi/2)-alpha*x2 + nonlinear_cancellation
    
    # apply action
    env.step([u]) # take a random action env.action_space.sample()
    x_vec.append(x1)
    u_vec.append(u)
    xdot_vec.append(x2)

    if np.abs(obs[0]) <= 0.001:
        print(ii)
        break


env.close()

plt.figure()
plt.plot(x_vec,label=r"$\theta$")
plt.plot(u_vec,label="u")
plt.plot(xdot_vec,label=r"$\dot{\theta}$")
plt.legend()
plt.show()
