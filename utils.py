import numpy as np
import matplotlib.pyplot as plt
import gym.envs.classic_control.pendulum as pendulum
import torch



# function for data generation

def generate_data(num_sims,
                  num_time_steps,
                  control_method="feedback linearization",
                  return_training_data=False,
                  noise_level=0,
                  model=None,
                  alpha=1,
                  beta=1):
    ''' 
        num_sims: scalar, number of simulations, each has random initial condition
        num_time_steps: scalar, length of each simulation
        control_method: "generate data", "feedback linearization","linearized approx","nn model"
        return_training_data: if True, returns theta, theta-dot, delta-theta, delta-theta-dot, and u_vec vectors
                            o/w only plots trajectories
        noise level: scalar, magnitude of input noise
        model: trained model
        alpha: feeback linearization weight for theta
        beta: feedback linearization weight for theta-dot
    '''
    
    # set up parameters
    env = pendulum.PendulumEnv(max_torque = 50,max_speed=50)
    env.seed(14)
    obs      = env.reset()
    dt       = 0.1
    env.dt   = dt
    u_vec    = []
    x_vec    = []
    xdot_vec = []
    deltax   = [] # expected - actual for x1
    deltaxd  = [] # expected - actual for x2
    m        = env.m
    l        = env.l
    g        = env.g

    for jj in range(num_sims):
    
        # make new environment with random initial condition
        if num_sims>1:
            env.close()
            env = pendulum.PendulumEnv(max_torque = 50,max_speed=50)
            env.reset()
            env.dt=0.1

        for ii in range(num_time_steps):
            # feedback linearization
            x1,x2 = env.state # theta, theta-dot
            
            if control_method =="generate data":
                u = use_generate_data(alpha,beta,x1,x2,noise_level)
            elif control_method == "feedback linearization":
                u = use_feedback_linearization(alpha,beta,x1,x2,noise_level,m,l,g)
            elif control_method == "linearized approx":
                u = use_linear_approx(alpha,beta,x1,x2,noise_level)
            elif control_method == "nn model":
                u = use_nn_model(model,alpha,beta,x1,x2,noise_level,g,dt)
            elif control_method == "local regression model":
                u = use_local_regression_model(model,alpha,beta,x1,x2,noise_level,g,dt)
            
            # apply action
            env.step([u]) # take an action
            
            # save next steps in trajectory
            x_vec.append(x1)
            xdot_vec.append(x2)
            u_vec.append(u)

            if control_method=="generate data":
                # sample state; compute difference between expected and actual
                # we are assuming x_dot = Ax + Bu + f(x) where f(x) is a nonlinearity
                x1_new,x2_new = env.state # theta, theta-dot
                x2_expected = x2 + 3*( g/(2*l)*x1 + u/(m*l**2) )*env.dt
                x1_expected = x1 + env.dt*x2_expected
                # compute and store differences
                deltax.append(x1_new-x1_expected)
                deltaxd.append(x2_new-x2_expected)

            # stop simulation if we have achieved our goal
            if np.abs(x1) <= 0.001 and np.abs(x2) <=0.001:
                print("Simulation completed after "+str(ii)+" time steps.")
                break

    # simulation is done; plot results
    if control_method=="generate data":
        plot_series(x_vec,xdot_vec,deltax,deltaxd,dt)
    else:
        plot_single(x_vec,xdot_vec,u_vec,dt,control_method)
    
    # close environment
    env.close()
    
    # if we wanted the data, return what we have
    if return_training_data:
        return x_vec,xdot_vec,deltax,deltaxd,u_vec


def use_nn_model(model,alpha,beta,x1,x2,noise_level,g,dt):
    x = torch.Tensor([x1,x2])
    y = model(x.float())
    nonlinear_cancellation = -g/2*( 20*float(y[0].detach().numpy())/(3*g*dt) +x1 )
    u = -alpha*x2-beta*x1+nonlinear_cancellation + np.random.randn()*noise_level
    return u

def use_local_regression_model(model,alpha,beta,x1,x2,noise_level,g,dt):
    # prediction using model trained over sim/steps
    print("TADAAAAAA")
    local_regression, training_data = model
    y, y_dot = local_regression(training_data, x1)
    nonlinear_cancellation = -g/2*( 20*float(y[0])/(3*g*dt) +x1 )
    u = -alpha*x2-beta*x1+nonlinear_cancellation + np.random.randn()*noise_level
    return u


def use_feedback_linearization(alpha,beta,x1,x2,noise_level,m,l,g):
    nonlinear_cancellation = -m*l*g/2 * np.sin(x1 + np.pi) + np.random.randn()*noise_level
    u = -beta*x2 - alpha*x1 - nonlinear_cancellation 
    return u

def use_linear_approx(alpha,beta,x1,x2,noise_level):
    u = -beta*x1 - alpha*x2 + np.random.randn()*noise_level
    return u

def use_generate_data(alpha,beta,x1,x2,noise_level):
    input_term = np.random.uniform(-6,6)
    u = -alpha*x2 - beta*x1 + input_term + np.random.randn()*noise_level
    return u

def plot_series(x_vec,xdot_vec,deltax,deltaxd,dt):
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

def plot_single(x_vec,xdot_vec,u_vec,dt,control_method):
    t = dt*np.arange(len(x_vec))
    plt.figure()
    plt.subplot(211)
    plt.plot(t,x_vec,label=r"$\theta$")
    plt.plot(t,xdot_vec,label=r"$\dot{\theta}$")
    plt.ylabel(r"$\circ \ \ \ \ \circ/s$")
    plt.legend()
    plt.title(control_method.replace("_"," "))
    plt.subplot(212)
    plt.plot(t,u_vec,label="u")
    plt.xlabel('time (s)')
    plt.ylabel(r"$N \cdot m$")
    plt.legend()
    plt.show() 