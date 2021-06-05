import sys
sys.path.append('./Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/')
from gym.envs.classic_control import pendulum
from control_objects.probabilistic_gp_mpc_controller import ProbabiliticGpMpcController
from control_objects.utils import LivePlotClass, LivePlotClassParallel
import numpy as np
import matplotlib.pyplot as plt
import gym
import json
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import time
import os
import datetime



env_to_control = 'Pendulum-v0'
hyperparameters_init = {"noise_std": [1e-2, 1e-2, 1e-2],
                        "lengthscale": [[0.75, 0.75, 0.75, 0.75],
                                        [0.75, 0.75, 0.75, 0.75],
                                        [0.75, 0.75, 0.75, 0.75]],
                        "scale": [5e-2, 5e-2, 5e-2]}
params_constraints_hyperparams = {"min_std_noise": 1e-3,
                                  "max_std_noise": 3e-1,
                                  "min_outputscale":  1e-10,
                                  "max_outputscale": 1e2,
                                  "min_lengthscale":  4e-5,
                                  "max_lengthscale": 25.0}
params_controller = {"target_state": [1, 0.5, 0.5],
                     "weights_target_state": [1, 0.1, 0.1],
                     "weights_target_state_terminal_cost": [10, 3, 3],
                     "target_action": [0.5],
                     "weights_target_action": [0.1],
                     "s_observation": [1e-6, 1e-6, 1e-6],
                     "len_horizon": 15,
                     "exploration_factor": 3,
                     "limit_derivative_actions": 0,
                     "max_derivative_actions_norm": [0.05],
                     "num_repeat_actions": 1,
                     "clip_lower_bound_cost_to_0": 0,
                     "compute_factorization_each_iteration": 1}
params_constraints_states = {"use_constraints": 0,
                             "states_min": [-0.1, 0.05, 0.05],
                             "states_max":  [1.1, 0.95, 0.925],
                             "area_penalty_multiplier": 2}
params_train = {"lr_train": 7e-3,
                "n_iter_train": 15,
                "train_every_n_points": 10,
                "clip_grad_value": 1e-3,
                "print_train": 0,
                "step_print_train": 5}
params_actions_optimizer = {"disp": None,
                            "maxcor": 2,
                            "ftol": 1e-15,
                            "gtol": 1e-15,
                            "eps": 1e-2,
                            "maxfun": 2,
                            "maxiter": 2,
                            "iprint": -1,
                            "maxls": 2,
                            "finite_diff_rel_step": None}
params_memory = {"min_error_prediction_states_for_storage": [5e-4,
                                                             5e-4, 5e-4],
                 "min_prediction_states_std_for_storage":  [4e-3,
                                                            4e-3, 4e-3]}

num_repeat_actions = 1


target_state = np.array(params_controller['target_state'])
weights_target_state = np.diag(params_controller['weights_target_state'])
weights_target_state_terminal_cost \
    = np.diag(params_controller['weights_target_state_terminal_cost'])
target_action = np.array(params_controller['target_action'])
weights_target_action = np.diag(params_controller['weights_target_action'])


def state_to_obs(state):
    theta, thetadot = state
    return np.array([np.cos(theta), np.sin(theta), thetadot])


def obs_to_state(obs):
    co, si, thetadot = obs
    theta = np.arctan2(si, co)
    return np.array([theta, thetadot])


def train_gp_model(num_training_rollouts, num_training_actions, num_test_steps,
                   training_input_range, obs_variance, max_control_torque=50,
                   alpha=1, beta=5.5):
    env = pendulum.PendulumEnv(max_torque=max_control_torque)
    env.reset()

    s_observation = np.diag(np.full(3, obs_variance))

    datetime_now = datetime.datetime.now()
    folder_save = os.path.join('folder_save', env_to_control, 'y' + str(datetime_now.year) \
                + '_mon' + str(datetime_now.month) + '_d' + str(datetime_now.day) + '_h' + str(datetime_now.hour) \
                + '_min' + str(datetime_now.minute) + '_s' + str(datetime_now.second))
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    live_plot_obj = LivePlotClass(num_training_rollouts*num_training_actions
                                  + num_test_steps,
                                  env.observation_space,
                                  env.action_space,
                                  params_constraints_states,
                                  num_repeat_actions)

    control_object = ProbabiliticGpMpcController(env.observation_space,
                                                 env.action_space,
                                                 params_controller,
                                                 params_train,
                                                 params_actions_optimizer,
                                                 params_constraints_states,
                                                 hyperparameters_init,
                                                 target_state,
                                                 weights_target_state,
                                                 weights_target_state_terminal_cost,
                                                 target_action,
                                                 weights_target_action,
                                                 params_constraints_hyperparams,
                                                 env_to_control,
                                                 folder_save,
                                                 num_repeat_actions)

    for idx_rollout in range(num_training_rollouts):
        observation, reward, done, info = env.step(env.action_space.sample())
        for idx_action in range(num_training_actions):
    #         observation = env.observation_space.sample()#state_to_obs(env.state)
    #         env.state = obs_to_state(observation)
            x1, x2 = env.state
            action = np.random.uniform(-training_input_range,
                                                             training_input_range) # env.action_space.sample() -alpha*x2 - beta*x1 +
            noise = np.sqrt(obs_variance)*np.random.randn(1)
            control_object.action = action + noise

            new_observation, reward, _, _ = env.step(action + noise)
            try:
                env.render()
            except:
                pass
            live_plot_obj.add_point_update(observation, action)
            control_object.add_point_memory(observation, action, new_observation, reward)
            observation = new_observation
        env.reset()

    losses_tests = np.ones(num_test_steps // num_repeat_actions)
    for index_iter in range(num_test_steps):
        observation = state_to_obs(env.state)
        action, add_info_dict = control_object.compute_prediction_action(observation, s_observation)
        for idx_action in range(num_repeat_actions):
            new_observation, reward, done, info = env.step(action)
            try:
                env.render()
            except:
                pass
        losses_tests[index_iter] = add_info_dict['cost']
        control_object.add_point_memory(observation, action, new_observation, reward,
                                        add_info_dict=add_info_dict, params_memory=params_memory)
        live_plot_obj.add_point_update(observation, action, add_info_dict)
        observation = new_observation
    # env.__exit__()
    env.close()
    return control_object

