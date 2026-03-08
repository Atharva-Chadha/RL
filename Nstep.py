#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent
from Helper import linear_anneal

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(rewards)
        # Snapshot Q-values to avoid mixing old/new values within the same episode
        Q_snapshot = self.Q_sa.copy()
        for t in range(T_ep):
            m = min(n, T_ep - t)
            # Compute n-step return using power series
            G = sum(self.gamma ** k * rewards[t + k] for k in range(m))
            # Bootstrap from Q if we haven't reached episode end, or episode was truncated (not terminal)
            if t + m < T_ep or not done:
                G += (self.gamma ** m) * np.max(Q_snapshot[states[t + m]])
            # Update Q-value
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - Q_snapshot[states[t], actions[t]])

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of n-step Q-learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    t_total = 0
    next_eval = 0
    while t_total < n_timesteps:
        # Evaluate at regular intervals
        while next_eval <= t_total and next_eval < n_timesteps:
            eval_returns.append(pi.evaluate(eval_env))
            eval_timesteps.append(next_eval)
            next_eval += eval_interval
        
        # Collect an episode
        s = env.reset()
        states = [s]
        actions = []
        rewards = []
        done = False
        
        for _ in range(max_episode_length):
            # Anneal epsilon for exploration-exploitation tradeoff
            if policy == 'egreedy' and epsilon is not None:
                epsilon_t = linear_anneal(t_total, n_timesteps, epsilon, 0.01, 0.75)
            else:
                epsilon_t = epsilon
            a = pi.select_action(s, policy, epsilon_t, temp)
            s_next, r, done = env.step(a)
            states.append(s_next)
            actions.append(a)
            rewards.append(r)
            t_total += 1
            
            if done or t_total >= n_timesteps:
                break
            s = s_next
        
        # Update Q-values after episode
        pi.update(states, actions, rewards, done, n)

    # Final evaluation at last timestep if not yet done
    while next_eval <= t_total and next_eval < n_timesteps:
        eval_returns.append(pi.evaluate(eval_env))
        eval_timesteps.append(next_eval)
        next_eval += eval_interval

    if plot:
        env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1)
        
    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
