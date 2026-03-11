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

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        # TO DO: Add own code
        #pass
        if done: 
            target= r
        else:
            max_next_q = np.max(self.Q_sa[s_next] )
            target = r + self.gamma *max_next_q

        old_value = self.Q_sa[s, a]
        self.Q_sa[s,a] = old_value +self.learning_rate *(target-old_value)
        

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []
    
    # TO DO: Write your Q-learning algorithm here!
    

    current_state= env.reset() #initialize enviromnment &gets initial state

    for step in range(n_timesteps) :

        chosen_action = agent.select_action(current_state, policy, epsilon, temp)

        #simulate 1 step
        next_state, reward, done = env.step(chosen_action)

        #update qvalies
        agent.update(current_state, chosen_action, reward, next_state, done)

        #resets if done,,else conintues
        if done:
            #print("reached thegoal at training step number", step + 1)
            current_state=env.reset()
        else:

            current_state= next_state

        #evaluate for every eval_interval steps
        if (step+1) %eval_interval== 0:
            eval_timesteps.append(step+1)
            eval_returns.append( agent.evaluate(eval_env) )
    
    return np.array(eval_returns),np.array(eval_timesteps)
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution



def test():
    
    n_timesteps = 10000
    eval_interval=1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
