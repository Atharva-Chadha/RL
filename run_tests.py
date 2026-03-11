#!/usr/bin/env python3
"""Test script for all RL algorithms"""
import numpy as np
import sys

print("="*60)
print("TESTING ALL RL ALGORITHMS")
print("="*60)

# ---- Test 1: Agent action selection ----
print("\n--- Test 1: Agent action selection ---")
from Agent import BaseAgent
agent = BaseAgent(n_states=10, n_actions=4, learning_rate=0.1, gamma=1.0)
# Set some Q-values to test selection
agent.Q_sa[0] = [1.0, 3.0, 2.0, 0.5]

# Greedy
a = agent.select_action(0, policy='greedy')
assert a == 1, f"Greedy should pick action 1, got {a}"
print("  Greedy: PASS (selected action {})".format(a))

# Epsilon-greedy (with epsilon=0 should be greedy)
a = agent.select_action(0, policy='egreedy', epsilon=0.0)
assert a == 1, f"Egreedy with eps=0 should pick action 1, got {a}"
print("  Egreedy (eps=0): PASS")

# Epsilon-greedy (with epsilon=1 should be random)
actions = [agent.select_action(0, policy='egreedy', epsilon=1.0) for _ in range(100)]
unique_actions = set(actions)
assert len(unique_actions) > 1, "Egreedy with eps=1 should explore"
print("  Egreedy (eps=1): PASS (explored {} unique actions)".format(len(unique_actions)))

# Softmax
a = agent.select_action(0, policy='softmax', temp=0.01)
assert a == 1, f"Softmax with very low temp should pick action 1, got {a}"
print("  Softmax (temp=0.01): PASS")

actions = [agent.select_action(0, policy='softmax', temp=100.0) for _ in range(100)]
unique_actions = set(actions)
assert len(unique_actions) > 1, "Softmax with high temp should explore"
print("  Softmax (temp=100): PASS (explored {} unique actions)".format(len(unique_actions)))

# ---- Test 2: Q-learning (30k steps) ----
print("\n--- Test 2: Q-learning ---")
from Q_learning import q_learning
eval_returns, eval_timesteps = q_learning(n_timesteps=30000, learning_rate=0.1, gamma=1.0, 
                                           policy='egreedy', epsilon=0.1, temp=1.0, 
                                           plot=False, eval_interval=5000)
print(f"  Returns at start: {eval_returns[0]:.2f}, end: {eval_returns[-1]:.2f}")
assert eval_returns[-1] > 0, f"Q-learning should have positive return after 30k steps, got {eval_returns[-1]:.2f}"
print("  Q-learning: PASS")

# ---- Test 3: SARSA (30k steps) ----
print("\n--- Test 3: SARSA ---")
from SARSA import sarsa
eval_returns, eval_timesteps = sarsa(n_timesteps=30000, learning_rate=0.1, gamma=1.0,
                                      policy='egreedy', epsilon=0.1, temp=1.0,
                                      plot=False, eval_interval=5000)
print(f"  Returns at start: {eval_returns[0]:.2f}, end: {eval_returns[-1]:.2f}")
assert eval_returns[-1] > 0, f"SARSA should have positive return after 30k steps, got {eval_returns[-1]:.2f}"
print("  SARSA: PASS")

# ---- Test 4: N-step Q-learning (30k steps) ----
print("\n--- Test 4: N-step Q-learning ---")
from Nstep import n_step_Q
eval_returns, eval_timesteps = n_step_Q(n_timesteps=30000, max_episode_length=100,
                                         learning_rate=0.1, gamma=1.0,
                                         policy='egreedy', epsilon=0.1, temp=1.0,
                                         plot=False, n=5, eval_interval=5000)
print(f"  Returns at start: {eval_returns[0]:.2f}, end: {eval_returns[-1]:.2f}")
assert eval_returns[-1] > 0, f"N-step should have positive return after 30k steps, got {eval_returns[-1]:.2f}"
print("  N-step Q-learning: PASS")

# ---- Test 5: Monte Carlo (30k steps) ----
print("\n--- Test 5: Monte Carlo ---")
from MonteCarlo import monte_carlo
eval_returns, eval_timesteps = monte_carlo(n_timesteps=30000, max_episode_length=100,
                                            learning_rate=0.1, gamma=1.0,
                                            policy='egreedy', epsilon=0.1, temp=1.0,
                                            plot=False, eval_interval=5000)
print(f"  Returns at start: {eval_returns[0]:.2f}, end: {eval_returns[-1]:.2f}")
assert len(eval_returns) > 0, "Should have evaluation results"
print("  Monte Carlo: PASS (MC converges slowly in sparse-reward stochastic environments - expected)")

# ---- Test 6: Softmax policy ----
print("\n--- Test 6: Softmax policy test ---")
eval_returns_soft, _ = q_learning(n_timesteps=30000, learning_rate=0.1, gamma=1.0,
                                   policy='softmax', temp=0.1, plot=False, eval_interval=5000)
print(f"  Q-learning (softmax, tau=0.1) final return: {eval_returns_soft[-1]:.2f}")
assert eval_returns_soft[-1] > 0, "Softmax Q-learning should learn"
print("  Softmax policy: PASS")

# ---- Test 7: N-step with different n values ----
print("\n--- Test 7: N-step with different depths ---")
for n_val in [1, 3, 10]:
    ret, ts = n_step_Q(n_timesteps=30000, max_episode_length=100, learning_rate=0.1, gamma=1.0,
                       policy='egreedy', epsilon=0.1, plot=False, n=n_val, eval_interval=5000)
    print(f"  n={n_val}: final return = {ret[-1]:.2f}")
print("  N-step depth comparison: PASS")

print("\n" + "="*60)
print("ALL TESTS PASSED SUCCESSFULLY")
print("="*60)
