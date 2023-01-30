import random

import numpy as np


class Agent:
    def __init__(self, number_of_states, alpha, gamma, epsilon):
        self.number_of_states = number_of_states
        self.number_of_actions = 5
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((self.number_of_states, self.number_of_actions))

    def act(self, state):
        """ Produces action based on a given state

        Args:
            state: State of the environment

        Returns:
            Action to perform
        """
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 4)
        else:
            action = np.argmax(self.q_table[state])

        return action

    def update(self, state, new_state, action, reward):
        """ Q-learning

        Args:
            state: Old state
            new_state: Next state
            action: Action taken
            reward: Received reward
        """
        new_state_max = np.max(self.q_table[new_state])
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (
                    reward + self.gamma * new_state_max - self.q_table[state, action])
