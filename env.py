import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

def get_reward(prob, n = 10):
    """Create a random integer reward in proportion to the assigned probabilities
    to the levers
    """
    reward = 0
    for i in range(n):
        if random.random() < prob:
            reward += 1
    return reward

def update_record(record, action, r):
    """Update the moving avarage with the new action reward
    for a given action and new reward
    """

    new_r = (record[action,0] * record[action,1] +r) / (record[action,0] + 1)
    record[action, 0] += 1
    record[action, 1] = new_r

    return record

def get_best_arm(record):
    """Finds the arm associated with the highest avarage reward
    """
    arm_index = np.argmax(record[:,1], axis = 0)
    return arm_index

def softmax(av, tau=1.12):
    """Return softmax probabilities of the average rewards
    with a given temperature parameter tau
    """
    softm = np.exp(av/tau) / np.sum(np.exp(av/tau))
    return softm