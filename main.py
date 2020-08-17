import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

from env import get_reward, update_record, get_best_arm, softmax


n = 10 #number of levers
probs = np.random.rand(n) #probabilities for each lever
print(f"Lever probs: {probs}")
print(f"Maximum reward given lever is {np.argmax(probs)}")
eps = 0.3 #epsilon-greedy parameter

fig, ax = plt.subplots(1,1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")

record = np.zeros((n,2))

rewards = [0]

# Use softmax or alternatively "epsilon-greedy" strategy
use_softmax = True

# ... we can get into the main loop for playing the n-armed bandit game. If a random
# number is greater than the epsilon parameter, we just calculate the best action using
# the get_best_arm function and take that action. Otherwise we take a random action
# to ensure some amount of exploration. After choosing the arm, we use the get_reward
# function and observe the reward value. We then update the record array with this new
# observation. We repeat this process a bunch of times, and it will continually update
# the record array. The arm with the highest reward probability should eventually get
# chosen most often, since it will give out the highest average reward.
# excerpt from the Book: Deep Reinforcement Learning in Action A.Zai, B. Brown

for i in range(500):
    if(use_softmax):
        # use softmax probabilities of values to choose action
        p = softmax(record[:,1])
        choice = np.random.choice(np.arange(n), p = p)
    else:
        # use for epsilon greedy approach
        if random.random() > eps:
            choice = get_best_arm(record)
        else:
            choice = np.random.randint(10)
    
    # computes the reward for choosing the arm
    r = get_reward(probs[choice]) 
    # updates the Q-tale
    record = update_record(record, choice, r)
    #print(f"Updated record: {record}")

    #keeps track of the running average of rewards
    mean_reward = ((i+1) * rewards[-1] + r) / (i+2) 
    rewards.append(mean_reward)

# plot the cumalative rewards vs iteration
ax.scatter(np.arange(len(rewards)), rewards)
plt.savefig("./cumulative_rewards.png")