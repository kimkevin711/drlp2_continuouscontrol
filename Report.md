
## Describing the Learning Algorithm

This solution uses the Deep Q Network outlined in Google's DQN paper in Nature (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). The code for the agent and model are modified from dqn_agent.py and model.py given in the Deep Q Networks Lesson. The model used for the local and target networks is a fully connected neural network with two hidden layers with 64 nodes each. The input to the network is the state vector (size 37) and the output is the action space (size 4). This architecture is used for both the local and target networks. 

The agent uses an epsilon-greedy policy with an experience replay buffer. The total buffer size is 100,000, the batch size for sampling is 64, the discount factor is 0.99, the soft update tau factor is 0.001, the learning rate is 0.0005, and the networks are updated every 4 time steps. The algorithm first experiences episodes until batch size is met. Then, the actions are taken following epsilon greedy and the reward/state/action are stored in the experience replay. From the experience buffer, a randomized sample of batch size is used to learn from (which decorrelates temporal data). In addition to the experience replay, the DQN algorithm used here decouples the target and local Q action-value function to further reduce correlations. 


## Plot of Rewards

The plot of the rewards at each episode is given below. 

The number of episodes required for the agent to "solve" the environment is 445 episodes.
 
While it only took 445 episodes to get to the minimum +13 reward, a weight file with a higher average reward of ~16 is chosen to test. 

(reward_plot.jpg)


## Ideas for Future Work

There are many hyperparameters that can be tuned to improve the agent's performance. First, it is noticed that stopping the training after it averages a score of 13 may not result in an optimal solution, since, the agent was observed to get stuck in some episodes. This leads to the idea of tweaking epsilon and its decay rate in order to sway the agent from taking too many random movements at the start of training, which may affect the final behavior. In addition, hyperparamters that are related to experience replay (buffer size, batch size, update rate) can be tweaked to see how the agent behavior may change. For example, it would be interesting to see the tradeoff between training time and agent robustness by tweaking the experience memory size. Finally, changing the architecture of the Q networks would be a very useful investigation. Presumably, the pixel navigation project uses a convolutional network, so that would be a point of interest (e.g., adding additional hidden layers).
