#######################################
# CartTrainer v0.1 ####################
#######################################

############################
# Imports ##################
############################

import gym 
import numpy as np
import random
from IPython.display import clear_output
import math 
import time
from sklearn.preprocessing import KBinsDiscretizer

############################
# GameTrainer Class ########
############################

class GameTrainer:
    def __init__(self, alpha = 0.1, gamma = 0.6, epsilon = 0.1, buckets = (3, 3, 6, 6)):
        
        self.env = gym.make("CartPole-v1").env
        self.buckets = buckets

        
        self.Qmatrix = np.zeros(self.buckets + (self.env.action_space.n,))            
        
        #hyperparams
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        
        #Storing for stats
        self.all_epochs = []
        self.all_penalties = []
        
        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]
        

    def discretizeState(self, obs):
        discretized = list()
        for i in range(len(obs)):
            scaling = (obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)
        
    def train(self, n_epochs = 25000, render = True):
        for i in range(1, n_epochs + 1):
            state = self.discretizeState(self.env.reset())
                
            epochs, penalties, reward = 0, 0, 0
            done = False
            
            while not done:
                if(render):
                    self.env.render()
                
                if(random.uniform(0,1) < self.epsilon):
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Qmatrix[state])
                    
                next_state, reward, done, info = self.env.step(action)
                

                old_value = self.Qmatrix[state][action]
                    

                next_state = self.discretizeState(next_state)
                state = next_state
                    
                next_max = np.max(self.Qmatrix[next_state])
                new_value = (1-self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

                self.Qmatrix[state][action] = new_value
                 
                epochs +=1
                
            if(i % 100 == 0):
                clear_output(wait = True)
                print(f"Episode: {i}")
                
        print("Q-learning train ended after ", n_epochs, " epochs.")
        
        
    def test(self, nTests = 1):
        for i in range(1, nTests+1):
            state = self.env.reset()

            done = False
            while not done: 
                state = self.discretizeState(self.env.reset())
                time.sleep(0.03)
                
                action = np.argmax(self.Qmatrix[state])
                state, reward, done, info = self.env.step(action)
                self.env.render()
                        
                
        print('Done !')
        
    def displayRandomGameCartPole(self, nTests = 1):
        env = gym.make('CartPole-v0')
        env.reset()
        for _ in range(300):
            env.render()
            time.sleep(0.03)
            env.step(env.action_space.sample()) # take a random action
        env.close()
                
    #def displayPolicyGame(self):
        
    

############################
# Code Example Usage #######
############################




#gt = GameTrainer(epsilon = 0.1, alpha = 0.1)
#gt.displayRandomGameCartPole()
#gt.train(n_epochs=5000, render = False)
#aaa = gt.Qmatrix
#gt.test(nTests=10)

gt = GameTrainer(alpha = 0.1, gamma = 0.6, epsilon = 0.1)
gt.displayRandomGameCartPole()
gt.train(n_epochs = 5000, render = False)
gt.test()









