import gymnasium as gym
env = gym.make('CartPole-v1')
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error as mse
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import itertools
import pandas as pd
from Helper import softmax, argmax


class Q_network(Model):
    def __init__(self, num_layers=2, num_units=3):
        super().__init__(input)
        self.in_ = tf.keras.layers.Dense(units=num_units, input_shape=(None,4), activation=tf.nn.tanh)
        self.dense = tf.keras.layers.Dense(units=num_units, activation=tf.nn.tanh)
        self.out = tf.keras.layers.Dense(2)
        self.num_layers = num_layers

    def call(self,x):
        x = self.in_(x)
        for _ in range(self.num_layers-1):
          x = self.dense(x)
        out = self.out(x)
        return out
    

class CartPoleDQN:

    def __init__(self, n_actions=2, num_layers=1, num_units=20, learning_rate=1e-2, decay=0.999, policy = 'softmax', epsilon=0.99, temp=0.99, \
                 min_decay = 0.05, gamma=0.95, batch_size=128, episodes=500, copy_steps=1, replay_size=10000, steps_bf_training=500):
        self.env = gym.make('CartPole-v1')
        self.n_actions = n_actions
        self.agent, _ = self.env.reset()
        self.learning_rate = learning_rate
        self.decay = decay
        self.min_decay = min_decay
        self.policy = policy
        self.gamma = gamma
        self.epsilon = epsilon
        self.temp = temp
        self.Q_net = Q_network(num_layers=num_layers, num_units=num_units)
        self.target_Q = Q_network(num_layers=num_layers, num_units=num_units)
        self.Q_net.build(input_shape=(None,4))
        self.Q_net.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        self.Q_net.summary()
        self.batch_size = batch_size if replay_size else 1
        self.episodes = episodes
        self.copy_steps = copy_steps
        self.replay_size = replay_size
        self.replay_buffer = deque(maxlen=self.replay_size)
        self.total_rewards = []
        self.steps_bf_training = steps_bf_training

    def Q(self, s, target=False):

      Q_net = self.target_Q if target else self.Q_net
      s = np.array(s).reshape((-1,4))

      return Q_net.predict(x=s, verbose=0)
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            if np.random.rand() < epsilon:
                a = np.random.randint(0, self.n_actions)
            else:
                a = argmax(self.Q(s))
            
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            a = np.random.choice(self.n_actions, p=softmax(self.Q(s)[0], temp))
        return a
        
    def update(self, batch):

        r = batch[:,5]
        s_next = batch[:,6:10]
        done = batch[:,-1]
        return np.where(done, r, r + self.gamma * np.amax(self.Q(s_next, target=True), axis=1))

    def sample_and_store(self):

        s = self.agent
        a = self.select_action(s, policy=self.policy, epsilon=self.epsilon, temp=self.temp)
        s_next, r, terminated, truncated, info = self.env.step(a)
        done = terminated or truncated
        if done:
          s_next, info = self.env.reset()
        self.agent = s_next
        transition = np.fromiter(itertools.chain.from_iterable([s, [a], [r], s_next, [terminated]]), np.float64)
        if self.replay_size:
          self.replay_buffer.append(transition)
        else:
          self.replay_buffer = np.array([transition])
        
        return done

    def learn(self):

        batch = random.sample(self.replay_buffer, self.batch_size) if self.replay_size else self.replay_buffer
        batch = np.array(batch)
        y = self.update(batch)
        states = batch[:,:4]
        actions = batch[:,4]

        with tf.GradientTape() as tape:
          predict_Q = self.Q_net(states)
          select_Q = tf.math.reduce_sum(predict_Q * tf.one_hot(actions, self.n_actions), axis=1)
          loss = mse(y, select_Q)
        
        trainable_variables = self.Q_net.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.Q_net.optimizer.apply_gradients(zip(gradients, trainable_variables))

    def copy_weights(self):

        Q_vars = self.Q_net.trainable_variables
        target_Q_vars = self.target_Q.trainable_variables
        for Q_var, target_Q_var in zip(Q_vars, target_Q_vars):
          target_Q_var.assign(Q_var)

    def factor_decay(self):
       
        if self.temp >= self.min_decay: 
              self.temp *= self.decay
        if self.epsilon >= self.min_decay:
              self.epsilon *= self.decay

    def play(self):

        total_steps = 0
        steps_over_500 = 0
        for episode in tqdm(range(self.episodes)):
          reward = 0
          while True:
            done = self.sample_and_store()
            reward += 1
            total_steps += 1
            self.factor_decay()
            if len(self.replay_buffer) >= self.batch_size and total_steps >= self.steps_bf_training:
              self.learn()
            if done:
              break
          if episode % self.copy_steps == 0 and self.copy_steps:
            self.copy_weights()

          self.total_rewards.append(reward)
          if reward == 500:
            steps_over_500 += 1
          else:
            steps_over_500 = 0
          if steps_over_500 >= 5:
            break


def evaluate(DQN, times=10):
    
    rewards = []
    env = gym.make('CartPole-v1')
    s, _ = env.reset()
    for _ in range(times):
      reward = 0
      while True:
        reward += 1
        a = DQN.select_action(s, policy=DQN.policy, epsilon=DQN.epsilon, temp=DQN.temp)
        s, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        if done:
          s, info = env.reset()
          break
      rewards.append(reward)
    return rewards