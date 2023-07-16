from Cartpole import Q_network, CartPoleDQN, evaluate
from Helper import LearningCurvePlot, smooth
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--net", help="Tune the network hyperparameters",
                    action="store_true")
parser.add_argument("--lr", help="Tune the learning rate",
                    action="store_true")
parser.add_argument("--exp", help="Tune the exploration factors",
                    action="store_true")
parser.add_argument("--gamma", help="Tune the discounted factor gamma",
                    action="store_true")
parser.add_argument("--copy", help="Tune the copy steps",
                    action="store_true")
parser.add_argument("--eva", help="Train the DQN model and evaluate",
                    action="store_true")
args = parser.parse_args()

smoothing_window = 11

#### Network Architecture ####
if args.net:

    ns = [10,32]
    lrs = [1e-2,1e-3,1e-4]
    
    print("Tuning network hyperparameters.")

    reward = []
    Plot = LearningCurvePlot(title = 'Network Architecture')    

    for n in ns:
        cartpole = CartPoleDQN( n_actions=2, num_layers=2, num_units=n, learning_rate=1e-2, decay=0.9999, epsilon=0.99, temp=0.99, \
                        min_decay = 0.05, gamma=0.95, batch_size=128, episodes=200, copy_steps=1, replay_size=10000, steps_bf_training=500,policy='softmax')
        cartpole.play()
        reward.append(cartpole.total_rewards)
        learning_curve = smooth(cartpole.total_rewards,smoothing_window) # additional smoothing
        Plot.add_curve(learning_curve,label=r'{} layers, each {} neurons'.format(2,n))

        cartpole = CartPoleDQN( n_actions=2, num_layers=1, num_units=10, learning_rate=1e-2, decay=0.9999, epsilon=0.99, temp=0.99, \
                        min_decay = 0.05, gamma=0.95, batch_size=128, episodes=200, copy_steps=1, replay_size=10000, steps_bf_training=500,policy='softmax')
        cartpole.play()
        reward.append(cartpole.total_rewards)
        learning_curve = smooth(cartpole.total_rewards,smoothing_window) # additional smoothing
        Plot.add_curve(learning_curve,label=r'{} layers, each {} neurons'.format(1,20))
    
    Plot.save('Network Architecture.png')



#### Learning Rate ####
if args.lr:

    lrs = [1e-2,1e-3,1e-4]

    print("Tune the learning rate.")

    Plot = LearningCurvePlot(title = 'Learning Rate')    
    reward = []
    for lr in lrs:
        cartpole = CartPoleDQN( n_actions=2, num_layers=2, num_units=10, learning_rate=lr, decay=0.9999, epsilon=0.99, temp=0.99, \
                        min_decay = 0.05, gamma=0.95, batch_size=128, episodes=200, copy_steps=1, replay_size=10000, steps_bf_training=500,policy='softmax')
        cartpole.play()
        reward.append(cartpole.total_rewards)
        learning_curve = smooth(cartpole.total_rewards,smoothing_window) # additional smoothing
        Plot.add_curve(learning_curve,label=r'learning rate = {}'.format(lr))


    Plot.save('Learning Rate.png')


#### Exploration ####
if args.exp:

    print("Tune the exploration factors.")

    decays = [0.999,0.9999]
    smoothing_window = 11
    reward = []
    Plot = LearningCurvePlot(title = 'Exploration Policy')    
    for decay in decays:
        cartpole = CartPoleDQN( n_actions=2, num_layers=2, num_units=10, learning_rate=1e-2 ,decay=decay, epsilon=0.99, temp=0.99, \
                        min_decay = 0.05, gamma=0.95, batch_size=128, episodes=200, copy_steps=1, replay_size=10000, steps_bf_training=500,policy='softmax')
        cartpole.play()
        reward.append(cartpole.total_rewards)
        learning_curve = smooth(cartpole.total_rewards,smoothing_window) # additional smoothing
        Plot.add_curve(learning_curve,label=r'softmax, exploration decay = {}'.format(decay))

    for decay in decays:
        cartpole = CartPoleDQN( n_actions=2, num_layers=2, num_units=10, learning_rate=1e-2 ,decay=decay, epsilon=0.99, temp=0.99, \
                        min_decay = 0.05, gamma=0.95, batch_size=128, episodes=200, copy_steps=1, replay_size=10000, steps_bf_training=500,policy='egreedy')
        cartpole.play()
        reward.append(cartpole.total_rewards)
        learning_curve = smooth(cartpole.total_rewards,smoothing_window) # additional smoothing
        Plot.add_curve(learning_curve,label=r'egreedy, exploration decay = {}'.format(decay))

    Plot.save('Exploration Policy.png')


#### Gamma ####
if args.gamma:

    print("Tune the gamma.")

    gammas = [0.9,0.95,0.99]
    Plot = LearningCurvePlot(title = 'Different value of gamma')    

    for gamma in gammas:
        cartpole = CartPoleDQN(n_actions=2, num_layers=2, num_units=10, learning_rate=1e-2, decay=0.9999, epsilon=0.99, temp=0.5, \
                        gamma=gamma, batch_size=128, episodes=200, copy_steps=1, replay_size=10000,policy="softmax")
        cartpole.play()
        smoothing_window = 41
        learning_curve = smooth(cartpole.total_rewards,smoothing_window) # additional smoothing
        Plot.add_curve(learning_curve,label=r'Different value of gamma = {}'.format(gamma))
    
    Plot.save('gammas.png')


#### Copy Steps ####
if args.copy:

    print("Tune the copy steps.")

    copy_steps = [1,2,5,10]
    Plot = LearningCurvePlot(title = 'Different value of copy_steps')    

    for step in copy_steps:
        cartpole = CartPoleDQN(n_actions=2, num_layers=2, num_units=10, learning_rate=1e-2, decay=0.9999, epsilon=0.99, temp=0.5, \
                        gamma=0.95, batch_size=128, episodes=200, copy_steps=step, replay_size=10000,policy="softmax")
        cartpole.play()
        smoothing_window = 41
        learning_curve = smooth(cartpole.total_rewards,smoothing_window) # additional smoothing
        Plot.add_curve(learning_curve,label=r'Different value of copy_steps = {}'.format(step))
    Plot.save('copy_steps.png')


#### Training and Evaluation ####
if args.eva:

    print("Train the DQN and evaluate.")

    cartpole = CartPoleDQN(n_actions=2, num_layers=1, num_units=20, learning_rate=1e-2, decay=0.999, policy = 'softmax', epsilon=0.99, temp=0.99, \
                    min_decay = 0.05, gamma=0.95, batch_size=128, episodes=500, copy_steps=1, replay_size=10000, steps_bf_training=500)
    cartpole.play()

    times = 100
    rewards = evaluate(cartpole, times)

    x= cartpole.total_rewards
    plt.plot(x)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training of DQN after hyperparameter tuning")
    plt.savefig("training_dqn.png")

    print("Mean reward: {}, median: {}, >=475 percentage: {}.".format(np.mean(rewards), np.median(rewards), np.mean(rewards>=475)*100))

    plt.hist(rewards)
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.title("Histogram of rewards")
    plt.savefig("hist_rewards.png")

print("Finish!")