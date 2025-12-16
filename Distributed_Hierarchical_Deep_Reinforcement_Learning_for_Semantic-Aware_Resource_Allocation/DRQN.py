# import numpy as np
# #to reproduce results in keras, the seed must be set before keras is imported
# #to do: find some way to make this a parameter
# #comment out for random
# #np.random.seed(1)
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.layers import LSTM
# from collections import deque
# from config import Config
# from tensorflow.python.keras.optimizers import rmsprop_v2
#
#
# class drqn:
#     def __init__(self, num_actions=Config().n_low_actions, num_observations=74,
#                  memory_step=8, memory_episodes=8, lr=0.0005, lr_decay=1e-4,minimum_epsilon=0.01,
#                  maximum_epsilon=0.6, epsilon_decay=0.9, target_copy_iterations=100,
#                  target_copy_start_steps=10, num_neurons=(64, 32), memory_size=500, future_discount=0.5,
#                  learning_rate_decay=1, learning_rate_decay_ep=500, activation_function='relu'):
#         self.num_actions = num_actions
#         self.num_observations = num_observations
#         self.memory_steps = memory_step                  # Number of Steps to Include in an Episode Sample
#         self.layer_size_1 = num_neurons[0]
#         self.layer_activation = activation_function
#         self.layer_size_2 = num_neurons[1]
#         self.lr = lr
#         self.lr_decay = lr_decay
#         self.memory_size = memory_size
#         self.learning_rate_decay_ep = learning_rate_decay_ep
#         self.memory_episodes = memory_episodes                # Number of Episodes to Include in a Memory Sample
#         self.minimum_epsilon = minimum_epsilon
#         self.target_copy_iterations = target_copy_iterations
#         self.target_copy_start_steps = target_copy_start_steps
#         self.future_discount = future_discount
#         self.epsilon_decay = epsilon_decay
#         self.learning_rate_decay = learning_rate_decay
#         self.maximum_epsilon = maximum_epsilon
#         self.save_path = 'model/'
#         self.optimizer = rmsprop_v2.RMSprop(lr=self.lr, decay=self.lr_decay)
#
#         #Create the model that will be trained
#         self.model = Sequential()
#         self.model.add(Dense(self.layer_size_1, input_shape=(self.memory_steps, self.num_observations),
#                              activation=self.layer_activation))
#         self.model.add(LSTM(200, activation=self.layer_activation,unroll=1))
#         self.model.add(Dense(self.layer_size_2, activation='linear'))
#         self.model.add(Dense(self.num_actions, activation='linear'))
#         self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
#
#         #Create the model that will calculate target values
#         self.model_target = Sequential()
#         self.model_target = Sequential.from_config(self.model.get_config())
#         self.model_target.set_weights(self.model.get_weights())
#         self.model_target.compile(loss='mean_squared_error', optimizer=self.optimizer)
#
#         #Since the LSTM has an internal state that is changed when predictions are made,
#         #we keep a separate copy of the training model in order to select actions. This
#         #is simpler than tracking the internal state and resetting the values
#         self.model_action = Sequential()
#         self.model_action = Sequential.from_config(self.model.get_config())
#         self.model_action.set_weights(self.model.get_weights())
#         self.model_action.compile(loss='mean_squared_error', optimizer=self.optimizer)
#
#         #Create the Replay Memory
#         self.replay_memory = deque(maxlen=self.memory_size)
#         self.replay_current = []
#
#         self.current_epsilon = self.maximum_epsilon
#         self.current_learning_rate = self.lr
#
#         self.train_iterations = 0
#         self.first_iterations = 0
#         self.current_episode = 0
#         self.current_step = 0
#
#         self.average_train_rewards = deque(maxlen=100)
#         self.average_test_rewards = deque(maxlen=100)
#
#         # self.train_path = self.save_path + ".train"
#         # self.train_file = open(self.train_path, 'w')
#         # self.train_file.write('episode reward average_reward\n')
#         #
#         # self.test_path = self.save_path + ".test"
#         # self.test_file = open(self.test_path, 'w')
#         # self.test_file.write('episode reward\n')
#
#         #the lstm layer requires inputs of timesteps
#         #a history of of past observations are kept to pass into the lstm
#         self.lstm_obs_history = deque([[0 for y in range(self.num_observations)] for x in range(self.memory_steps)], self.memory_steps)
#
#
#     # def __del__(self):
#     #     self.train_file.close()
#     #     self.test_file.close()
#     #     pass
#
#     def clear_history(self):
#         self.lstm_obs_history = deque([[0 for y in range(self.num_observations)] for x in range(self.memory_steps)], self.memory_steps)
#
#     def get_random_action(self):
#         return np.random.randint(0, self.num_actions)
#
#     def choose_action(self, observation):
#         #get the current action for the model or a random action depending on arguments
#         #and the current episode
#         self.lstm_obs_history.append(observation)
#         if np.random.random() < self.current_epsilon:
#             #if training, choose random actions
#             return np.random.randint(0, self.num_actions)
#         else:
#             #choose action from model
#             q_values = self.model_action.predict(np.array(self.lstm_obs_history).reshape(1, self.memory_steps,
#                                                                                          self.num_observations))
#             action = np.argmax(q_values)
#             return action
#
#     def save_transition(self, state, action, reward, next_state):
#         #add a transaction to replay memory, should be called after performing
#         #an action and getting an observation
#         self.replay_current.append((state, action, reward, next_state))
#         self.current_step += 1
#         #make end of episode checks
#         self.replay_memory.append(self.replay_current)
#         self.end_of_episode()
#
#     def end_of_episode(self):
#         self.current_episode += 1
#         self.current_step = 0
#         self.clear_history()
#         self.replay_current = []
#         self.update_action_network()
#         self.model_action.reset_states()
#         if self.current_epsilon > self.minimum_epsilon:
#             self.decay_epsilon()
#         else:
#             self.current_epsilon = self.minimum_epsilon
#         if self.current_episode % self.learning_rate_decay_ep == 0:
#             self.decay_learning_rate()
#
#     def sample_memory(self, batch_size, trace_length):
#         # samples the replay memory returning a batch_size of random transactions
#         sampled_episodes = []
#         while True:
#             rand_ep = np.random.randint(0, len(self.replay_memory))
#             sampled_episodes.append(rand_ep)
#             if len(sampled_episodes) == batch_size:
#                 break
#         sampled_traces = []
#         for ep in sampled_episodes:
#             episode = self.replay_memory[ep]
#             start_step = np.random.randint(0, max(1, len(episode) - trace_length + 1))
#             current_trace = episode[start_step:start_step + trace_length]
#             action = current_trace[-1][1]
#             reward = current_trace[-1][2]
#             states = []
#             next_states = []
#             for step, transaction in enumerate(current_trace):
#                 states.append(transaction[0])
#                 next_states.append(transaction[3])
#             if len(current_trace) < trace_length:
#                 empty = [0 for x in states[0]]
#                 for i in range(trace_length - len(current_trace)):
#                     states.insert(0, empty)
#                     next_states.insert(0, empty)
#             sampled_traces.append([states, action, reward, next_states])
#         return sampled_traces
#
#     def learn(self):
#         if len(self.replay_memory) < self.memory_episodes:
#             print('Not enough transactions in replay memory to train.')
#             return
#         if self.train_iterations >= self.target_copy_iterations:
#             self.update_target_network()
#         if self.first_iterations < self.target_copy_start_steps:
#             # update the target network a few times on episode 0 so
#             # the model isn't training toward a completely random network
#             self.update_target_network()
#             self.first_iterations += 1
#
#         self.model.reset_states()
#         self.model_target.reset_states()
#
#         samples = self.sample_memory(self.memory_episodes, self.memory_steps)
#         observations = next_observations = rewards = np.array([])
#         actions = np.array([], dtype=int)
#         for transaction in samples:
#             observations = np.append(observations, transaction[0])
#             actions = np.append(actions, transaction[1])
#             next_observations = np.append(next_observations, transaction[3])
#             rewards = np.append(rewards, transaction[2])
#         observations = observations.reshape(self.memory_episodes, self.memory_steps, self.num_observations)
#         next_observations = next_observations.reshape(self.memory_episodes, self.memory_steps,  self.num_observations)
#         targets = updates = None
#         if self.target_copy_iterations == 0:
#             #this instance is not using a target copy network, use original model
#             targets = self.model.predict(observations)
#             updates = rewards + self.future_discount * self.model.predict(next_observations).max(axis=1)
#         else:
#             #this instance uses a target copy network
#             targets = self.model_target.predict(observations)
#             updates = rewards + self.future_discount * self.model_target.predict(next_observations).max(axis=1)
#         for i, action in enumerate(actions):
#             targets[i][action] = updates[i]
#         self.model.fit(observations, targets, batch_size=self.memory_episodes, verbose=0)
#
#         self.train_iterations += 1
#
#     def update_target_network(self):
#         self.model_target.set_weights(self.model.get_weights())
#
#     def update_action_network(self):
#         self.model_action.set_weights(self.model.get_weights())
#
#     def decay_epsilon(self):
#         self.current_epsilon *= self.epsilon_decay
#
#     def decay_learning_rate(self):
#         self.current_learning_rate *= self.learning_rate_decay
#
#     # def write_training_episode(self, episode, reward):
#     #     self.average_train_rewards.append(reward)
#     #     self.train_file.write(str(episode) + ' ')
#     #     self.train_file.write(str(reward) + ' ')
#     #     if len(self.average_train_rewards) >= 100:
#     #         self.train_file.write(str(np.mean(self.average_train_rewards)))
#     #     self.train_file.write('\n')
#     #
#     # def write_testing_episode(self, episode, reward):
#     #     self.average_test_rewards.append(reward)
#     #     self.test_file.write(str(episode) + ' ')
#     #     self.test_file.write(str(reward) + ' ')
#     #     self.test_file.write('\n')
#
#     def save_model(self, file_name):
#         file_path = self.save_path + file_name + '.hdf5'
#         self.model.save(file_path, True)

""" DQN agent at each base station """

import numpy as np
import random
from neural_network import NeuralNetwork, NeuralNetwork_low
from collections import deque
import tensorflow as tf  # ★ 新增：后面用 GradientTape 和 model(...)
from tensorflow.python.keras.optimizers import rmsprop_v2
import os
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python import keras as keras_py  # ★ 新增：拿到内部 keras 模块

# ★ 兼容补丁：为 tensorflow.python.keras 补一个 __version__，防止保存时 ImportError
if not hasattr(keras_py, "__version__"):
    keras_py.__version__ = tf.__version__



class DQN_low:
    # hyper params
    def __init__(self,
                 n_actions=NeuralNetwork().output_ports,
                 n_features=NeuralNetwork().input_ports,
                 lr=5e-4,
                 lr_decay=1e-4,
                 reward_decay=0.5,
                 e_greedy=0.6,
                 epsilon_min=1e-2,
                 replace_target_iter=100,
                 memory_size=500,
                 batch_size=8,
                 e_greedy_decay=1e-4):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = reward_decay
        # epsilon-greedy params
        self.epsilon = e_greedy
        self.epsilon_decay = e_greedy_decay
        self.epsilon_min = epsilon_min
        self.save_path = 'model/'

        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.loss = []
        self.accuracy = []

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.lstm_obs_history = deque([[0 for y in range(self.n_features)] for x in range(8)], 8)
        self._built_net()
        self.model = Sequential()
    #
    # def clear_history(self):
    #     self.lstm_obs_history = deque([[0 for y in range(self.n_features)] for x in range(8)], 8)
    # #
    # #
    def load_mod(self, indx):
        self.model = load_model('model/DQN_common_PL=5BS_{}.hdf5'.format(indx))
    #
    # def choose_action(self, observation):
    #     # epsilon greedy
    #     observation = observation[np.newaxis, :]
    #     actions_value = self.model.predict(observation)
    #     action = np.argmax(actions_value)
    #     return action

    # def choose_action(self, observation):
    #     self.lstm_obs_history.append(observation)
    #     #choose action from model
    #     q_values = self.model.predict(np.array(self.lstm_obs_history).reshape(1, 8, self.n_features))
    #     action = np.argmax(q_values)
    #     self.clear_history()
    #     return action

    def _built_net(self):

        tar_nn = NeuralNetwork_low()
        eval_nn = NeuralNetwork_low()
        self.model1 = tar_nn.get_model(1)
        self.model2 = eval_nn.get_model(1)
        self.target_replace_op()
        # RMSProp optimizer
        # ★ 把优化器挂到 self 上，learn() 里 GradientTape 会用到
        self.optimizer = rmsprop_v2.RMSprop(lr=self.lr, decay=self.lr_decay)

        # compile 保留，不再使用 model2.fit()，但不会有副作用
        self.model2.compile(loss='mse', optimizer=self.optimizer)


    def _store_transition_(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, r, s_))
        # print(transition, '\n')
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = np.array(transition)
        self.memory_counter += 1

    def save_transition(self, s, a, r, s_):
        self._store_transition_(s, a, r, s_)

    # def choose_action(self, observation):
    #     # epsilon greedy
    #     if random.uniform(0, 1) > self.epsilon:
    #         observation = observation[np.newaxis, :]
    #         actions_value = self.model2(observation, training=False)
    #         action = np.argmax(actions_value)
    #     else:
    #         action = random.randint(0, self.n_actions - 1)
    #     return action

    def choose_action(self, observation):
        # epsilon greedy
        if random.uniform(0, 1) > self.epsilon:
            observation = observation[np.newaxis, :].astype(np.float32)
            q_values = self.model2(observation, training=False)
            q_values = q_values.numpy()
            action = int(np.argmax(q_values))
        else:
            action = random.randint(0, self.n_actions - 1)
        return action


    def save_model(self, file_name):
        file_path = self.save_path + file_name + '.hdf5'
        self.model2.save(file_path, True)

    def target_replace_op(self):
        temp = self.model2.get_weights()
        print('Parameters updated')
        self.model1.set_weights(temp)

    def learn(self):
        # update target network's params
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()

        # sample mini-batch from experience replay
        if self.memory_counter > self.memory_size:
            sample_index = random.sample(list(range(self.memory_size)), self.batch_size)
        else:
            sample_index = random.sample(list(range(self.memory_counter)), self.batch_size)

        # mini-batch data
        batch_memory = self.memory[sample_index, :]

        # 拆分出 s, a, r, s'
        states = batch_memory[:, :self.n_features]
        actions = batch_memory[:, self.n_features].astype(np.int32)
        rewards = batch_memory[:, self.n_features + 1]
        next_states = batch_memory[:, -self.n_features:]

        # 转成 Tensor
        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)

        # 1) target 网络：max_a' Q_target(s', a')
        q_next = self.model1(next_states_tf, training=False)    # [batch, n_actions]
        q_next_max = tf.reduce_max(q_next, axis=1)              # [batch]

        # 2) eval 网络：Q_eval(s, ·)
        q_eval = self.model2(states_tf, training=False)         # [batch, n_actions]

        # 3) 构造 Q_target：只更新对应动作的那一列
        q_target = tf.identity(q_eval)                          # [batch, n_actions]
        batch_index = tf.range(self.batch_size, dtype=tf.int32) # [0,...,batch-1]
        indices = tf.stack([batch_index, actions_tf], axis=1)   # [batch, 2]
        updated_q = rewards_tf + self.gamma * q_next_max        # [batch]
        q_target = tf.tensor_scatter_nd_update(q_target, indices, updated_q)

        # 4) 用 MSE(Q_target, Q_eval) 反向传播更新 eval 网络参数
        with tf.GradientTape() as tape:
            q_eval_pred = self.model2(states_tf, training=True)
            loss = tf.reduce_mean(tf.square(q_target - q_eval_pred))

        grads = tape.gradient(loss, self.model2.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model2.trainable_variables))

        # 记录 loss，更新 epsilon 与步数
        self.loss.append(float(loss.numpy()))
        self.epsilon = max(self.epsilon / (1 + self.epsilon_decay), self.epsilon_min)
        self.learn_step_counter += 1


