""" DQN agent at each base station """

import numpy as np
import random
from neural_network import NeuralNetwork
from collections import deque
import tensorflow as tf
from tensorflow.python.keras.optimizers import rmsprop_v2
import os
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQN:
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
    def clear_history(self):
        self.lstm_obs_history = deque([[0 for y in range(self.n_features)] for x in range(8)], 8)
    #
    #
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

        tar_nn = NeuralNetwork()
        eval_nn = NeuralNetwork()
        self.model1 = tar_nn.get_model(1)
        self.model2 = eval_nn.get_model(1)
        self.target_replace_op()
        # RMSProp optimizer
        # ★ 把优化器存成成员变量，后面 GradientTape 会用到
        self.optimizer = rmsprop_v2.RMSprop(lr=self.lr, decay=self.lr_decay)

        # 虽然后面我们不用 model.fit()，但 compile 保留不影响，
        # 以防 NeuralNetwork 里有地方依赖 model 的 compile 状态
        self.model2.compile(loss='mse', optimizer=self.optimizer)   # 将字符串编译为字节代码


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

    def choose_action(self, observation):
        # epsilon greedy
        if random.uniform(0, 1) > self.epsilon:
            # 增加 batch 维度，并确保是 float32
            observation = observation[np.newaxis, :].astype(np.float32)
            # ★ 直接前向推理，不用 predict()，避免走 DataAdapter
            q_values = self.model2(observation, training=False)
            # q_values 是 Tensor，先转成 numpy 再 argmax（也可以直接在 Tensor 上 argmax）
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

        # 计算 Q_target 所需的 max_a' Q_target(s', a')
        # 注意这里用的是 target 网络 model1
        q_next = self.model1(next_states_tf, training=False)  # [batch, n_actions]
        q_next_max = tf.reduce_max(q_next, axis=1)            # [batch]

        # 用当前 eval 网络先算一遍 Q_eval(s, ·)
        q_eval_current = self.model2(states_tf, training=False)  # [batch, n_actions]

        # 构造 Q_target：只更新对应动作的那一列，其它动作保持原 q_eval 值
        q_target_full = tf.identity(q_eval_current)  # 复制一份

        batch_index = tf.range(self.batch_size, dtype=tf.int32)   # [0,1,...,batch_size-1]
        indices = tf.stack([batch_index, actions_tf], axis=1)     # [batch, 2]

        updated_q = rewards_tf + self.gamma * q_next_max         # [batch]
        q_target_full = tf.tensor_scatter_nd_update(q_target_full, indices, updated_q)

        # 使用 GradientTape 对 model2 的参数做一次 MSE 回归
        with tf.GradientTape() as tape:
            q_eval_pred = self.model2(states_tf, training=True)
            loss = tf.reduce_mean(tf.square(q_target_full - q_eval_pred))

        grads = tape.gradient(loss, self.model2.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model2.trainable_variables))

        # 记录 loss，更新 epsilon 和计数器
        self.loss.append(float(loss.numpy()))
        self.epsilon = max(self.epsilon / (1 + self.epsilon_decay), self.epsilon_min)
        self.learn_step_counter += 1

