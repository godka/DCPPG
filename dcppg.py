import math
import numpy as np
import tensorflow.compat.v1 as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 64
ACTION_EPS = 1e-6
GAMMA = 0.99
# PPO2
EPS = 0.2

class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            merge_net = tflearn.fully_connected(inputs, FEATURE_NUM, activation='relu')
            net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            pi = tflearn.fully_connected(net, self.a_dim, activation='softmax')


            val_net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            phasic_val = tflearn.fully_connected(val_net, 1, activation='linear')

        with tf.variable_scope('critic'):
            merge_net = tflearn.fully_connected(inputs, FEATURE_NUM, activation='relu')
            net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            value = tflearn.fully_connected(net, self.a_dim, activation='linear')
        return pi, value, phasic_val
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def r(self, pi_new, pi_old, acts):
        return tf.reduce_sum(tf.multiply(pi_new, acts), reduction_indices=1, keepdims=True) / \
                tf.reduce_sum(tf.multiply(pi_old, acts), reduction_indices=1, keepdims=True)

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self._entropy = 0.01
        
        self.ppo_training_epo, self.aux_training_epo = 5, 6
        self.beta = 1.
        self.ppg_repeat = 32

        self.R = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        
        self.old_pi = tf.placeholder(tf.float32, [None, self.a_dim])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        self.pi, self.val, self.phasic_val = self.CreateNetwork(inputs=self.inputs)
        
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)
        self.entropy = -tf.reduce_sum(tf.multiply(self.real_out, tf.log(self.real_out)), reduction_indices=1, keepdims=True)
    
        self.adv = tf.stop_gradient(self.R - self.val)
        self.ppo2loss = tf.minimum(self.r(self.real_out, self.old_pi, self.acts) * self.adv, 
                            tf.clip_by_value(self.r(self.real_out, self.old_pi, self.acts), 1 - EPS, 1 + EPS) * self.adv
                        )
        self.dual_loss = tf.where(tf.less(self.adv, 0.), tf.maximum(self.ppo2loss, 3. * self.adv), self.ppo2loss)

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.network_params += \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.policy_loss = - tf.reduce_mean(self.dual_loss) - self._entropy * tf.reduce_mean(self.entropy)
        self.value_loss = tflearn.mean_square(self.val, self.R)

        self.policy_opt = tf.train.AdamOptimizer(self.lr_rate).minimize(self.policy_loss)
        self.val_opt = tf.train.AdamOptimizer(self.lr_rate).minimize(self.value_loss)

        self.kl_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(self.old_pi, tf.log(self.old_pi) - tf.log(self.real_out)), reduction_indices=1, keepdims=True))
        self.aux_loss = tflearn.mean_square(self.phasic_val, self.R) + self.kl_loss
        self.aux_opt = tf.train.AdamOptimizer(self.lr_rate).minimize(self.aux_loss)

        # ppg buffer
        self.s_batch_buffer, self.v_batch_buffer = [], []

    def predict(self, inputs):
        inputs = np.reshape(inputs, [-1, self.s_dim])
        action = self.sess.run(self.real_out, feed_dict={
            self.inputs: inputs
        })
        return action[0]

    def train_policy(self, s_batch, a_batch, p_batch, v_batch, epoch):
        self.sess.run(self.policy_opt, feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch,
            self.R: v_batch, 
            self.old_pi: p_batch
        })
        
    def train(self, s_batch, a_batch, p_batch, v_batch, epoch):
        for i in range(self.ppo_training_epo):
            self.train_policy(s_batch, a_batch, p_batch, v_batch, epoch)
            self.train_value(s_batch, v_batch)

        self.s_batch_buffer += s_batch
        self.v_batch_buffer += v_batch

        # ppg here
        if epoch % self.ppg_repeat == 0:
            # refresh old-pi
            p_batch_buffer = self.sess.run(self.real_out, feed_dict={
                self.inputs: self.s_batch_buffer})

            for i in range(self.aux_training_epo):
                self.train_aux(self.s_batch_buffer, p_batch_buffer, self.v_batch_buffer)
                self.train_value(self.s_batch_buffer, self.v_batch_buffer)
            
            # clear ppg buffer
            self.s_batch_buffer, self.v_batch_buffer = [], []

    def train_aux(self, s_batch, p_batch, v_batch):
        self.sess.run(self.aux_opt, feed_dict={
            self.inputs: s_batch,
            self.R: v_batch, 
            self.old_pi: p_batch
        })

    def train_value(self, s_batch, v_batch):
        self.sess.run(self.val_opt, feed_dict={
            self.inputs: s_batch,
            self.R: v_batch
        })

    def compute_v(self, s_batch, r_batch, terminal):
        ba_size = len(s_batch)
        R_batch = np.zeros([len(r_batch), 1])

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        return list(R_batch)
