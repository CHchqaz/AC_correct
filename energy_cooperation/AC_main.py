import tensorflow as tf
import pandas as pd
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
import matplotlib.pyplot as plt
from AC_environment import enva
np.random.seed(2)
tf.set_random_seed(2)  # reproducible

tf.reset_default_graph()


class Actor(object):
    def __init__(self, sess, n_features, action_bound, lr=0.0001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        l1 = tf.layers.dense(
            inputs=self.s,
            units=80,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='l1'
        )

        mu1 = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases default = 0.1
            name='mu1'
        )

        mu2 = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu2'
        )

        mu3 = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu3'
        )

        mu4 = tf.layers.dense(
            inputs=l1,
            units=1,  # number of hidden units
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(0.1),  # biases
            name='mu4'
        )

        sigma1 = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.relu,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma1'
        )

        sigma2 = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.relu,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma2'
        )

        sigma3 = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.relu,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma3'
        )

        sigma4 = tf.layers.dense(
            inputs=l1,
            units=1,  # output units
            activation=tf.nn.relu,  # get action probabilities
            kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
            bias_initializer=tf.constant_initializer(1.),  # biases
            name='sigma4'
        )



        global_step = tf.Variable(0, trainable=False)

        bias_mu1 = tf.constant(0.5)
        mu1 = tf.add(mu1, bias_mu1)

        bias_mu2 = tf.constant(0.5)
        mu2 = tf.add(mu2, bias_mu2)

        bias_mu3 = tf.constant(0.2)
        mu3 = tf.add(mu3, bias_mu3)

        bias_mu4 = tf.constant(0.2)
        mu4 = tf.add(mu4, bias_mu4)

        # self.e = epsilon = tf.train.exponential_decay(2., global_step, 1000, 0.9)
        # self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
        self.mu1, self.sigma1 = tf.squeeze(mu1), tf.squeeze(sigma1 + 0.5)
        self.mu2, self.sigma2 = tf.squeeze(mu2), tf.squeeze(sigma2 + 0.5)
        self.mu3, self.sigma3 = tf.squeeze(mu3), tf.squeeze(sigma3 + 0.3)
        self.mu4, self.sigma4 = tf.squeeze(mu4), tf.squeeze(sigma4 + 0.3)


        tfd = tf.contrib.distributions  #统计分布TF.contrib.ditributions模块，Bernoulli、Beta、Binomial、Gamma、Ecponential、Normal、Poisson、Uniform等统计分布
        #tfd=tfp.distributions
        self.multi_dist = tfd.MultivariateNormalDiag(
            loc=[self.mu1, self.mu2, self.mu3, self.mu4],
            scale_diag=[self.sigma1, self.sigma2, self.sigma3, self.sigma4]) #多元正态分布
        # self.multi_dist = tfd.Uniform(
        #          low=[self.mu1, self.mu2, self.mu3, self.mu4],
        #          high=[self.sigma1, self.sigma2, self.sigma3, self.sigma4])
        print('self.multi_dist:',self.multi_dist.sample(1)[0:2])
        self.action = tf.clip_by_value(self.multi_dist.sample(1), action_bound[0], action_bound[1])
        #self.action=self.multi_dist.sample(1)
        print(self.action)
        with tf.name_scope('exp_v'):
            #            log_prob = self.normal_dist.log_prob(self.a)  # loss without advantage
            log_prob = self.multi_dist.log_prob(self.a)  # loss without advantage
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
            # Add cross entropy cost to encourage exploration
            #            self.exp_v += 0.01*self.normal_dist.entropy()
            self.exp_v += 0.01 * self.multi_dist.entropy()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v, global_step)  # min(v) = max(-v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        #        tf.summary.histogram("learn_exp_v",exp_v)
        #        print("---------------learn_exp_v----------------------")
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})  # get probabilities for all actions


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            self.v_ = tf.placeholder(tf.float32, [1, 1], name="v_next")
            self.r = tf.placeholder(tf.float32, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA * self.v_ - self.v)
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.s: s, self.v_: v_, self.r: r})
        return td_error


def test():
    s = env.reset()
    reward_list = []
    for i in range(100):
        a = actor.choose_action(s/100)*40
        s_, r,done,q = env.step(s,a)
        s = s_
        reward_list.append(q)

    reward_list1=[reward_list[n]-reward_list[0] for n in range(len(reward_list))]
    print(reward_list1)
    reward_list2=[reward_list1[n]-reward_list1[0] for n in range(len(reward_list1))]
    print(reward_list2)



OUTPUT_GRAPH = False
MAX_EP_STEPS = 2000# default value = 200
GAMMA = 0.9
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

avg_reward = np.zeros([MAX_EP_STEPS, ])
env = enva()
N_S = 2
A_BOUND = 1
sess = tf.Session()
actor = Actor(sess, n_features=N_S, lr=LR_A, action_bound=[0, A_BOUND])
critic = Critic(sess, n_features=N_S, lr=LR_C)

AC_writer = tf.summary.FileWriter("logs", sess.graph)
merge = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())


for i in range(MAX_EP_STEPS):
    s = env.reset()
    print('输入的当前新状态',s)
    a = actor.choose_action(s/100)*40
    s_, r, done,o = env.step(s,a)
    h=100
    while done:
        h=h-1
        td_error = critic.learn(s/100, r, s_/100)  # gradient = grad[r + gamma * V(s_) - V(s)]
        print('r:',r)
        print('td_error:',td_error)
        actor.learn(s/100, a/40, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
        s = s_
        a = actor.choose_action(s/100)*40
        s_, r, done,o = env.step(s,a)
        if h==0:
            done=False


test()




