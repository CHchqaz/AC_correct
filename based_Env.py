import tensorflow as tf
import numpy as np
import math

class envh:
    def __init__(self):
        self.a = 0.1  # 定义学习率
        self.loops_sum = 1000  # 循环次数
        self.gamma = 0.9  # 定义折扣因子
        self.Ts = 1  # 定义符号时间
        self.count = 0 # 训练次数
        self.e = 0.1;   self.W = 1000000;   self.BER = 0.001;   self.symbelTs = 1 / 1000000
        self.N0 = 8 * 10 ** -9;  self.deadline = 100
        self.Bmax = 80;    self.Emax = 10
        self.lamda1 = 2;    self.lamda2 = 3;    self.lamda3 = 5


    def reset(self):
        # initialize
        lamda1 = 18;   lamda2 = 14;    lamda3 = 16
        self.eh1 = np.random.poisson(lamda1)
        #print(self.eh1)
        self.eh2 = np.random.poisson(lamda2)
        #print(self.eh2)
        self.eh3 = np.random.poisson(lamda3)
        #print(self.eh3)
        self.B_1 = 0 + self.eh1
        self.B_2 = 0 + self.eh2
        self.B_3 = 0 + self.eh3

        #self.state = np.array([self.B_1, self.B_2, self.B_3])  # 当前状态
        self.state=np.array([26,10,10])
        return self.state

    def step(self,state1, a):
        # a = ([p1,p2,p3,p4,y1,y2,y3])
        B_1, B_2, B_3 = state1

        # Data flow on each link
        linkFlow = np.array([2, 1, 0.5, 2.125])       #4条链路的额定流量
        sigma = 0.1
        p_max = 11
        aq = 0.8

        lamda1 = 18;    lamda2 = 14;     lamda3 = 16

        p1 = a[0, 0];   p2 = a[0, 1];   p3 = a[0, 2];    p4 = a[0, 3]        #功率分配
        y1=a[0,4];      y2=a[0,5];

        done=True

        if  B_1==0 or B_2==0 or B_3==0:
            done=False

        # Calculate Power Bound
        # capLink = [0.5 * math.log(1+ (x / sigma)) for x in p_value]
        p_bound_ori = [sigma * (math.exp(2 * x) - 1) for x in linkFlow]
        p_bound = [x + 0.2 for x in p_bound_ori]           #x为额定功率

        # Calculate Reward
        p_value = a[:, [0, 1, 2, 3]].flatten()
        # self.action = tf.clip_by_value(self.multi_dist.sample(1), action_bound[0], action_bound[1])
        p_value[0] = np.clip(p_value[0], p_bound[0], p_max)
        p_value[1] = np.clip(p_value[1], p_bound[1], p_max)
        p_value[2] = np.clip(p_value[2], p_bound[2], p_max)
        p_value[3] = np.clip(p_value[3], p_bound[3], p_max)

        if p1 <= 5.3 or p2 <=0.63 or p3 <= 0.17 or p4<=6.9:
            reward=-1000
        else:
            capLink = [0.5 * math.log(1 + (x / sigma)) for x in p_value]       #capLink为实际链路流量
            reward = sum(linkFlow / (capLink - linkFlow))
        #reward = linkFlow / (capLink - linkFlow)                       #优化目标，使reward的总和值最小

        # if reward/250>0.2:
        #     done=False
        # Node 1
        self.eh1 = np.random.poisson(lamda1)
        #print('step中的eh1=', self.eh1)
        B_1 = max(min(B_1 + self.eh1 - (p1 + p2) * self.Ts+(y1+y2)*aq , self.Bmax),0)
        #print('step中的B1=', newB_1)
        # Node 2
        self.eh2 = np.random.poisson(lamda2)

        #print('step中的eh2', self.eh2)
        B_2 = max(min(B_2 + self.eh2 - p3 * self.Ts-y1 , self.Bmax),0)
        #print('step中的B2=', newB_2)
        # Node 3
        self.eh3 = np.random.poisson(lamda3)
        #print('step中的eh3', self.eh3)
        B_3 = max(min(B_3 + self.eh3 - p4 * self.Ts-y2 , self.Bmax),0)
        #print('step中的B3=', newB_3)

        # if  -reward<-30 or newB_1 == 0 or newB_2 == 0 or newB_3 == 0 or \
        #         p1 > 13 or p2 > 13 or p3 < 0 or p4 < 0 or y1 < 0 or y2 < 0:
        #     done=False
        # else:
        state2 = np.array([B_1, B_2, B_3])

        return state2, -reward, done



    def step1(self, state,a):
        B_1, B_2, B_3 = state
        # Data flow on each link
        linkFlow = np.array([2, 1, 0.5, 2.125])  # 4条链路的额定流量
        sigma = 0.1;p_max = 10
        lamda1 = 18;lamda2 = 14;lamda3 = 16

        p1 = a[0, 0];p2 = a[0, 1];p3 = a[0, 2];p4 = a[0, 3]  # 功率分配

        p_bound_ori = [sigma * (math.exp(2 * x) - 1) for x in linkFlow]
        p_bound = [x + 0.2 for x in p_bound_ori]  # x为额定功率

        # Calculate Reward
        p_value = a[:, [0, 1, 2, 3]].flatten()
        # self.action = tf.clip_by_value(self.multi_dist.sample(1), action_bound[0], action_bound[1])
        p_value[0] = np.clip(p_value[0], p_bound[0], p_max)
        p_value[1] = np.clip(p_value[1], p_bound[1], p_max)
        p_value[2] = np.clip(p_value[2], p_bound[2], p_max)
        p_value[3] = np.clip(p_value[3], p_bound[3], p_max)

        capLink = [0.5 * math.log(1 + (x / sigma)) for x in p_value]  # capLink为实际链路流量
        reward = sum(linkFlow / (capLink - linkFlow))


        tf.summary.scalar("reward", reward)

        # Node 1
        self.eh1 = np.random.poisson(lamda1)
        # print('step中的eh1=', self.eh1)
        newB_1 = max(min(B_1 + self.eh1 - (p1 + p2) * self.Ts , self.Bmax), 0)
        # print('step中的B1=', newB_1)
        # Node 2
        self.eh2 = np.random.poisson(lamda2)
        # print('step中的eh2', self.eh2)
        newB_2 = max(min(B_2 + self.eh2 - p3 * self.Ts , self.Bmax), 0)
        # print('step中的B2=', newB_2)
        # Node 3
        self.eh3 = np.random.poisson(lamda3)
        # print('step中的eh3', self.eh3)
        newB_3 = max(min(B_3 + self.eh3 - p4 * self.Ts , self.Bmax), 0)

        state1 = np.array([newB_1, newB_2, newB_3])

        return state1, -reward

