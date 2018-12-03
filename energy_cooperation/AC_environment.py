import math
import numpy as np
dm=200
Gb=18;Gu=0
Nj=1
B=10**9
λ=10
σ = -174 + 10 * math.log10(B) + 7
Bmax=80

class enva(object):
    def __init__(self):
        self.Dtotal1 = 0;self.Dtotal2=0
        self.D1total=0;self.D2total=0
        self.a=math.exp(-200/30+5.2);self.b=(1-self.a)*math.exp(-200/67.1);self.c=1-self.a-self.b
        # self.d1 = np.random.choice([1, 2, 0], p=[self.b, self.c, self.a])  # 路径损耗的选择
        # self.d2 = np.random.choice([1, 2, 0], p=[self.b, self.c, self.a])
        self.d1=1;self.d2=1
    def reset(self):
        B1state = 80;B2state = 80
        E1harv = np.random.uniform(0, 40)
        E2harv = np.random.uniform(0, 40)
        B1state = min(B1state + E1harv , Bmax)
        B2state = min(B2state + E2harv, Bmax)
        return np.array([B1state,B2state])

    def step(self,s,a):
        done=True
        B1state_first,B2state_first=s
        P1=a[0,0];P2=a[0,1];Etra1=a[0,2];Etra2=a[0,3]
        print('#################')
        print('E1state: ',B1state_first)
        print('E2state: ', B2state_first)



        # 基站1每时隙服务信息的队列状态
        #D1 = np.random.poisson(λ)
        D1=18
        print('D1: ', D1)
        self.D1total=self.D1total+D1
        self.Dtotal1=self.Dtotal1+D1

        # 基站2每时隙服务信息的队列状态
        #D2 = np.random.poisson(λ)
        D2=18
        print('D2: ', D2)
        self.D2total=self.D2total+D2
        self.Dtotal2 = self.Dtotal2+ D2

        #基站1电池状态
        B1state=min(B1state_first+Etra2-Etra1,Bmax)

        # 基站2电池状态
        B2state = min(B2state_first-Etra2+Etra1, Bmax)


        print('P1: ',P1)
        print('P2: ',P2)
        print('Etra1: ',Etra1)
        print('Etra2: ',Etra2)

        #基站1，2每时隙需要发送的下行链路信息，P1，P2为基站1，2的下行功率

        if self.d1!=0:
            if self.d1==1:
                α = 61.4;η = 2;ξ = np.random.normal(loc=0.0, scale=5.8, size=None)
                Lij = α + 10 * η * math.log10(dm) + ξ
                Ri1 = math.log2(1 + 10 ** ((P1 + Lij + Gb + Gu + σ * 2) / 10)) / Nj
            else:
                α =72.0;η =2.92;ξ = np.random.normal(loc=0.0, scale=8.7, size=None)
                Lij = α + 10 * η * math.log10(dm) + ξ
                Ri1 = math.log2(1 + 10 ** ((P1 + Lij + Gb + Gu + σ * 2) / 10)) / Nj
        else:
            Ri1=0

        if self.d2!=0:
            if self.d2==1:
                α = 61.4;η = 2;ξ = np.random.normal(loc=0.0, scale=5.8, size=None)
                Lij = α + 10 * η * math.log10(dm) + ξ
                Ri2 = math.log2(1 + 10 ** ((P2 + Lij + Gb + Gu + σ * 2) / 10)) / Nj
            else:
                α =72.0;η =2.92;ξ = np.random.normal(loc=0.0, scale=8.7, size=None)
                Lij = α + 10 * η * math.log10(dm) + ξ
                Ri2 = math.log2(1 + 10 ** ((P2 + Lij + Gb + Gu + σ * 2) / 10)) / Nj
        else:
            Ri2=0

        print('Ri1: ',Ri1)
        print('Ri2: ',Ri2)


    ########更新后的信息#########

        #基站1,2 服务信息的队列状态
        self.Dtotal1=max(self.Dtotal1-Ri1,0)
        self.Dtotal2=max(self.Dtotal2-Ri2,0)


        #更新的基站1，2的电池信息
        B1state_=max(B1state-P1,0)
        B2state_=max(B2state-P2,0)

        #奖励
        # 首先判断P1,P2的功率是否超出范围

        #reward=max(self.Dtotal1/self.D1total+self.Dtotal2/self.D2total,0)*(B1state/(B1state+P1+Etra1)+B2state/(B2state+P2+Etra2))
        if B1state_first-P1-Etra1>0 and B1state_first-P1-Etra2>0:
            reward =Ri1+Ri2
        else:
            reward = -(Ri1 + Ri2)
        o=self.Dtotal1+self.Dtotal2

        B1state_ = min(B1state_ + np.random.uniform(0, 40) , Bmax)
        B2state_ = min(B2state_ + np.random.uniform(0, 40) , Bmax)
        s_ = np.array([B1state_, B2state_])
        return s_,reward,done,o











