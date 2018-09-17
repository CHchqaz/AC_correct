import numpy as np
import math
import pandas as pd
from scipy.special import lambertw
import matplotlib.pyplot as plt
linkFlow = np.array([2, 1, 0.5, 2.125])
sigma = 0.1
aq=0.8
o=0.1

def initial_algorithmλ(state):
    Eh1,Eh2,Eh3=state
    λ=0.001
    λ1 = Eh1 - sigma * (math.exp(
        2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 * λ * sigma))).real) + linkFlow[
            0]) - 1 + math.exp(
        2 * (lambertw(math.sqrt(linkFlow[1] * math.exp(-2 * linkFlow[1]) / (2 * λ * sigma))).real) + linkFlow[1]) - 1)
    while λ1<10**(-3):
        λ = λ + 0.00001
        λ1 = Eh1 - sigma * (math.exp(
            2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 * λ * sigma))).real) + linkFlow[0]) - 1 + math.exp(
            2 * (lambertw(math.sqrt(linkFlow[1] * math.exp(-2 * linkFlow[1]) / (2 * λ * sigma))).real) + linkFlow[1]) - 1)
    #print(λ1)
    λ2 = linkFlow[2] / (2 * sigma) * (0.5 * math.log(1 + (Eh2 / sigma)) - linkFlow[2]) ** (-2) * (
                1 + (Eh2 / sigma)) ** (-1)
    λ3 = linkFlow[3] / (2 * sigma) * (0.5 * math.log(1 + (Eh3 / sigma)) - linkFlow[3]) ** (-2) * (
                1 + (Eh3 / sigma)) ** (-1)
    return np.array([λ1,λ2,λ3])

def optimal_algorithmλy1(state):
    Eh1,Eh2,Eh3=state
    λ_=0.001
    λ_1 = aq*Eh1+Eh3 - sigma * ((math.exp(
        2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 * aq*λ_ * sigma))).real) + linkFlow[
            0]) - 1 + math.exp(2 * (lambertw(math.sqrt(linkFlow[1] * math.exp(-2 * linkFlow[1]) / (2 * aq*λ_ * sigma))).real) + linkFlow[1]) - 1)*aq
                                +math.exp(2 * (lambertw(math.sqrt(linkFlow[3] * math.exp(-2 * linkFlow[3]) / (2 * λ_ * sigma))).real) + linkFlow[
            3]) - 1)
    while λ_1 < 10 ** (-3):
        λ_ = λ_ + 0.00001
        λ_1 = aq*Eh1+Eh3 - sigma * ((math.exp(
        2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 * aq*λ_ * sigma))).real) + linkFlow[
            0]) - 1 + math.exp(2 * (lambertw(math.sqrt(linkFlow[1] * math.exp(-2 * linkFlow[1]) / (2 * aq*λ_ * sigma))).real) + linkFlow[1]) - 1)*aq
                                +math.exp(2 * (lambertw(math.sqrt(linkFlow[3] * math.exp(-2 * linkFlow[3]) / (2 * λ_ * sigma))).real) + linkFlow[
            3]) - 1)

    return λ_

def optimal_algorithmλy2(state):
    Eh1, Eh2, Eh3 = state
    λ_ = 0.001
    λ_2=aq*Eh3+Eh2-sigma * ((math.exp(2 * (lambertw(math.sqrt(linkFlow[3] * math.exp(-2 * linkFlow[3]) / (2 * aq*λ_ * sigma))).real) + linkFlow[
                3]) - 1)*aq + math.exp(2 * (lambertw(math.sqrt(linkFlow[2] * math.exp(-2 * linkFlow[2]) / (2 * λ_ * sigma))).real) + linkFlow[
                2]) - 1)
    while λ_2 < 10 ** (-3):
        λ_ = λ_ + 0.00001
        λ_2 = aq*Eh3+Eh2-sigma * ((math.exp(2 * (lambertw(math.sqrt(linkFlow[3] * math.exp(-2 * linkFlow[3]) / (2 * aq*λ_ * sigma))).real) + linkFlow[
                3]) - 1)*aq + math.exp(2 * (lambertw(math.sqrt(linkFlow[2] * math.exp(-2 * linkFlow[2]) / (2 * λ_ * sigma))).real) + linkFlow[
                2]) - 1)
    # tapq2=Eh3-sigma * (math.exp(2 * (lambertw(math.sqrt(linkFlow[3] * math.exp(-2 * linkFlow[3]) / (2 * λ_ * sigma))).real) + linkFlow[
    #             3]) - 1)

    return λ_

def tapq1(state,λ):
    Eh1,Eh2,Eh3=state
    tapq1 = Eh1 - sigma * (math.exp(
        2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 *λ * sigma))).real) + linkFlow[
            0]) - 1 + math.exp(
        2 * (lambertw(math.sqrt(linkFlow[1] * math.exp(-2 * linkFlow[1]) / (2 * λ * sigma))).real) + linkFlow[
            1]) - 1)
    return tapq1

def tapq2(state,λ):
    Eh1, Eh2, Eh3 = state
    tapq = Eh3 - sigma * (math.exp(2 * (lambertw(math.sqrt(linkFlow[3] * math.exp(-2 * linkFlow[3]) / (2 * λ * sigma))).real) + linkFlow[
        3]) - 1)
    return tapq

# with open('k.txt') as file_object:
#     contents = file_object.read()
    # for i in range(len(contents)):
    #     if contents[i+1]=='[':
    #         for j in range(65):
    #             if contents[i+1+j]==']':
    #                 print(contents[i+2:i+j])
    #                 continue
table1=pd.read_csv('table.csv')
reward_list=np.zeros([10,])
reward_list1=np.zeros([10,])
#print(len(table1))
# for i in range(len(table1)):
for i in range(120):             #针对table中数据的实际情况，将状态为0的全部不学习
    a=list(table1.loc[i])[1:]
    #a=[round(a[k])for k in range(len(a))]

    λ1=initial_algorithmλ(a)[0]
    λ2=initial_algorithmλ(a)[1]
    λ3=initial_algorithmλ(a)[2]
    if λ1<λ3*aq:
        λq3 = optimal_algorithmλy1(a)
        #tapq1 = tapq1(a, λq3)
        λ_op=[λq3/aq,λ2,λq3]
        Eh1=sigma * (math.exp(
            2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 * λq3 * sigma))).real) + linkFlow[
                0]) - 1 + math.exp(
            2 * (lambertw(math.sqrt(linkFlow[1] * math.exp(-2 * linkFlow[1]) / (2 * λq3 * sigma))).real) + linkFlow[1]) - 1)
        Eh3=sigma * (math.exp(2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 * λq3 * sigma))).real) + linkFlow[0]) - 1)
        b=[Eh1,a[1],Eh3]
    elif λ1>λ3*aq:
        tapq1_=tapq1(a,aq*λ3)
        while tapq1_>=0 and λ1>λ3*aq and a[2]>=0:
            a[0]=a[0]+o
            a[2]=a[2]-aq*o
            tapq1_=tapq1_-o
            λ1 = initial_algorithmλ(a)[0]
            λ3 = initial_algorithmλ(a)[2]
        λq3=optimal_algorithmλy1(a)
        #tapq1=tapq1(a,λ3)
        λ_op=[λq3/aq,λ2,λq3]
        Eh1 = sigma * (math.exp(
            2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 * λq3 * sigma))).real) + linkFlow[
                0]) - 1 + math.exp(
            2 * (lambertw(math.sqrt(linkFlow[1] * math.exp(-2 * linkFlow[1]) / (2 * λq3 * sigma))).real) + linkFlow[1]) - 1)
        Eh3 = sigma * (math.exp(
            2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 * λq3 * sigma))).real) + linkFlow[0]) - 1)
        b = [Eh1,a[1],Eh3]

    if λ_op[2]<λ_op[1]*aq:
        λq2 = optimal_algorithmλy2(b)
        λ_op1 =[λ_op[0],λq2 / aq, λq2]
    #tapq2 = tapq2(b, aq * λ_op[1])
    elif λ_op[2]>λ_op[1]*aq:
        tapq_2= tapq2(b, aq*λ_op[1])
        #print(tapq2)
        while tapq_2 >= 0 and λ_op[2]>λ_op[1]*aq and b[1] >= 0:
            b[2] = b[2] + o
            b[1] = b[1] - aq * o
            tapq_2 = tapq_2 - o
            λ_op[2] = initial_algorithmλ(b)[2]
            λ_op[1] = initial_algorithmλ(b)[1]
        λq2 = optimal_algorithmλy1(b)
        λ_op1 =[λ_op[0],λq2 / aq, λq2]


    #求解最优功率
    p1=sigma * (math.exp(2 * (lambertw(math.sqrt(linkFlow[0] * math.exp(-2 * linkFlow[0]) / (2 * λ_op1[0] * sigma))).real) + linkFlow[0]) - 1)
    p2=sigma * (math.exp(2 * (lambertw(math.sqrt(linkFlow[1] * math.exp(-2 * linkFlow[1]) / (2 * λ_op1[0] * sigma))).real) + linkFlow[1]) - 1)
    p3=sigma * (math.exp(2 * (lambertw(math.sqrt(linkFlow[2] * math.exp(-2 * linkFlow[1]) / (2 * λ_op1[1] * sigma))).real) + linkFlow[2]) - 1)
    p4=sigma * (math.exp(2 * (lambertw(math.sqrt(linkFlow[3] * math.exp(-2 * linkFlow[1]) / (2 * λ_op1[2] * sigma))).real) + linkFlow[3]) - 1)
    p_value=[p1,p2,p3,p4]
    capLink = [0.5 * math.log(1 + (x / sigma)) for x in p_value]  # capLink为实际链路流量
    reward = sum(linkFlow/(capLink - linkFlow))

    if i==i+i%10:
        reward_list = reward_list + reward_list1
        reward_list1=np.zeros([10,])
    reward_list1[i % 10] = -reward

print(reward_list/12)
plt.plot(np.arange(len(reward_list)),np.cumsum(reward_list/12))
plt.show()






























