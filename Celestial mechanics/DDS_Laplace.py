import pandas as pd
import numpy as np
# import math
# import matplotlib.pyplot as plt
import datetime
import time

import DDS_CMpack as DDS


def Calculatefort0(position_fram_fil,obs,t0):

    # 为循环重新format，取出需要的数据
    Observe_time = len(position_fram_fil)

    # obs = [1,3,5] # 取全部观测，做最小二乘法定初轨

    # 取定t0
    # index = 1
    # t0 = 1.5
    # t0 = 0.5*(position_fram_fil['deltaT'].values[0] + position_fram_fil['deltaT'].values[5])
    # print(t0)
    # set F0 and G0 to begin the loop

    F0 = np.ones(Observe_time).tolist()
    G0 = (position_fram_fil['deltaT'].values - t0).tolist()

    FN0 = []; GN0 = []
    P = []; Q = []
    Lambda = []; Niu = []; Miu = []
    tao = [];
    for i in range(len(obs)): # 把要用的那几次观测数据提取出来：
        j = obs[i]
        FN0.append(F0[j])
        GN0.append(G0[j])
        P.append(position_fram_fil['P'].values[j])
        Q.append(position_fram_fil['Q'].values[j])
        Niu.append(position_fram_fil['Niu'].values[j])
        Miu.append(position_fram_fil['Miu'].values[j])
        Lambda.append(position_fram_fil['Lambda'].values[j])
        tao.append(position_fram_fil['deltaT'].values[j] - t0)

    # np.array(tao)*time_unit

    # 循环，解出F,G。

    Nwind = 0
    key = 0
    F = [FN0]; G = [GN0]

    while key == 0:

        # 在一次循环中:
        # 最终的目的是解(A,b)的增广矩阵 - 对应(x,y,z,vx,vy,vz)
        A = []
        b = []

        # we have : Lt, position_fram_fil
        ## 从F,G 解r0,v0
        for j in range(len(obs)): # 三次观测(现在的F,G已经只剩我们需要的那几次观测的值了)

            A.append([Niu[j]*F[Nwind][j],0,-1*Lambda[j]*F[Nwind][j],Niu[j]*G[Nwind][j],0,-1*Lambda[j]*G[Nwind][j]])
            A.append([0,Niu[j]*F[Nwind][j],-1*Miu[j]*F[Nwind][j],0,Niu[j]*G[Nwind][j],-1*Miu[j]*G[Nwind][j]])
            b.append(P[j])
            b.append(Q[j])

    #################################################################
        # 三次观测，解矩阵得到最终的结果

    #     AA = np.array(A)
    #     bb = np.array(b)

    #     print(Nwind,b)

    #     ans = np.linalg.solve(AA,bb) #ans 6 位对应(x,y,z,vx,vy,vz)

    #     type(ans)
    #     r0 = np.sqrt(ans[0]**2 + ans[1]**2 + ans[2]**2)
    #     v0 = np.sqrt(ans[3]**2 + ans[4]**2 + ans[5]**2)
    #     print(Nwind,':',r0,v0)

    #################################################################
        # 多次测量、多资料问题的解法：

        AA = np.array(A)
        b = np.array(b)

    #     print(np.dot(AA.T.copy(),AA))
    #     print(np.dot(AA.T.copy(),b))

        ans = np.linalg.solve(np.dot(AA.T.copy(),AA),\
                                 np.dot(AA.T.copy(),b))

        r0 = np.sqrt(ans[0]**2 + ans[1]**2 + ans[2]**2)
        v0 = np.sqrt(ans[3]**2 + ans[4]**2 + ans[5]**2)
        print(Nwind,':',r0,v0)

    #################################################################
        # 算出新的F 和 G，为接下来的迭代做准备

        # r0,v0获得F,G
        FNw = []; GNw = [];
        for j in range(len(obs)):
    #         print(j)
            FNw.append(DDS.F(r0,v0,tao[j]))
            GNw.append(DDS.G(r0,v0,tao[j]))

        F.append(FNw); G.append(GNw)

    #################################################################
        # 判定deltaF or G的大小，并且决定最后是不是选取这一项。
        Nwind += 1
        delta_F = np.abs(np.array(F[Nwind]) - np.array(F[Nwind - 1]));
        delta_G = np.abs(np.array(G[Nwind]) - np.array(G[Nwind - 1]))
        big_delta = np.max([np.max(delta_F),np.max(delta_G)])

        print(delta_F,delta_G)
        print('--------------------')

        if big_delta < 1e-13:
            print('end')
            print(delta_F,delta_G)
            r0_a = np.array([ans[0],ans[1],ans[2]])
            v0_a = np.array([ans[3],ans[4],ans[5]])
            key = 1

    # 地球半径是 6371 km

    # print(r0_a*R_earth,r0*R_earth,(r0-r_station_earth)*R_earth)
    #
    # print(v0_a*R_earth/time_unit,v0*R_earth/time_unit)
    #
    # print(t0*time_unit)

    return r0_a,v0_a
