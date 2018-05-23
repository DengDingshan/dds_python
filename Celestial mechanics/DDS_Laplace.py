# coding=utf-8
# import pandas as pd
import numpy as np
# import math
# import matplotlib.pyplot as plt
# import datetime
# import time

import DDS_CMpack as DDS


def Calculatefort0(position_fram_fil,obs,t0,epsilon):

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
    G0 = (position_fram_fil['formatT'].values - t0).tolist()

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
        tao.append(position_fram_fil['formatT'].values[j] - t0)

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

        # AA = np.array(A)
        # bb = np.array(b)
        #
        # print(Nwind,b)
        #
        # ans = np.linalg.solve(AA,bb) #ans 6 位对应(x,y,z,vx,vy,vz)
        #
        # # type(ans)
        # r0 = np.sqrt(ans[0]**2 + ans[1]**2 + ans[2]**2)
        # v0 = np.sqrt(ans[3]**2 + ans[4]**2 + ans[5]**2)
        # print(Nwind,':',r0,v0)

    #################################################################
        # 多次测量、多资料问题的解法：

        AA = np.array(A)
        b = np.array(b)

    #     print(np.dot(AA.T.copy(),AA))
    #     print(np.dot(AA.T.copy(),b))

        # print(AA.shape,b.shape)

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

        if big_delta < epsilon:
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


def arctan2(sinE,cosE):
    if sinE >=0 and cosE >=0: # I
        output = np.arcsin(sinE)
    elif sinE >=0 and cosE <=0: # II
        output = np.pi - np.arcsin(sinE)
    elif sinE <=0 and cosE >=0: # IV
        output = 2*np.pi + np.arcsin(sinE)
    else: # III
        output = np.pi - np.arcsin(sinE)
    return output


def Calculatefor6elements(r0_a,v0_a,t0,key):
    # 从第二步开始，因为不需要用到F,G的计算，所以不用再取理论单位，现在取国际单位制
    # key == 0,绕太阳旋转； key== 1，绕地球旋转
    # astronomy constants

    R_earth = 6371e3 # m
    au = 149597870e3 # m

    M_earth = 5.965e24 # kg
    M_sun = 1.989e30 # kg

    G_graviation = 6.672e-11 # N·m^2 /kg^2

    r_earth_sun = 1*au

    if key == 0:
        ## unit for time -> planet
        time_unit_day = 58.1324409 # Mean solar day
        time_unit = time_unit_day * 86400

        miu_GM = G_graviation*M_sun

        R_unit = r_earth_sun

    elif key == 1:
        ### unit for time -> artifact satellite
        time_unit = 806.81163 #806.8116 # SI

        # miu_GM = 398600.5e-6 # km^3/SI^2
        miu_GM = G_graviation*M_earth # m^3/SI^2

        R_unit = R_earth

    # input r0,v0,t0
    r0 = r0_a*R_unit # unit= m
    r0_norm = np.sqrt(sum([ i*i for i in r0])) # m

    v0 = v0_a*R_unit/time_unit # unit= m/SI
    v0_norm = np.sqrt(sum([ i*i for i in v0])) # m/SI

    t0 = t0*time_unit # unit = SI from the first data we have
    ## t0 根据自己的单位去选择。

    # get a

    a = 1/(2/r0_norm - v0_norm**2/miu_GM) # unit = m

    print('a',a,'m')

    # get n\e\E after a
    n = np.sqrt(miu_GM/(a**3))
    print('n:',n,'rad/s','- Not Output')

    # tan_E = (1 - r0_norm/a)*(a**2*n)/(np.dot(r0,v0))

    e = np.sqrt( (1 - r0_norm/a)**2 + (np.dot(r0,v0) / (n*a**2))**2 )
    # e = 0.01002
    print('e:',e,'Nan')

    # print(1 - r0_norm/a)
    # print(np.dot(r0,v0) / (n*a**2))

    cos_E = (1 - r0_norm/a)/e
    sin_E = (np.dot(r0,v0) / (a**2*n))/e

    # print(sin_E,cos_E,tan_E)

    E = arctan2(sin_E,cos_E)
    print('E:',E,'rad','- Not Output')

    # keplar 定 M
    M = E - e*sin_E
    print('M:',M,'rad')

    # 定 i Omega w

    #### method 1
    P = (cos_E/r0_norm)*r0 - (sin_E/(a*n))*v0
    Q = (sin_E/(r0_norm)*np.sqrt(1-e**2))*r0 + ((cos_E - e)/(a*n*np.sqrt(1-e**2)))*v0
    R = np.cross(P,Q)

    # cos_i = R[2]
    # tan_Omega = -1*R[0]/R[1]

    # get w ->[-90,+90]
    tan_w = P[2]/Q[2]

    ############################################################

    #### method 2
    h = np.cross(r0,v0)
    h_norm = np.sqrt(np.dot(h,h))

    h_A, h_B, h_C = h[0:3]

    # get cos_i, sin_i
    cos_i_check = (h_C/h_norm)
    tan_i = np.sqrt(h_A**2 + h_B**2) / h_C

    sin_i = tan_i*cos_i_check


    # get cos_\sin_Omega
    sin_Omega = h_A/(h_norm*sin_i)
    cos_Omega = -h_B/(h_norm*sin_i)

    # finally get i Omega w

    i = arctan2(sin_i,cos_i_check)

    Omega = arctan2(sin_Omega,cos_Omega)

    w = np.arctan(tan_w)

    # np.degrees(np.array([i,Omega,w]))
    print('i,Omega,w:',i,Omega,w,'rad/rad/rad')

    return a,e,M,i,Omega,w,P,Q,R
