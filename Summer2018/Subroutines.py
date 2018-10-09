def rv2elements(r0,v0):
    ## 输入为向量的r0,v0，用于计算相应的轨道六根数，输出的顺序为a,e,M,i,Omega,w
    import numpy as np
    import math
    # 先继续使用已经在使用的理论单位！
    miu_GM = 1
    R_earth = 1

    norm_r0 = np.sqrt(r0.dot(r0))
    norm_v0 = np.sqrt(v0.dot(v0))

    # a n e
    a = (2/norm_r0 - (norm_v0**2)/miu_GM)**(-1)

    n = np.sqrt(miu_GM/(a**3))

    e = np.sqrt( (1 - norm_r0/a)**2 + (r0.dot(v0) / (n*a**2))**2)

    # get M
    cos_E = (1 - norm_r0/a)/e
    sin_E = (r0.dot(v0) / ((a**2)*n))/e
    E = math.atan2(sin_E,cos_E)
    M = E - e*math.sin(E)

    # PQR
    P = (cos_E/norm_r0)*r0 - (sin_E/(a*n))*v0
    Q = (sin_E/(norm_r0)*np.sqrt(1-e**2))*r0 + ((cos_E - e)/(a*n*np.sqrt(1-e**2)))*v0
    R = np.cross(P,Q)

    # w
    pz = P[2]; qz = Q[2]
    tan_w_raw = pz/qz
    if pz >=0 and qz >= 0:
        # w I
        w = math.atan(tan_w_raw)
    elif pz >= 0 and qz < 0:
        # w II
        w = math.atan(tan_w_raw) + np.pi
    elif pz < 0 and qz < 0:
        # w III
        w = math.atan(tan_w_raw) + np.pi
    elif pz < 0 and qz >= 0:
        # w IV
        w = math.atan(tan_w_raw) + 2*np.pi

    # h
    h = np.cross(r0,v0)
    norm_h = np.sqrt(np.dot(h,h))

    h_A, h_B, h_C = h[0:3]

    # get cos_i, sin_i
    cos_i = (h_C/norm_h)
    tan_i = np.sqrt(h_A**2 + h_B**2) / h_C

    sin_i = tan_i*cos_i

    # get cos_\sin_Omega
    sin_Omega = h_A/(norm_h*sin_i)
    cos_Omega = -h_B/(norm_h*sin_i)

    i_final = math.atan2(sin_i,cos_i)
    Omega = math.atan2(sin_Omega,cos_Omega)

    return [a,e,M,i_final,Omega,w]

def elements2rv(a,e,M,i,Omega,w,T):
    ## 输入轨道六根数和需要求根数的时间（时间取做一个array)，以算出当时的轨道位置和速度矢量
    import numpy as np
    import math

    miu_GM = 1

    n = np.sqrt(miu_GM/(a**3))
    M_total = M + n*T

    # 迭代 开普勒方程解 E
    E_total = []
    for i in range(len(M_total)):
        epsilon = 1e-10
        key = 0
        En0 = M_total[i]

        En_new = En0
        while key == 0:
            En_old = En_new
            En_new = En_old - (En_old - M_total[i] - e*np.sin(En_old))/(1-e*np.cos(En_old))
            delta_En = np.abs(En_new - En_old)

            if delta_En < epsilon:
                key = 1;

        E_total.append(En_new)

    # 计算新的 P,Q

    P_new = np.array([np.cos(Omega)*np.cos(w-2*np.pi) - np.sin(Omega)*np.sin(w)*np.cos(i_fin),\
                     np.sin(Omega)*np.cos(w-2*np.pi) + np.cos(Omega)*np.sin(w)*np.cos(i_fin),\
                     np.sin(w)*np.sin(i_fin)])
    Q_new = np.array([-np.cos(Omega)*np.sin(w) - np.sin(Omega)*np.cos(w)*np.cos(i_fin),\
                     -np.sin(Omega)*np.sin(w) + np.cos(Omega)*np.cos(w)*np.cos(i_fin),\
                     np.cos(w)*np.sin(i_fin)])

    # 计算新的 r,v
    Rt = [];Vt = []
    for Et in E_total:
        rt = a*(np.cos(Et) - e)*P_new + a*np.sqrt(1 - e**2)*np.sin(Et)*Q_new
        rt_norm = np.sqrt( np.dot(rt,rt) )

        vt = -a**2*n/rt_norm*np.sin(Et)*P_new + a**2*n/rt_norm*np.sqrt(1 - e**2)*np.cos(Et)*Q_new
        vt_norm = np.sqrt( np.dot(vt,vt) )

        L_rt = rt/ np.sqrt( np.dot(rt,rt))

        print(rt_norm,vt_norm)

        Rt.append(rt)
        Vt.append(vt)

    return Rt,Vt
