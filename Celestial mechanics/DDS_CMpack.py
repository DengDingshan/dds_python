def UTC2SG(yyyy,mm,dd,hours):
    # J2000，从UTC计算对应的jd_目的是最终算出儒略世纪数
    J2000 = 2451545.0
    # J2000：UTC时间2000年1月1日11:58:55.816。
    date_start = datetime.datetime.strptime("2000-1-1 11:58:55","%Y-%m-%d %H:%M:%S")
    Date_str = str(yyyy) +'-'+ str(mm) +'-'+ str(dd)
    Date = datetime.datetime.strptime(Date_str,"%Y-%m-%d")

    # 用unix时间戳计算到J2000的时间差
    un_start = time.mktime(date_start.timetuple()) + 0.816
    un_this = time.mktime(Date.timetuple()) + hours*3600

    # 返回儒略世纪数（1儒略年=31557600 SI秒）
    t = (un_this - un_start)/(31557600*100)

    # 返回以J2000开始的儒略日数量
    jd = t*(365.25*100)

    # 给出平恒星时(s为单位)
    SG = 18.6973746*3600 + 879000.0513367*3600*t + 0.093104*t**2 - 6.2e-6*t**3
    # degree = SG%(3600*24)/(3600*24)*360

    return SG,jd

def RADEC2xyz(r,RA,DEC):
    # input: RA,DEC in the unite of degree, r is the distance
    # steps: u(rad) = RA(deg); v(rad) = DEC(deg)
    # output: x,y,z
    u,v = RA*2*np.pi/360,DEC*2*np.pi/360
    x = r*np.cos(u)*np.cos(v)
    y = r*np.sin(u)*np.cos(v)
    z = r*np.sin(v)
    return np.array([x,y,z])

def Rx(x):
    import numpy as np
    # x is in the unit of deg, while xx is in the unit of rad
    xx = x*2*np.pi/360
    Rx = np.array([[1,0,0],\
                     [0,np.cos(xx),np.sin(xx)],\
                     [0,-np.sin(xx),np.cos(xx)]])
    return Rx

def equator2ecliptic(origin,epsilon,key):
    # epsilon是黄赤交角
    # from equ to ecl , key = 1
    # from ec1 to equ , key = 2
    # origin是个np.的1x3向量
    if key == 1:
        output = np.dot(origin,Rx(epsilon).T)
    elif key == 2:
        output = np.dot(origin,Rx(-epsilon).T)
    else:
        print('you need to input a key as: \n from equ to ecl , key = 1' + \
        '\n from ec1 to equ , key = 2')

    return output
