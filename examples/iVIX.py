# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:52:52 2018

@author: 量小白
"""
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import interpolate

sys.path.append('.')
sys.path.append('../../stockguru')
import warnings

warnings.filterwarnings("ignore")

# from trinity.utils import data, db


def load_shibor():
    sql = 'select * from shibor order by date desc'
    df = pd.read_sql_query(sql, db.engine)
    df = df.set_index('date')
    return df

def load_options(opt_code):
    sql = 'select ts_code, name, opt_code, call_put, exercise_price, list_date, delist_date from index_opt_basic where opt_code = \'{}\''.format(opt_code)
    df1 = pd.read_sql_query(sql, db.engine)
    codes = df1['ts_code'].values.tolist()
    if (len(codes) <= 0):
        return None
    codes = tuple(codes)
    sql = 'select ts_code, trade_date, close from index_opt_daily where ts_code in {}'.format(codes)
    df2 = pd.read_sql_query(sql, db.engine)
    df = df2.merge(df1,on='ts_code',how='left')
    df = df.set_index('trade_date')
    return df

shibor_rate = load_shibor() #pd.read_csv('c:/Develop/stockguru/trinity/libs/ivix/shibor.csv',index_col=0,encoding='GBK')
options_data = load_options('510300.OF') #pd.read_csv('c:/Develop/stockguru/trinity/libs/ivix/options.csv',index_col=0,encoding='GBK')
# print(options_data.index)
if (not options_data is None):
    tradeday = options_data.index #[datetime.strptime(d,'%Y%m%d').strftime('%Y%m%d') for d in options_data.index] #pd.read_csv('c:/Develop/stockguru/trinity/libs/ivix/tradeday.csv',encoding='GBK')
#true_ivix = pd.read_csv('c:/Develop/stockguru/trinity/libs/ivix/ivixx.csv',encoding='GBK')

#==============================================================================
# 开始计算ivix部分
#==============================================================================
def periodsSplineRiskFreeInterestRate(options, date):
    """
    params: options: 计算VIX的当天的options数据用来获取expDate
            date: 计算哪天的VIX
    return：shibor：该date到每个到期日exoDate的risk free rate

    """
    date = datetime.strptime(date,'%Y%m%d')
    #date = datetime(date.year,date.month,date.day)
    exp_dates = np.sort(options.delist_date.unique())
    periods = {}
    for epd in exp_dates:
        epd = pd.to_datetime(epd)
        periods[epd] = (epd - date).days*1.0/365.0
    shibor_date = datetime.strptime(shibor_rate.index[0], "%Y%m%d") 
    if date >= shibor_date:
        date_str = shibor_rate.index[0]
        shibor_values = shibor_rate.iloc[0].values
        #shibor_values = np.asarray(list(map(float,shibor_values)))
    else:
        date_str = date.strftime("%Y%m%d") 
        shibor_values = shibor_rate.loc[date_str].values
        #shibor_values = np.asarray(list(map(float,shibor_values)))
        
    shibor = {}
    period = np.asarray([1.0, 7.0, 14.0, 30.0, 90.0, 180.0, 270.0, 360.0]) / 360.0
    min_period = min(period)
    max_period = max(period)
    for p in periods.keys():
        tmp = periods[p]
        if periods[p] > max_period:
            tmp = max_period * 0.99999
        elif periods[p] < min_period:
            tmp = min_period * 1.00001
        # 此处使用SHIBOR来插值
        # sh = interpolate.spline(period, shibor_values, tmp, order=3)

        tck = interpolate.splrep(period, shibor_values)
        sh = interpolate.splev(tmp, tck)

        shibor[p] = sh/100.0
    return shibor


def getHistDayOptions(vixDate,options_data):
    options_data = options_data.loc[vixDate,:]
    return options_data
    
    

def getNearNextOptExpDate(options, vixDate):
    # 找到options中的当月和次月期权到期日；
    # 用这两个期权隐含的未来波动率来插值计算未来30隐含波动率，是为市场恐慌指数VIX；
    # 如果options中的最近到期期权离到期日仅剩1天以内，则抛弃这一期权，改
    # 选择次月期权和次月期权之后第一个到期的期权来计算。
    # 返回的near和next就是用来计算VIX的两个期权的到期日
    """
    params: options: 该date为交易日的所有期权合约的基本信息和价格信息
            vixDate: VIX的计算日期
    return: near: 当月合约到期日（ps：大于1天到期）
            next：次月合约到期日
    """
    vixDate = datetime.strptime(vixDate,'%Y%m%d')
    optionsExpDate = list(pd.Series(options.delist_date.values.ravel()).unique())
    optionsExpDate = [datetime.strptime(i,'%Y%m%d') for i in optionsExpDate]
    near = min(optionsExpDate)
    optionsExpDate.remove(near)
    if (near.year == vixDate.year and near.month == vixDate.month and near.day - vixDate.day <= 1):
        near = min(optionsExpDate)
        optionsExpDate.remove(near)
    nt = min(optionsExpDate)
    return near, nt

def getStrikeMinCallMinusPutClosePrice(options):
    # options 中包括计算某日VIX的call和put两种期权，
    # 对每个行权价，计算相应的call和put的价格差的绝对值，
    # 返回这一价格差的绝对值最小的那个行权价，
    # 并返回该行权价对应的call和put期权价格的差
    """
    params:options: 该date为交易日的所有期权合约的基本信息和价格信息
    return: strike: 看涨合约价格-看跌合约价格 的差值的绝对值最小的行权价
            priceDiff: 以及这个差值，这个是用来确定中间行权价的第一步
    """
    call = options[options.call_put=="C"].set_index("exercise_price").sort_index()
    put  = options[options.call_put=="P"].set_index("exercise_price").sort_index()
    callMinusPut = call.close - put.close
    strike = abs(callMinusPut).idxmin()
    priceDiff = callMinusPut[strike].min()
    return strike, priceDiff

def calSigmaSquare( options, FF, R, T):
    # 计算某个到期日期权对于VIX的贡献sigma；
    # 输入为期权数据options，FF为forward index price，
    # R为无风险利率， T为期权剩余到期时间
    """
    params: options:该date为交易日的所有期权合约的基本信息和价格信息
            FF: 根据上一步计算得来的strike，然后再计算得到的forward index price， 根据它对所需要的看涨看跌合约进行划分。
                取小于FF的第一个行权价为中间行权价K0， 然后选取大于等于K0的所有看涨合约， 选取小于等于K0的所有看跌合约。
                对行权价为K0的看涨看跌合约，删除看涨合约，不过看跌合约的价格为两者的均值。
            R： 这部分期权合约到期日对应的无风险利率 shibor
            T： 还有多久到期（年化）
    return：Sigma：得到的结果是传入该到期日数据的Sigma
    """
    callAll = options[options.call_put==u"C"].set_index(u"exercise_price").sort_index()
    putAll  = options[options.call_put==u"P"].set_index(u"exercise_price").sort_index()
    callAll['deltaK'] = 0.05
    putAll['deltaK']  = 0.05
    
    # Interval between strike prices
    index = callAll.index
    if len(index) < 3:
        callAll['deltaK'] = index[-1] - index[0]
    else:
        for i in range(1,len(index)-1):
            callAll['deltaK'].loc[index[i]] = (index[i+1]-index[i-1])/2.0
        callAll['deltaK'].loc[index[0]] = index[1]-index[0]
        callAll['deltaK'].loc[index[-1]] = index[-1] - index[-2]
    index = putAll.index
    if len(index) < 3:
        putAll['deltaK'] = index[-1] - index[0]
    else:
        for i in range(1,len(index)-1):
            putAll['deltaK'].loc[index[i]] = (index[i+1]-index[i-1])/2.0
        putAll['deltaK'].loc[index[0]] = index[1]-index[0]
        putAll['deltaK'].loc[index[-1]] = index[-1] - index[-2]
    
    call = callAll[callAll.index > FF]
    put  = putAll[putAll.index < FF]
    FF_idx = FF
    if put.empty:
        FF_idx = call.index[0]
        callComponent = call.close*call.deltaK/call.index/call.index
        sigma = (sum(callComponent))*np.exp(T*R)*2/T
        sigma = sigma - (FF/FF_idx - 1)**2/T
    elif call.empty:
        FF_idx = put.index[-1]
        putComponent = put.close*put.deltaK/put.index/put.index
        sigma = (sum(putComponent))*np.exp(T*R)*2/T
        sigma = sigma - (FF/FF_idx - 1)**2/T
    else:
        FF_idx = put.index[-1]
        try:
            if len(putAll.loc[FF_idx].close.values) > 1:
                put['close'].iloc[-1] = (putAll.loc[FF_idx].close.values[1] + callAll.loc[FF_idx].close.values[0])/2.0
        except:
            put['close'].iloc[-1] = (putAll.loc[FF_idx].close + callAll.loc[FF_idx].close)/2.0

        callComponent = call.close*call.deltaK/call.index/call.index
        putComponent  = put.close*put.deltaK/put.index/put.index
        sigma = (sum(callComponent)+sum(putComponent))*np.exp(T*R)*2/T
        sigma = sigma - (FF/FF_idx - 1)**2/T
    return sigma

def changeste(t):
    if t.month>=10:
        str_t = t.strftime('%Y%m%d')
    else:
        str_t = t.strftime('%Y%m%d')
        # str_t = str_t[:5]+str_t[6:]
    return str_t

def calDayVIX(vixDate):
    # 利用CBOE的计算方法，计算历史某一日的未来30日期权波动率指数VIX
    """
    params：vixDate：计算VIX的日期  '%Y%m%d' 字符串格式
    return：VIX结果
    """

    # 拿取所需期权信息
    options = getHistDayOptions(vixDate,options_data)
    near, nexts = getNearNextOptExpDate(options, vixDate)
    shibor = periodsSplineRiskFreeInterestRate(options, vixDate)
    R_near = shibor[datetime(near.year,near.month,near.day)]
    R_next = shibor[datetime(nexts.year,nexts.month,nexts.day)]
    
    str_near = changeste(near)
    str_nexts = changeste(nexts)
    optionsNearTerm = options[options.delist_date == str_near]
    optionsNextTerm = options[options.delist_date == str_nexts]
    # time to expiration
    vixDate = datetime.strptime(vixDate,'%Y%m%d')
    T_near = (near - vixDate).days/365.0
    T_next = (nexts- vixDate).days/365.0
    # the forward index prices
    nearPriceDiff = getStrikeMinCallMinusPutClosePrice(optionsNearTerm)
    nextPriceDiff = getStrikeMinCallMinusPutClosePrice(optionsNextTerm)
    near_F = nearPriceDiff[0] + np.exp(T_near*R_near)*nearPriceDiff[1]
    next_F = nextPriceDiff[0] + np.exp(T_next*R_next)*nextPriceDiff[1]
    # 计算不同到期日期权对于VIX的贡献
    near_sigma = calSigmaSquare( optionsNearTerm, near_F, R_near, T_near)
    next_sigma = calSigmaSquare(optionsNextTerm, next_F, R_next, T_next)

    # 利用两个不同到期日的期权对VIX的贡献sig1和sig2，
    # 已经相应的期权剩余到期时间T1和T2；
    # 差值得到并返回VIX指数(%)
    w = (T_next - 30.0/365.0)/(T_next - T_near)
    vix = T_near*w*near_sigma + T_next*(1 - w)*next_sigma
    return 100*np.sqrt(abs(vix)*365.0/30.0)


def test_day():
    day = '20180607'
    dvix = calDayVIX(day)
    print(dvix)

def calc_days(start_date, end_date):
    ivix = []
    days = data.get_cal_dates(start_date, end_date)
    for day in days:
        dvix = calDayVIX(day)
        ivix.append(round(dvix,2))
        print("\r{}".format(dvix),end="")
        # print(dvix)
    print(ivix)


def draw():  
    attr = true_ivix['日期'].tolist()
    # import matplotlib.pyplot as plt
    # plt.plot(attr, ivix)
        
    from pyecharts.charts import Line
    line = Line("中国波指")
    line.add("中证指数发布", attr, true_ivix['收盘价(元)'].tolist(), mark_point=["max"])
    line.add("手动计算", attr, ivix, mark_line=["max",'average'])
    line.render('vix.html')

def get_ivix(trade_date, opt_code='510050.OF'):
    global options_data
    try:
        options_data = load_options(opt_code=opt_code)
        dvix = calDayVIX(trade_date)
        return dvix
    except Exception as e:
        import traceback
        traceback.print_exc()
        return -1

if __name__ == "__main__":
    # load_options('OP510300.SH')
    days = data.get_cal_dates('20220101','20220308')
    for day in days:
        print(day)
        print(get_ivix(day))
        print(get_ivix(day,opt_code='510300.OF'))
    





