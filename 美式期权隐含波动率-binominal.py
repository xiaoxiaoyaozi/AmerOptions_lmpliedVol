import pandas as pd
import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt


n_steps = 1000    #number of steps
time= 142/365   #time to maturity
strike = 3200    # strike price
spot = 2734     # spot
time_step = float(time / n_steps) # 每一步的时间长度
time_sqrt = np.sqrt(time_step)

R=0.031    #无风险利率：中债国债十年期到期收益率
discount = np.exp(-R * time_step)
call_price = 52.5
put_price = 511.5

def AmerImpliedVol(opt_price, spot, strike, time, callPutInd, n_steps):
    '''
    :param callPutInd: call/put标识，call为1，put为-1
    '''
    horizontal = np.array(range(0, n_steps+1), dtype='float64')
    horizontal.shape = (1, n_steps+1)
    vertical=np.array(range(0, -2*n_steps-2, -2), dtype='float64')
    vertical.shape=(n_steps+1, 1)
    tree_template = vertical + horizontal
    growth_factor = np.exp(R * time_step)

    def step_price(sigma):
        up = np.exp(sigma * time_sqrt)
        down = 1 / up
        p_up = (growth_factor - down) / (up - down)
        p_down =1 - p_up

        tree = np.power(up, tree_template) * spot

        cp = callPutInd
        def payoff(values):
            return np.maximum(cp * (values-strike), 0)

        current = payoff(tree[:, n_steps])

        for i in range(n_steps, 0, -1):
            after_step = discount * (p_up * current[:n_steps] + p_down * current[1:])
            payoffs = payoff(tree[:, i-1])
            current[:-1] = np.maximum(after_step, payoffs[:-1])
        return current[0] - opt_price

    return newton(step_price, 0.3)

print(AmerImpliedVol(put_price, spot, strike, time, -1., n_steps))
print(AmerImpliedVol(call_price, spot, strike, time, 1., n_steps))