#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:12:23 2022

@author: mohammadrezqalla, kelseybiscocho
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


DC_data = np.loadtxt("DC_output.csv", delimiter = ",", encoding='utf-8-sig')
avedata_model = np.loadtxt("total_daily_hpec_avedata.csv", delimiter = ",")
testdata_model = np.loadtxt("total_daily_hpec_testdata.csv", delimiter = ",")

time = pd.date_range(start=datetime.date(2020,8,25), end=datetime.date(2022,3,22)).tolist()
fig, ax = plt.subplots()
# ax.plot(time, testdata_model[237:812], label='Test Model Output')
ax.plot(time, avedata_model[237:812], label='Ave Model Output', c='tab:orange')
ax.plot(time, DC_data, label = "Nanogrid House Data", c='tab:green')
plt.xticks(rotation = 45)

plt.legend()
plt.xlabel("Time")
plt.ylabel("Daily Heat Pump Electricity (kWh)")

plt.savefig('figure5.tiff', dpi=600, bbox_inches='tight')
# plt.savefig('figure5.eps', bbox_inches='tight')

#%% Error Calculation
def root_mean_squared_error(true, pred):
    squared_error = np.square(true - pred) 
    sum_squared_error = np.sum(squared_error)
    rmse_loss = np.sqrt(sum_squared_error / true.size)
    return rmse_loss
rmse = root_mean_squared_error(DC_data, avedata_model[237:812]) 

def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss
rrmse2 = relative_root_mean_squared_error(DC_data,avedata_model[237:812]) 

