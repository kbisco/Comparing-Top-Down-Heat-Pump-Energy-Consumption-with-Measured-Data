#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:12:23 2022

@author: mohammadrezqalla
"""
import os
os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/hp_proj1')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# model_output = np.loadtxt("total_model_output.csv", delimiter = ",")
# model_output_hp_fix = np.loadtxt("total_model_output_hp_fix.csv", delimiter = ",")
# model_output_hp_seer_10 = np.loadtxt("total_model_output_10_fix.csv", delimiter = ",")
# model_output_hp_seer_13 = np.loadtxt("total_model_output_13_fix.csv", delimiter = ",")
DC_data = np.loadtxt("DC_output.csv", delimiter = ",", encoding='utf-8-sig')
avedata_model = np.loadtxt("total_daily_hpec_avedata.csv", delimiter = ",")
testdata_model = np.loadtxt("total_daily_hpec_testdata.csv", delimiter = ",")

# model_in_range = model_output[237:812]
# model_in_range_hp_fix = model_output_hp_fix[237:812]
# model_in_range_seer_10 = model_output_hp_seer_10[237:812]
# model_in_range_seer_13 = model_output_hp_seer_13[237:812]
# plt.plot(testdata_model[237:812], label='Test Model Output')
# plt.plot(avedata_model[237:812], label='Ave Model Output', c='tab:orange')
# plt.plot(DC_data, label = "Nanogrid House Data", c='tab:green')

# plt.plot(model_in_range, label = "Model Output")
# plt.plot(model_in_range_hp_fix, label = "Model Output")
# plt.plot(model_in_range_seer_10, label = "Model Output")
# plt.plot(model_in_range_seer_13, label = "Model Output")

time = pd.date_range(start=datetime.date(2020,8,25), end=datetime.date(2022,3,22)).tolist()
fig, ax = plt.subplots()
ax.plot(time, testdata_model[237:812], label='Test Model Output')
ax.plot(time, avedata_model[237:812], label='Ave Model Output', c='tab:orange')
ax.plot(time, DC_data, label = "Nanogrid House Data", c='tab:green')
plt.xticks(rotation = 45)
# fig.autofmt_xdate()
# ax.set_xlim([datetime.date(2020, 12, 31), datetime.date(2021, 1, 8)])
# ax.set_ylim([0, 8])
# plt.show()

plt.legend()
plt.xlabel("Time")
plt.ylabel("Daily Heat Pump Electricity (kWh)")
# plt.title("SEER 13 Fix")
os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/hp_proj1/researchletter_plots')
plt.savefig('nanogridVSmodel_alldata.png', dpi=600, bbox_inches='tight')
plt.savefig('nanogridVSmodel_alldata.eps', bbox_inches='tight')

#%% Error Calculation
def root_mean_squared_error(true, pred):
    squared_error = np.square(true - pred) 
    sum_squared_error = np.sum(squared_error)
    rmse_loss = np.sqrt(sum_squared_error / true.size)
    return rmse_loss
rmse = root_mean_squared_error(everything[237:812], DC_data) # 894.632

def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss
rrmse = relative_root_mean_squared_error(everything[237:812], DC_data) # 19.781
