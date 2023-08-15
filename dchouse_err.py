#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:41:19 2023

@author: kbiscoch
"""
import numpy as np
import os

os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/hp_proj1')
DC_data = np.loadtxt("DC_output.csv", delimiter = ",", encoding='utf-8-sig')
avedata_model = np.loadtxt("total_daily_hpec_avedata.csv", delimiter = ",")

def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss
rrmse2 = relative_root_mean_squared_error(DC_data,avedata_model[237:812])  # 27.8%
rrmse2_sum = relative_root_mean_squared_error(np.sum(DC_data), np.sum(avedata_model[237:812])) # 0.05%