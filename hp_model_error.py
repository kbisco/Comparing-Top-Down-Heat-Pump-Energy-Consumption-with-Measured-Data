#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:58:12 2023

@author: kbiscoch
"""
import os
import numpy as np
import scipy.stats
# import seaborn as sns

os.chdir('/Users/kbiscoch/Documents/Research_remote/GitHub/Comparing-Top-Down-Heat-Pump-Energy-Consumption-with-Measured-Data/hp_model_data')
col_names = ['temp', 'COP']

D1_cool = np.loadtxt('D1cool_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5))
D2_cool = np.loadtxt('D2cool_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6))
D3_cool = np.loadtxt('D3cool_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,))
D4_cool = np.loadtxt('D4cool_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4))
D5_cool = np.loadtxt('D5cool_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4))
HL_cool = np.loadtxt('HLAB_cooling_temp_COPs.csv', delimiter=',')

D1_heat = np.loadtxt('D1heat_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5))
D2_heat = np.loadtxt('D2heat_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6))
D3_heat = np.loadtxt('D3heat_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,))
D4_heat = np.loadtxt('D4heat_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4))
D5_heat = np.loadtxt('D5heat_COP_temp.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4))
HL_heat = np.loadtxt('HLAB_heating_temp_COPs.csv', delimiter=',')

# D1_cool = pd.read_csv('D1cool_COP_temp.csv', delimiter=',', index_col=0)
# D2_cool = pd.read_csv('D2cool_COP_temp.csv', delimiter=',', index_col=0)
# D3_cool = pd.read_csv('D3cool_COP_temp.csv', delimiter=',', index_col=0)
# D4_cool = pd.read_csv('D4cool_COP_temp.csv', delimiter=',', index_col=0)
# D5_cool = pd.read_csv('D5cool_COP_temp.csv', delimiter=',', index_col=0)
# HL_cool = pd.read_csv('HLAB_cooling_temp_COPs.csv', delimiter=',', names=col_names)

# D1_heat = pd.read_csv('D1heat_COP_temp.csv', delimiter=',', index_col=0)
# D2_heat = pd.read_csv('D2heat_COP_temp.csv', delimiter=',', index_col=0)
# D3_heat = pd.read_csv('D3heat_COP_temp.csv', delimiter=',', index_col=0)
# D4_heat = pd.read_csv('D4heat_COP_temp.csv', delimiter=',', index_col=0)
# D5_heat = pd.read_csv('D5heat_COP_temp.csv', delimiter=',', index_col=0)
# HL_heat = pd.read_csv('HLAB_heating_temp_COPs.csv', delimiter=',', names=col_names)

#%% Process data, combine all cooling points
temp1_cool = np.tile(D1_cool[:,0],reps=4)
cops1 = np.concatenate((D1_cool[:,1],D1_cool[:,2],D1_cool[:,3],D1_cool[:,4]), axis=0)
D1_cool_data = np.column_stack((temp1_cool,cops1))

temp2_cool = np.tile(D2_cool[:,0],reps=5)
cops2 = np.concatenate((D2_cool[:,1],D2_cool[:,2],D2_cool[:,3],D2_cool[:,4],D2_cool[:,5]), axis=0)
D2_cool_data = np.column_stack((temp2_cool,cops2))

temp3_cool = np.tile(D3_cool[:,0],reps=2)
cops3 = np.concatenate((D3_cool[:,1],D3_cool[:,2]), axis=0)
D3_cool_data = np.column_stack((temp3_cool,cops3))

temp4_cool = np.tile(D4_cool[:,0],reps=3)
cops4 = np.concatenate((D4_cool[:,1],D4_cool[:,2],D4_cool[:,3]), axis=0)
D4_cool_data = np.column_stack((temp4_cool,cops4))

temp5_cool = np.tile(D5_cool[:,0],reps=3)
cops5 = np.concatenate((D5_cool[:,1],D5_cool[:,2],D5_cool[:,3]), axis=0)
D5_cool_data = np.column_stack((temp5_cool,cops5))

cool_data = np.concatenate((D1_cool_data, D2_cool_data, D3_cool_data, D4_cool_data,
                           D5_cool_data, HL_cool),axis=0)

#%% Process data, combine all heating points
temp1_heat = np.tile(D1_heat[:,0],reps=4)
cops1 = np.concatenate((D1_heat[:,1],D1_heat[:,2],D1_heat[:,3],D1_heat[:,4]), axis=0)
D1_heat_data = np.column_stack((temp1_heat,cops1))

temp2_heat = np.tile(D2_heat[:,0],reps=5)
cops2 = np.concatenate((D2_heat[:,1],D2_heat[:,2],D2_heat[:,3],D2_heat[:,4],D2_heat[:,5]), axis=0)
D2_heat_data = np.column_stack((temp2_heat,cops2))

temp3_heat = np.tile(D3_heat[:,0],reps=2)
cops3 = np.concatenate((D3_heat[:,1],D3_heat[:,2]), axis=0)
D3_heat_data = np.column_stack((temp3_heat,cops3))

temp4_heat = np.tile(D4_heat[:,0],reps=3)
cops4 = np.concatenate((D4_heat[:,1],D4_heat[:,2],D4_heat[:,3]), axis=0)
D4_heat_data = np.column_stack((temp4_heat,cops4))

temp5_heat = np.tile(D5_heat[:,0],reps=3)
cops5 = np.concatenate((D5_heat[:,1],D5_heat[:,2],D5_heat[:,3]), axis=0)
D5_heat_data = np.column_stack((temp5_heat,cops5))

heat_data = np.concatenate((D1_heat_data, D2_heat_data, D3_heat_data, D4_heat_data,
                           D5_heat_data, HL_heat),axis=0)

#%%  
coeff_c_R410a_F = [-0.0335,-0.0372,-0.0343,-0.0559,-0.0631,-0.0531,-0.0610,-0.0528,-0.0528,-0.0451,-0.0567,-0.0683,-0.0639,-0.0639,-0.0667,-0.0729,-0.0641,-0.0583]
coeff_c_R410a_F  = np.array(coeff_c_R410a_F)
coeff_c_R410a_C = coeff_c_R410a_F * (9/5)
coeff_c_R410a_C = coeff_c_R410a_C.tolist()
names_c_R410a = ["D1_1","D1_2","D1_3","D1_4","D2_1","D2_2","D2_3","D2_4","D2_5","HL","D3_1","D3_2","D4_1","D4_2","D4_3","D5_1","D5_2","D5_3"]
intrcpts_c_R410a_F = [6.638,7.446,6.886,8.532,10.298,8.661,9.865,8.598,8.538,7.313,8.558,9.753,9.662,9.667,10.083,11.2865,9.9430,9.118]
intrcpts_c_R410a_F  = np.array(intrcpts_c_R410a_F)
intrcpts_c_R410a_C = intrcpts_c_R410a_F + (160/5) * coeff_c_R410a_F
intrcpts_c_R410a_C = intrcpts_c_R410a_C.tolist()
coeff_h_R410a_F = [0.0379, 0.0512, 0.0419, 0.0397, 0.0354, 0.0348,0.0323, 0.0296, 0.0273, 0.03650576,0.0387,0.0343,0.0228,0.03442,0.0270,0.0396,0.0355,0.0350]
coeff_h_R410a_F  = np.array(coeff_h_R410a_F)
coeff_h_R410a_C = coeff_h_R410a_F * (9/5)
coeff_h_R410a_C = coeff_h_R410a_C.tolist()
names_h_R410a = ["D1_1_h","D1_2_h","D1_3_h","D1_4_h","D2_1_h","D2_2_h","D2_3_h","D2_4_h","D2_5_h","HL","D3_1_h","D3_2_h","D4_1_h","D4_2_h","D4_3_h","D5_1_h","D5_2_h","D5_3_h"]
intrcpts_h_R410a_F = [1.864,1.531,1.702,1.723,2.953,2.879,2.638,2.445,2.214, 1.0846564669825693,2.2404,2.0080,1.7482,1.9978,1.9857,2.7962,2.5103,2.4816]
intrcpts_h_R410a_F  = np.array(intrcpts_h_R410a_F)
intrcpts_h_R410a_C = intrcpts_h_R410a_F + (160/5) * coeff_h_R410a_F
intrcpts_h_R410a_C = intrcpts_h_R410a_C.tolist()

#%%
coeff_c_ave_C = np.mean(coeff_c_R410a_C) # -0.10036
coeff_h_ave_C = np.mean(coeff_h_R410a_C) # 0.0633925
intrcpts_c_ave_C = np.mean(intrcpts_c_R410a_C) # 7.15168
intrcpts_h_ave_C = np.mean(intrcpts_h_R410a_C) # 3.28264

# coeff_c_ave_F = np.mean(coeff_c_R410a_F) # -0.05575555555555556
# coeff_h_ave_F = np.mean(coeff_h_R410a_F) # 0.03521809777777778
# intrcpts_c_ave_F = np.mean(intrcpts_c_R410a_F) # 8.93586111111111
# intrcpts_h_ave_F = np.mean(intrcpts_h_R410a_F) # 2.155658692610143

#%% 
# x_cool = np.concatenate((temp1_cool,temp2_cool,temp3_cool,temp4_cool,temp5_cool),axis=0)
# x_heat = np.concatenate((temp1_heat,temp2_heat,temp3_heat,temp4_heat,temp5_heat),axis=0)
# estimates_cool = coeff_c_ave_F * cool_data[:,0] + intrcpts_c_ave_F
# estimates_heat =  coeff_h_ave_F * heat_data[:,0] + intrcpts_h_ave_F
cool_data_C = cool_data
cool_data_C[:,0] = (cool_data_C[:,0] - 32) * (5/9)
heat_data_C = heat_data
heat_data_C[:,0] = (heat_data_C[:,0] - 32) * (5/9)
estimates_cool = coeff_c_ave_C * cool_data_C[:,0] + intrcpts_c_ave_C
estimates_heat =  coeff_h_ave_C * heat_data_C[:,0] + intrcpts_h_ave_C

def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred)) # rss
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

rrmse_cool = relative_root_mean_squared_error(cool_data[:,1],estimates_cool) # 24.74%
rrmse_heat = relative_root_mean_squared_error(heat_data[:,1],estimates_heat) # 38.46%

#%% 
def CI_slope(slope,true,pred,x,alpha,n,p):
    rse_2 = np.sum(np.square(true-pred)) / (n-2)
    x_bar = np.mean(x)
    se = np.sqrt(rse_2 * (1/n + x_bar**2 / np.sum(np.square(x-x_bar))))
    t = scipy.stats.t.isf(alpha / 2, n - p - 1)
    bound1 = slope + t*se
    bound2 = slope - t*se
    return bound1, bound2

def CI_intrcpt(intrcpt,true,pred,x,alpha,n,p):
    rse_2 = np.sum(np.square(true-pred)) / (n-2)
    x_bar = np.mean(x)
    se = np.sqrt(rse_2 / np.sum(np.square(x-x_bar)))
    t = scipy.stats.t.isf(alpha / 2, n - p - 1)
    bound1 = intrcpt + t*se
    bound2 = intrcpt - t*se
    return bound1, bound2

cool_slope_CI = CI_slope(coeff_c_ave_C,cool_data_C[:,1],estimates_cool,cool_data_C[:,0],0.05,3328,1)

cool_intrcpt_CI = CI_intrcpt(intrcpts_c_ave_C,cool_data_C[:,1],estimates_cool,cool_data_C[:,0],0.05,3328,1)

heat_slope_CI = CI_slope(coeff_h_ave_C,heat_data_C[:,1],estimates_heat,heat_data_C[:,0],0.05,6090,1)

heat_intrcpt_CI = CI_intrcpt(intrcpts_h_ave_C,heat_data_C[:,1],estimates_heat,heat_data_C[:,0],0.05,6090,1)
