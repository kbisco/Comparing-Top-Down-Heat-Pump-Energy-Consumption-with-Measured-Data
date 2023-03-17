#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:01:28 2023

@author: kbiscoch
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:58:12 2023

@author: kbiscoch
"""
import os
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression

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

def runRegModels(inData, fData, cData, c_x, f_x):
    for n in range(inData.shape[1]-1):
        lmF = LinearRegression().fit(inData[:,0].reshape(-1,1), inData[:,n+1].reshape(-1,1))
        fData = np.append(fData, [np.array([lmF.intercept_[0], lmF.coef_[0,0]])], axis=0)
        lmFpred = inData[:,0]*lmF.coef_ + lmF.intercept_
        lmC = LinearRegression().fit(((inData[:,0]-32)*5/9).reshape(-1,1), inData[:,n+1].reshape(-1,1))
        cData = np.append(cData, [np.array([lmC.intercept_[0], lmC.coef_[0,0]])], axis=0)
    return (fData, cData)

fDatacool = np.zeros([1,2])
cDatacool = np.zeros((1,2))
c_x = np.arange(-10, 18.3, 10)
f_x = np.arange(-10*9/5+32,65,10)
fDatacool, cDatacool = runRegModels(D1_cool, fDatacool, cDatacool, c_x, f_x)
fDatacool, cDatacool = runRegModels(D2_cool, fDatacool, cDatacool, c_x, f_x)
fDatacool, cDatacool = runRegModels(D3_cool, fDatacool, cDatacool, c_x, f_x)
fDatacool, cDatacool = runRegModels(D4_cool, fDatacool, cDatacool,c_x, f_x)
fDatacool, cDatacool = runRegModels(D5_cool, fDatacool, cDatacool, c_x, f_x)
fDatacool, cDatacool = runRegModels(HL_cool, fDatacool, cDatacool, c_x, f_x)
fDatacool = np.delete(fDatacool,0,0)
cDatacool = np.delete(cDatacool,0,0)
print(np.average(fDatacool, axis=0))
print(np.average(cDatacool, axis=0))

fDataheat = np.zeros([1,2])
cDataheat = np.zeros((1,2))
c_x_h = np.arange(18.3, 38, 10)
f_x_h = np.arange(65, 100, 10)
fDataheat, cDataheat = runRegModels(D1_heat, fDataheat, cDataheat, c_x_h, f_x_h)
fDataheat, cDataheat = runRegModels(D2_heat, fDataheat, cDataheat, c_x_h, f_x_h)
fDataheat, cDataheat = runRegModels(D3_heat, fDataheat, cDataheat, c_x_h, f_x_h)
fDataheat, cDataheat = runRegModels(D4_heat, fDataheat, cDataheat,c_x_h, f_x_h)
fDataheat, cDataheat = runRegModels(D5_heat, fDataheat, cDataheat, c_x_h, f_x_h)
fDataheat, cDataheat = runRegModels(HL_heat, fDataheat, cDataheat, c_x_h, f_x_h)
fDataheat = np.delete(fDataheat,0,0)
cDataheat = np.delete(cDataheat,0,0)
print(np.average(fDataheat, axis=0))
print(np.average(cDataheat, axis=0))

#%% 
# x_cool = np.concatenate((temp1_cool,temp2_cool,temp3_cool,temp4_cool,temp5_cool),axis=0)
# x_heat = np.concatenate((temp1_heat,temp2_heat,temp3_heat,temp4_heat,temp5_heat),axis=0)
# estimates_cool = coeff_c_ave_F * cool_data[:,0] + intrcpts_c_ave_F
# estimates_heat =  coeff_h_ave_F * heat_data[:,0] + intrcpts_h_ave_F
cool_data_C = cool_data.copy()
cool_data_C[:,0] = (cool_data_C[:,0] - 32) * (5/9)
heat_data_C = heat_data.copy()
heat_data_C[:,0] = (heat_data_C[:,0] - 32) * (5/9)
# estimates_cool = coeff_c_ave_C * cool_data_C[:,0] + intrcpts_c_ave_C
# estimates_heat =  coeff_h_ave_C * heat_data_C[:,0] + intrcpts_h_ave_C

coeffs_c_ave_C = np.average(cDatacool,axis=0) # [intercept, slope]
coeffs_h_ave_C = np.average(cDataheat,axis=0)
estimates_cool = coeffs_c_ave_C[1] * cool_data_C[:,0] + coeffs_c_ave_C[0]
estimates_heat = coeffs_h_ave_C[1] * heat_data_C[:,0] + coeffs_h_ave_C[0]

def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred)) # rss
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

rrmse_cool = relative_root_mean_squared_error(cool_data[:,1],estimates_cool) # 16.81%
rrmse_heat = relative_root_mean_squared_error(heat_data[:,1],estimates_heat) # 38.17%
print(rrmse_cool)
print(rrmse_heat)
#%% 
def CI_slope(slope,true,pred,x,alpha,n,p):
    rse_2 = np.sum(np.square(true-pred)) / (n-2) # cool: 0.498295849, heat: 1.0993
    x_bar = np.mean(x) # cool: 33.781, heat: -9.568
    se = np.sqrt(rse_2 * (1/n + x_bar**2 / np.sum(np.square(x-x_bar)))) # cool: 0.06685, heat: 0.01874
    t = scipy.stats.t.isf(alpha / 2, n - p - 1) # 1.96
    bound1 = slope + t*se # cool: 0.0307, heat: 0.1001297
    bound2 = slope - t*se # cool: -0.231, heat: 0.02665
    return bound1, bound2

def CI_intrcpt(intrcpt,true,pred,x,alpha,n,p):
    rse_2 = np.sum(np.square(true-pred)) / (n-2)
    x_bar = np.mean(x)
    se = np.sqrt(rse_2 / np.sum(np.square(x-x_bar)))
    t = scipy.stats.t.isf(alpha / 2, n - p - 1)
    bound1 = intrcpt + t*se
    bound2 = intrcpt - t*se
    return bound1, bound2

cool_slope_CI = CI_slope(coeffs_c_ave_C[1],cool_data_C[:,1],estimates_cool,cool_data_C[:,0],0.05,3280,1)

cool_intrcpt_CI = CI_intrcpt(coeffs_c_ave_C[0],cool_data_C[:,1],estimates_cool,cool_data_C[:,0],0.05,3280,1)

heat_slope_CI = CI_slope(coeffs_h_ave_C[1],heat_data_C[:,1],estimates_heat,heat_data_C[:,0],0.05,6138,1)

heat_intrcpt_CI = CI_intrcpt(coeffs_h_ave_C[0],heat_data_C[:,1],estimates_heat,heat_data_C[:,0],0.05,6138,1)

print(cool_slope_CI)
print(cool_intrcpt_CI)
print(heat_slope_CI)
print(heat_intrcpt_CI)
#%%
def confInts(fMod, cMod, true, alpha, p):
    true_c = (true[:,0]-32)*5/9
    fModpred = fMod[0] + true[:,0]*fMod[1]
    cModpred = cMod[0] + true_c*cMod[1]
    rse_2F = np.sum(np.square(true[:,1] - fModpred))/(true.shape[0]-2)
    rse_2C = np.sum(np.square(true[:,1] - cModpred))/(true.shape[0]-2)
    # print(rse_2C) # cool: 0.4083
    x_barF = np.mean(true[:,0])
    x_barC = np.mean((true[:,0]-32)*5/9)
    # print(x_barC) # cool: 33.781
    seFs = np.sqrt(rse_2F * (1/true.shape[0] + x_barF**2 / np.sum(np.square(true[:,0]-x_barF))))
    seCs = np.sqrt(rse_2C * (1/true.shape[0] + x_barC**2 / np.sum(np.square(((true[:,0]-32)*5/9)-x_barC))))
    # print(seCs) # cool: 0.06685
    seFi = np.sqrt(rse_2F / (np.sum(np.square(true[:,0]-x_barF))))
    seCi = np.sqrt(rse_2C / np.sum(np.square(((true[:,0]-32)*5/9)-x_barC)))
    t = scipy.stats.t.isf(alpha / 2, true.shape[0] - p - 1)
    bound1Fs = fMod[1] + t*seFs
    bound2Fs = fMod[1] - t*seFs
    bound1Cs = cMod[1] + t*seCs
    bound2Cs = cMod[1] - t*seCs
    bound1Fi = fMod[0] + t*seFi
    bound2Fi = fMod[0] - t*seFi
    bound1Ci = cMod[0] + t*seCi
    bound2Ci = cMod[0] - t*seCi
    outBoundsF = [bound1Fi, bound2Fi, bound1Fs, bound2Fs]
    outBoundsC = [bound1Ci, bound2Ci, bound1Cs, bound2Cs]
    return outBoundsF, outBoundsC

b1, b20 = confInts(np.average(fDatacool, axis=0), np.average(cDatacool, axis=0), cool_data, 0.05, 1)
print(b2, np.average(cDatacool, axis=0))

b1, b2 = confInts(np.average(fDataheat, axis=0), np.average(cDataheat, axis=0), heat_data, 0.05, 1)
print(b2, np.average(cDataheat, axis=0))