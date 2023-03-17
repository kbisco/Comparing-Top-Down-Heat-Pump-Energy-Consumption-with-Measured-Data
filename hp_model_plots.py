# Created by Mohammad Rezqalla

import matplotlib.pyplot as plt
import numpy as np

T_ref = 18.333
x = np.linspace(T_ref,(120-32)*5/9)
x_heating = np.linspace((-20-32)*5/9,T_ref)

# these are with only temp (and not deltaT) as the dependent variable
coeff_c_R410a_F = [-0.0335,-0.0372,-0.0343,-0.0559,-0.0631,-0.0531,-0.0610,-0.0528,-0.0528,-0.0451,-0.0567,-0.0683,-0.0639,-0.0639,-0.0667,-0.0729,-0.0641,-0.0583]
coeff_c_R410a_F  = np.array(coeff_c_R410a_F)
coeff_c_R410a_C = coeff_c_R410a_F * (9/5)
coeff_c_R410a_C = coeff_c_R410a_C.tolist()
# names_c_R410a = ["D1","D2","D3","D4","D2_1","D2_2","D2_3","D2_4","D2_5","HL","D3_1","D3_2","D4_1","D4_2","D4_3","D5_1","D5_2","D5_3"]
names_c_R410a = ["D1_1","D1_2","D1_3","D1_4","D2_1","D2_2","D2_3","D2_4","D2_5","HL","D3_1","D3_2","D4_1","D4_2","D4_3","D5_1","D5_2","D5_3"]
intrcpts_c_R410a_F = [6.638,7.446,6.886,8.532,10.298,8.661,9.865,8.598,8.538,7.313,8.558,9.753,9.662,9.667,10.083,11.2865,9.9430,9.118]
intrcpts_c_R410a_F  = np.array(intrcpts_c_R410a_F)
intrcpts_c_R410a_C = intrcpts_c_R410a_F + (160/5) * coeff_c_R410a_F
intrcpts_c_R410a_C = intrcpts_c_R410a_C.tolist()
coeff_h_R410a_F = [0.0379, 0.0512, 0.0419, 0.0397, 0.0354, 0.0348,0.0323, 0.0296, 0.0273, 0.03650576,0.0387,0.0343,0.0228,0.03442,0.0270,0.0396,0.0355,0.0350]
coeff_h_R410a_F  = np.array(coeff_h_R410a_F)
coeff_h_R410a_C = coeff_h_R410a_F * (9/5)
coeff_h_R410a_C = coeff_h_R410a_C.tolist()
# names_h_R410a = ["D1_h","D2_h","D3_h","D4_h","D2_1_h","D2_2_h","D2_3_h","D2_4_h","D2_5_h","HL","D3_1_h","D3_2_h","D4_1_h","D4_2_h","D4_3_h","D5_1_h","D5_2_h","D5_3_h"]
names_h_R410a = ["D1_1_h","D1_2_h","D1_3_h","D1_4_h","D2_1_h","D2_2_h","D2_3_h","D2_4_h","D2_5_h","HL","D3_1_h","D3_2_h","D4_1_h","D4_2_h","D4_3_h","D5_1_h","D5_2_h","D5_3_h"]
intrcpts_h_R410a_F = [1.864,1.531,1.702,1.723,2.953,2.879,2.638,2.445,2.214, 1.0846564669825693,2.2404,2.0080,1.7482,1.9978,1.9857,2.7962,2.5103,2.4816]
intrcpts_h_R410a_F  = np.array(intrcpts_h_R410a_F)
intrcpts_h_R410a_C = intrcpts_h_R410a_F + (160/5) * coeff_h_R410a_F
intrcpts_h_R410a_C = intrcpts_h_R410a_C.tolist()

coeff_c_ave_C = np.mean(coeff_c_R410a_C) # -0.10036
coeff_h_ave_C = np.mean(coeff_h_R410a_C) # 0.0633925
intrcpts_c_ave_C = np.mean(intrcpts_c_R410a_C) # 7.15168
intrcpts_h_ave_C = np.mean(intrcpts_h_R410a_C) # 3.28264

coeff_c_ave_F = np.mean(coeff_c_R410a_F) # -0.05575555555555556
coeff_h_ave_F = np.mean(coeff_h_R410a_F) # 0.03521809777777778
intrcpts_c_ave_F = np.mean(intrcpts_c_R410a_F) # 8.93586111111111
intrcpts_h_ave_F = np.mean(intrcpts_h_R410a_F) # 2.155658692610143

#%%
hp_names = ['DZ20VC + 0241B', 'DZ20VC + 0361B', 'DZ20VC + 0481B', 'DZ20VC + 0601C', # D1
            'FTXS09LVJU', 'FTXS12LVJU', 'FTXS15LVJU', 'FTXS18LVJU', 'FTXS24LVJU', # D2
            '25VNA', # HLAB heat pump
            'D3_1', 'D3_2', 
            'FTX09NMVJUA', 'FTX12NMVJUA', 'FTX15NMVJUA',
            'FTXG09HVJU', 'FTXG12HVJU', 'FTXG15HVJU'
            ]
colors = ["","","","","","","","","","","","","","","","","","",]

#%%
# plt.figure(1)
fig, ax = plt.subplots(figsize= (14, 8.5))
for (c,i,name) in zip(coeff_c_R410a_C, intrcpts_c_R410a_C,names_c_R410a):
    k = x * c + i
    ax.plot(x,k)
plt.gca().set_prop_cycle(None)
for (c,i,name) in zip(coeff_h_R410a_C, intrcpts_h_R410a_C,names_h_R410a):
    k = x_heating * c + i
    ax.plot(x_heating,k)
    
# ax.title("R410A Heat Pumps")
ax.set_xlabel("Temperature (Celsius)", fontsize=15)
ax.set_ylabel("COP", fontsize=15)
ax.grid(alpha=0.15)
plt.savefig('R410a_hpmodels_plot.eps')
plt.savefig('R410a_hpmodels_plot.png', dpi=600)

#%%
endpts = np.zeros((len(names_c_R410a),1)) # Get COP at temp endpoint for each heat pump
for count, value in enumerate(coeff_c_R410a_C):
    endpts[count] = x[-1]*value + intrcpts_c_R410a_C[count]
    
fig2, ax2 = plt.subplots(figsize= (15,8.5))
ax2.set_yticks([y for y in np.arange(0, 7)])
ax2.set_yticklabels(
    [y for y in np.arange(0, 7)], 
    fontsize=16,
)

ax2.set_xticks([x for x in np.arange(-30, 55, 10)])
ax2.set_xticklabels(
    [x for x in np.arange(-30, 55, 10)], 
    fontsize=16,
)
    
for (c,i,name) in zip(coeff_c_R410a_C, intrcpts_c_R410a_C, names_c_R410a):
    k = x * c + i
    if(name != "HL"):
        ax2.plot(x,k)
    else:
        ax2.plot(x,k,linewidth=3.0, ls='dashed')
        # plt.plot(x,k,linewidth=3.0,linestyle='dashed')
plt.gca().set_prop_cycle(None)
for (c,i,name) in zip(coeff_h_R410a_C, intrcpts_h_R410a_C, names_h_R410a):
    k = x_heating * c + i
    if (name != "HL"):
        ax2.plot(x_heating,k)
    else:
        # ax2.plot(x_heating,k,linewidth=3.0,linestyle='dashed')
        ax2.plot(x_heating,k,linewidth=3.0, ls='dashed')
        
plt.gca().set_prop_cycle(None)
ax2.set_xlim(-30,72)
ax2.set_ylim(0, 7)

# Least value endpt COP (0) to highest (17)
LABEL_Y = [
    3.32, # 14
    3.86, # 17
    3.68, # 16
    1.16, # 2
    3.5, # 15
    2.78, # 11
    3.14, # 13
    2.6, # 10
    2.24, # 8
    1.34, # 3
    0.98, # 1
    0.8, # 0
    1.52, # 4
    1.7, # 5
    1.88, # 6
    2.96, # 12
    2.42, # 9
    2.06 # 7
    ]
# LABEL_Y = np.linspace(1, 3.5, num=18)
x_start = x[-1]
x_end = 55
PAD = 0.1

for count, value in enumerate(hp_names):
    text = value
    y_start = endpts[count]
    y_end = LABEL_Y[count]
    
    ax2.plot([x_start, (x_start + x_end - PAD) / 2 , x_end - PAD], 
        [y_start, y_end, y_end],
        alpha = 0.5,
        ls = 'dashed'
        )
    
    ax2.text(x_end, 
        y_end, 
        text, 
        fontsize=14,
        weight='bold',
        va='center'
        )
# ax2.set_xticklabels(fontsize=13)
ax2.set_xlabel("Temperature (Celsius)", fontsize=18)
ax2.set_ylabel("COP", fontsize=18)
ax2.grid(alpha=0.15)
# fig2.savefig('hp_model_labeled.png', dpi=600)
fig2.savefig('hp_model_labeled.tiff', dpi=600, bbox_inches='tight')
#%%
spacing = np.linspace(0.8,4.04)
nums = np.arange(0,18) 
endpts_names = np.column_stack((np.array(hp_names),endpts,nums))
endpts_names_sort = endpts_names[endpts_names[:,1].argsort()]
