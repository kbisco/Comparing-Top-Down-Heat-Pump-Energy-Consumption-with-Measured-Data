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
coeff_h_R410a_F = [0.0379, 0.0512, 0.0419, 0.0397, 0.0354, 0.0348,0.0323, 0.0296,0.0273,0.0427,0.0387,0.0343,0.0228,0.03442,0.0270,0.0396,0.0355,0.0350]
coeff_h_R410a_F  = np.array(coeff_h_R410a_F)
coeff_h_R410a_C = coeff_h_R410a_F * (9/5)
coeff_h_R410a_C = coeff_h_R410a_C.tolist()
# names_h_R410a = ["D1_h","D2_h","D3_h","D4_h","D2_1_h","D2_2_h","D2_3_h","D2_4_h","D2_5_h","HL","D3_1_h","D3_2_h","D4_1_h","D4_2_h","D4_3_h","D5_1_h","D5_2_h","D5_3_h"]
names_h_R410a = ["D1_1_h","D1_2_h","D1_3_h","D1_4_h","D2_1_h","D2_2_h","D2_3_h","D2_4_h","D2_5_h","HL","D3_1_h","D3_2_h","D4_1_h","D4_2_h","D4_3_h","D5_1_h","D5_2_h","D5_3_h"]
intrcpts_h_R410a_F = [1.864,1.531,1.702,1.723,2.953,2.879,2.638,2.445,2.214,1.0896,2.2404,2.0080,1.7482,1.9978,1.9857,2.7962,2.5103,2.4816]
intrcpts_h_R410a_F  = np.array(intrcpts_h_R410a_F)
intrcpts_h_R410a_C = intrcpts_h_R410a_F + (160/5) * coeff_h_R410a_F
intrcpts_h_R410a_C = intrcpts_h_R410a_C.tolist()

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

# LABEL_Y = [
#     3.38, # D1_1
#     4.06, # D1_2
#     3.25, # D1_3
#     1.34, # D1_4
#     3.55, # D2_1
#     2.87, # D2_2
#     3.21, # D2_3
#     2.7, # D2_4
#     2.36, # D2_5
#     1.51, # HL
#     1.17, # D3_1
#     1.0, # D3_2
#     1.68, # D4_1
#     1.85, # D4_2
#     2.02, # D4_3
#     3.04, # D5_1
#     2.53, # D5_2
#     2.19 # D5_3
#     ]
LABEL_Y = np.linspace(np.amin(endpts), np.amax(endpts), num=18)
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
ax2.set_xlabel("Temperature (Celsius)", fontsize=15)
ax2.set_ylabel("COP", fontsize=15)
ax2.grid(alpha=0.15)

#%%
endpts_names = np.column_stack((np.array(hp_names),endpts))
endpts_names_sort = endpts_names[endpts_names[:,1].argsort()]
