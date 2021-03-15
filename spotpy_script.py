from spotpy.algorithms import dds
from spot_setup_Volta_DDS import spot_setup_DDS1
from spot_setup_Volta_DDS import spot_setup_DDS2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import time

import warnings
warnings.filterwarnings("ignore")




#### BOX 2
start = time.time()
sampler = dds(spot_setup_DDS1(), dbname='DDS_ED_tot_params', dbformat='csv', parallel='mpc', save_sim=False)

rep = 1000
sampler.sample(rep, trials=1)
results=sampler.getdata()

end = time.time()
print(end - start)


#### BOX 3
results_data = np.loadtxt('Results_DDS.txt')
results_data = results_data.reshape(int(np.shape(results_data)[0]), 14, 13)

argmin = results_data[2:,0,0].argmin()

header = ['ED_tot', 'ED_cal', 'KGE_cal', 'KGE_BC_cal', 'KGE_FDC_cal', 'KGE_ACF_cal', 'KGE_Rc_cal', 'ED_val', 'KGE_val', 'KGE_BC_val', 'KGE_FDC_val', 'KGE_ACF_val', 'KGE_Rc_val' ]
df = pd.DataFrame(results_data[argmin+2, :, :], columns=header)
df





#### BOX 4
#rep=157
best_results_so_far = np.ones(rep) * 1000
# print(results_data[:, 0, 0])

for i in range(rep):
    value = results_data[i+2, 0, 0]
    if value < best_results_so_far[i]:
        best_results_so_far[i:] = value

plt.plot(best_results_so_far)
plt.ylabel('ED_tot')
plt.xlabel('Repetitions')
plt.title('DDS Optimization graph');