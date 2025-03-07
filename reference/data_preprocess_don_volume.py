"""Notes
        |      in       |     shape      |     out     |  normalization |
        |---------------|----------------|-------------|----------------|
branch  | aoa_total.npy |    (21,1 )     | aoa_inf.npy |   minmax       |
trunk   |    xyz.npy    | (50762870, 3)  |  x_inf.npy  | minmax axis=0  |
output  | data_total.npy| (21, 50762870) | y_inf.npy   |  minmax        |

branch: func
trunk: spatial
"""
import numpy as np
import os
import scipy.io as sio
import sys


NUM_GPUs = 4
INPUT_FOLDER = "/p/work1/projects/FrontierX/don_volume_data" 
OUTPUT_FOLDER = "/p/work1/projects/FrontierX/hspn/data/don_volume_data"

# Load data
aoa = np.load(f'{INPUT_FOLDER}/aoa_total.npy', allow_pickle=True) # branch
xyz = np.load(f'{INPUT_FOLDER}/xyz.npy', allow_pickle=True) # trunk
rho = np.load(f'{INPUT_FOLDER}/data_total.npy', allow_pickle=True) # outputs


aoa = aoa.reshape(-1, 1)
X_func_ls = aoa

# Normalize trunk (xyz)
X_loc_ls = xyz   
s_min = np.min(X_loc_ls, axis=0)
s_max = np.max(X_loc_ls, axis=0)
X_loc_ls = (X_loc_ls - s_min)/ (s_max - s_min) 
print(f"aoa are: {aoa.shape} and aoa: {aoa}")

print(f"mn: {np.min(X_loc_ls, axis=0)}, mx: {np.max(X_loc_ls, axis=0)}")
y_ls = rho     
print('X_func: ', X_func_ls.shape, ', min: ', np.min(X_func_ls), ', max: ', np.max(X_func_ls) )
print('X_loc: ', X_loc_ls.shape)

X_loc_ls = X_loc_ls

y_min = np.min(y_ls)
y_max = np.max(y_ls)
    
print(f"Y Min: {y_min} and Ymax: {y_max}")

# Rescale branch (aoa)
aoa_min = np.min(aoa)
aoa_max = np.max(aoa)
aoa_loc_ls = (aoa - aoa_min)/(aoa_max - aoa_min)
x_loc_ls = X_loc_ls
# Rescale output (y)
y_ls = (y_ls - y_min)/(y_max - y_min)

norm_data = {"branch_min": aoa_min, "branch_max":aoa_max, "trunk_min": s_min, "trunk_max": s_max, "out_min": y_min, "out_max":y_max }

sio.savemat(f'{OUTPUT_FOLDER}/norm_data.mat', norm_data)

ds = x_loc_ls.shape
local_sh = int(ds[0]/NUM_GPUs)
print(f"Local Size is: {local_sh}")
print(f"Y shape {y_ls.shape}")
print(f"Trunk shape {X_loc_ls.shape}")

np.save(f'{OUTPUT_FOLDER}/x_inf.npy', x_loc_ls, allow_pickle=True )
np.save(f'{OUTPUT_FOLDER}/aoa_inf.npy', aoa_loc_ls, allow_pickle=True )
np.save(f'{OUTPUT_FOLDER}/y_inf.npy', y_ls, allow_pickle=True )
#sys.exit()


rank = 0

# split by GPU
for i in range(NUM_GPUs):
    f_name = f"{OUTPUT_FOLDER}/trunk_in_" + str(i) + ".npy"
    out_name = f"{OUTPUT_FOLDER}/y_out_" + str(i) + ".npy"
    id_start = rank*local_sh
    id_end = id_start + local_sh
    print(f"id_start: {id_start}, id_end: {id_end}")
    x_local = x_loc_ls[id_start:id_end, :]
    y_local = y_ls[:, id_start:id_end]

    print(f" X local {x_local.shape}")
    print(f" Y local {y_local.shape}")

    np.save(f_name, x_local, allow_pickle=True)
    np.save(out_name, y_local, allow_pickle=True)

    rank = rank+1




    
