import os, sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

lc_mnm = np.load("lc_qcrit_test_mnm.npy")
lc_sgd_adam = np.load("lc_test_sgd_auto.npy")

lc_conj_grad_1_iter = np.load("lc_qcrit_test_cg_iter_1.npy")
lc_conj_grad_5_iter = np.load("lc_qcrit_test_cg_iter_5.npy")
lc_conj_grad_10_iter = np.load("lc_qcrit_test_cg_iter_10.npy")
lc_conj_grad_20_iter = np.load("lc_qcrit_test_cg_iter_20.npy")
lc_conj_grad_30_iter = np.load("lc_qcrit_test_cg_iter_30.npy")
lc_conj_grad_50_iter = np.load("lc_qcrit_test_cg_iter_50.npy")

block_per_epoch = 1316

curves = [lc_mnm, lc_sgd_adam, lc_conj_grad_1_iter, lc_conj_grad_5_iter, 
          lc_conj_grad_10_iter, lc_conj_grad_20_iter, lc_conj_grad_30_iter,
          lc_conj_grad_50_iter]

# Leave number of iterations multiple of a number of block per epoch
for j, curve in enumerate(curves):
    curves[j] = curve[:block_per_epoch * (len(curve) // block_per_epoch)]

curve_len = max([len(curve) for curve in curves])

epoch_num = curve_len // block_per_epoch
for j, curve in enumerate(curves):
    curr_epoch_num = len(curve) // block_per_epoch
    curve = curve.tolist()
    curve.extend([curve[-1]] * (epoch_num - curr_epoch_num) * block_per_epoch)
    curves[j] = np.array(curve)
    
colors = [
    'black',       # чёрный
    '#e41a1c',     # ярко-красный
    '#377eb8',     # насыщенный синий
    '#4daf4a',     # яркий зелёный
    '#ff7f00',     # насыщенный оранжевый
    '#984ea3',     # яркий фиолетовый
    '#f781bf',     # ярко-розовый
    '#a65628'      # насыщенный коричнево-рыжий
]

fontsize=13

legend = ["Mixed Newton method", "BGD+Adam+linear scheduler",
          "Conj. grad. 1 iter.", 
          "Conj. grad. 5 iter.", 
          "Conj. grad. 10 iter.",
          "Conj. grad. 20 iter.",
          "Conj. grad. 30 iter.",
          "Conj. grad. 50 iter."]

x = np.arange(len(curves[0]))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(10,10))

# plt.legend(legend, fontsize=fontsize, loc="upper right")

# --- Upper plot ---
for j, curve in enumerate(curves):
    ax1.plot(x, curve, color=colors[j])
ax1.set_xlim([-40, 4000])
ax1.set_ylim([-55, 0])
ax1.set_xticks(np.arange(0, 4000, 200))
ax1.legend(legend, fontsize=fontsize - 1, loc="upper right")
ax1.grid()

# --- Middle plot ---
for j, curve in enumerate(curves):
    ax2.plot(x, curve, color=colors[j])
ax2.set_xlim([-3000, 200000])
ax2.set_ylim([-53, -30])
ax2.set_xticks(np.arange(0, 200000, 20000))
ax2.grid()

# --- Lower plot ---
for j, curve in enumerate(curves):
    ax3.plot(x, curve, color=colors[j])
ax3.set_xlim([-100000, 6600000])
ax3.set_ylim([-53, -30])
ax3.set_xticks(np.arange(0, 6600000, 400000))
ax3.set_xlabel("Номер отсчёта", fontsize=fontsize)
ax3.grid()

# Общий Y-label
fig.text(0.01, 0.5, 'NMSE, dB', va='center', rotation='vertical', fontsize=fontsize)

plt.tight_layout()
plt.show()

# Calculate complexity
block = 60

gd = 59 * block
mnm = 59 * block + 59 ** 2 * block + 59 ** 3
cg_1_iter = 59 * block + 59 ** 2 * block + 1 * 59 ** 2
cg_5_iter = 59 * block + 59 ** 2 * block + 5 * 59 ** 2
cg_10_iter = 59 * block + 59 ** 2 * block + 10 * 59 ** 2
cg_20_iter = 59 * block + 59 ** 2 * block + 20 * 59 ** 2
cg_30_iter = 59 * block + 59 ** 2 * block + 30 * 59 ** 2
cg_50_iter = 59 * block + 59 ** 2 * block + 50 * 59 ** 2

# gd = (51 + 2) * block
# mnm = gd + (51 + 2 * 51 + 2 * 51 + 4) * block + 59 ** 3
# cg_1_iter = 1 * gd + (51 + 2 * 51 + 2 * 51 + 4)* block + 1 * 59 ** 2
# cg_5_iter = 1 * gd+ (51 + 2 * 51 + 2 * 51 + 4) * block + 5 * 59 ** 2
# cg_10_iter = 1 * gd + (51 + 2 * 51 + 2 * 51 + 4) * block + 10 * 59 ** 2
# cg_20_iter = 1 * gd + (51 + 2 * 51 + 2 * 51 + 4) * block + 20 * 59 ** 2
# cg_30_iter = 1 * gd + (51 + 2 * 51 + 2 * 51 + 4) * block + 30 * 59 ** 2
# cg_50_iter = 1 * gd + (51 + 2 * 51 + 2 * 51 + 4) * block + 50 * 59 ** 2

print(f"gd / mnm = {gd / mnm}")
print(f"mnm / mnm = {mnm / mnm}")
print(f"cg_1_iter / mnm = {cg_1_iter / mnm}")
print(f"cg_5_iter / mnm = {cg_5_iter / mnm}")
print(f"cg_10_iter / mnm = {cg_10_iter / mnm}")
print(f"cg_20_iter / mnm = {cg_20_iter / mnm}")
print(f"cg_30_iter / mnm = {cg_30_iter / mnm}")
print(f"cg_50_iter / mnm = {cg_50_iter / mnm}")
