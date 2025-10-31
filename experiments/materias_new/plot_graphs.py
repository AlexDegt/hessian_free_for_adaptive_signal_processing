import os, sys
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'cm'

lc_mnm_damped = np.load("lc_qcrit_train_mnm_damped.npy")
lc_sgd_adam = np.load("lc_train_sgd_auto.npy")
lc_conj_grad_1_iter = np.load("lc_qcrit_train_conj_grad_1_iter.npy")
lc_conj_grad_5_iter = np.load("lc_qcrit_train_conj_grad_5_iter.npy")
lc_conj_grad_10_iter = np.load("lc_qcrit_train_conj_grad_10_iter.npy")
lc_conj_grad_15_iter = np.load("lc_qcrit_train_conj_grad_15_iter.npy")
lc_conj_grad_20_iter = np.load("lc_qcrit_train_conj_grad_20_iter.npy")
lc_conj_grad_25_iter = np.load("lc_qcrit_train_conj_grad_25_iter.npy")
lc_conj_grad_30_iter = np.load("lc_qcrit_train_conj_grad_30_iter.npy")
lc_conj_grad_35_iter = np.load("lc_qcrit_train_conj_grad_35_iter.npy")
lc_conj_grad_40_iter = np.load("lc_qcrit_train_conj_grad_40_iter.npy")
lc_conj_grad_45_iter = np.load("lc_qcrit_train_conj_grad_45_iter.npy")
lc_conj_grad_50_iter = np.load("lc_qcrit_train_conj_grad_50_iter.npy")
lc_conj_grad_59_iter = np.load("lc_qcrit_train_conj_grad_59_iter.npy")

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

legend = ["Mixed Newton method", "SGD+Adam+linear scheduler",
          "Conj. grad. 1 iter.", 
          "Conj. grad. 5 iter.", 
          "Conj. grad. 10 iter.",
          "Conj. grad. 20 iter.",
          "Conj. grad. 30 iter.",
          "Conj. grad. 50 iter."]

x = np.arange(len(lc_mnm_damped))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(10,9))

# --- Верхний график ---
ax1.plot(x, lc_mnm_damped, color=colors[0])
ax1.plot(x, lc_sgd_adam, color=colors[1])
ax1.plot(x, lc_conj_grad_1_iter, color=colors[2])
ax1.plot(x, lc_conj_grad_5_iter, color=colors[3])
ax1.plot(x, lc_conj_grad_10_iter, color=colors[4])
ax1.plot(x, lc_conj_grad_20_iter, color=colors[5])
ax1.plot(x, lc_conj_grad_30_iter, color=colors[6])
ax1.plot(x, lc_conj_grad_50_iter, color=colors[7])
ax1.set_xlim([-0.5, 21])
ax1.set_ylim([-55, 30])
ax1.set_xticks(np.arange(0, 22, 2))
ax1.legend(legend, fontsize=fontsize, loc="upper right")
ax1.grid()

# --- Нижний график ---
ax2.plot(x, lc_mnm_damped, color=colors[0])
ax2.plot(x, lc_sgd_adam, color=colors[1])
ax2.plot(x, lc_conj_grad_1_iter, color=colors[2])
ax2.plot(x, lc_conj_grad_5_iter, color=colors[3])
ax2.plot(x, lc_conj_grad_10_iter, color=colors[4])
ax2.plot(x, lc_conj_grad_20_iter, color=colors[5])
ax2.plot(x, lc_conj_grad_30_iter, color=colors[6])
ax2.plot(x, lc_conj_grad_50_iter, color=colors[7])
ax2.set_xlim([-1, 1002])
ax2.set_ylim([-53, -30])
ax2.set_xticks(np.arange(0, 1100, 100))
ax2.set_xlabel("Эпохи", fontsize=fontsize)
ax2.grid()

# Общий Y-label
fig.text(0.01, 0.5, 'NMSE, dB', va='center', rotation='vertical', fontsize=fontsize)

plt.tight_layout()
plt.show()