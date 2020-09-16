# -*- coding: utf-8 -*-
"""
- Time distribution histogram in each cell for each plane

@author: Sara Costa Faya
"""

import sys
import os
from os.path import join as join_path
import numpy as np
import matplotlib.pyplot as plt

# Root Directory of the Project
ROOT_DIR = os.path.abspath("./")

# Add ROOT_DIR to $PATH
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Read data
luptabPath = join_path(ROOT_DIR, "luptab_20200720_tristan_.txt")
luptab = np.loadtxt(luptabPath, usecols=range(8))

dataPath = join_path(ROOT_DIR, "dst_export_sara_3.txt")
data = np.loadtxt(dataPath, delimiter=',', usecols=range(186))

# new
# samplePath = join_path(ROOT_DIR, "dst_sample.txt")
# sample = np.loadtxt(samplePath, delimiter=',', usecols=range(186))


nplan = 3
nvar = 2  # time and charge
ncep = 30  # nb. of cells per plane

nrows, ncols = data.shape

# Cells position in T & Q matrix
it1cel = 6
it2cel = 66
it3cel = 126

iq1cel = 36
iq2cel = 96
iq3cel = 156

index = []

# Check the events with multiplicity 1

# nrows = 20
for i in range(nrows):

    # Assume data ordered as T1, T2, T3

    t_plane1_index = np.asarray(list(np.nonzero(data[i, it1cel:(it1cel + 30)])))
    t_plane2_index = np.asarray(list(np.nonzero(data[i, it2cel:(it2cel + 30)])))
    t_plane3_index = np.asarray(list(np.nonzero(data[i, it3cel:(it3cel + 30)])))

    q_plane1_index = np.asarray(list(np.nonzero(data[i, iq1cel:(iq1cel + 30)])))
    q_plane2_index = np.asarray(list(np.nonzero(data[i, iq2cel:(iq2cel + 30)])))
    q_plane3_index = np.asarray(list(np.nonzero(data[i, iq3cel:(iq3cel + 30)])))

    t1_size = t_plane1_index.shape
    t2_size = t_plane2_index.shape
    t3_size = t_plane3_index.shape

    q1_size = q_plane1_index.shape
    q2_size = q_plane2_index.shape
    q3_size = q_plane3_index.shape

    # Save the index of good events with multiplicity = 1

    # Events with just one track
    if t1_size[1] == 1 and t2_size[1] == 1 and t3_size[1] == 1 and \
            q1_size[1] == 1 and q2_size[1] == 1 and q3_size[1] == 1:
        index.append(i)

t_plane1 = np.array([])
t_plane2 = np.array([])
t_plane3 = np.array([])

q_plane1 = np.array([])
q_plane2 = np.array([])
q_plane3 = np.array([])

kx_plane1 = np.array([])
kx_plane2 = np.array([])
kx_plane3 = np.array([])

ky_plane1 = np.array([])
ky_plane2 = np.array([])
ky_plane3 = np.array([])

# Compute an array for each plane with kx, ky, t, Q

for i in index:
    t_plane1_index = np.asarray(list(np.nonzero(data[i, it1cel:(it1cel + 30)])))
    t_plane2_index = np.asarray(list(np.nonzero(data[i, it2cel:(it2cel + 30)])))
    t_plane3_index = np.asarray(list(np.nonzero(data[i, it3cel:(it3cel + 30)])))

    q_plane1_index = np.asarray(list(np.nonzero(data[i, iq1cel:(iq1cel + 30)])))
    q_plane2_index = np.asarray(list(np.nonzero(data[i, iq2cel:(iq2cel + 30)])))
    q_plane3_index = np.asarray(list(np.nonzero(data[i, iq3cel:(iq3cel + 30)])))

    t_plane1 = np.append(t_plane1, data[i, t_plane1_index + it1cel])
    t_plane2 = np.append(t_plane2, data[i, t_plane2_index + it2cel])
    t_plane3 = np.append(t_plane3, data[i, t_plane3_index + it3cel])

    q_plane1 = np.append(q_plane1, data[i, q_plane1_index + iq1cel])
    q_plane2 = np.append(q_plane2, data[i, q_plane2_index + iq2cel])
    q_plane3 = np.append(q_plane3, data[i, q_plane3_index + iq3cel])

    kx_plane1 = np.append(kx_plane1, luptab[t_plane1_index + 1, 4])
    ky_plane1 = np.append(ky_plane1, luptab[t_plane1_index + 1, 5])

    kx_plane2 = np.append(kx_plane2, luptab[t_plane2_index + 31, 4])
    ky_plane2 = np.append(ky_plane2, luptab[t_plane2_index + 31, 5])

    kx_plane3 = np.append(kx_plane3, luptab[t_plane3_index + 61, 4])
    ky_plane3 = np.append(ky_plane3, luptab[t_plane3_index + 61, 5])

# Matrix with kx, ky, t, Q for the three planes
mdat = np.column_stack((kx_plane1, ky_plane1, t_plane1, q_plane1,
                        kx_plane2, ky_plane2, t_plane2, q_plane2,
                        kx_plane3, ky_plane3, t_plane3, q_plane3))

ix1 = 0
iy1 = 1
it1 = 2  # time plane 1
iq1 = 3
ix2 = 4
iy2 = 5
it2 = 6  # time plane 2
iq2 = 7  # charge plane 2
ix3 = 8
iy3 = 9
it3 = 10  # time plane 3
iq3 = 11  # charge plane 3

mdat_rows, mdat_cols = mdat.shape

# Time per cell

# PLANE T1

time_1_11 = np.array([])
time_1_12 = np.array([])
time_1_13 = np.array([])
time_1_14 = np.array([])
time_1_15 = np.array([])
time_1_21 = np.array([])
time_1_22 = np.array([])
time_1_23 = np.array([])
time_1_24 = np.array([])
time_1_25 = np.array([])
time_1_31 = np.array([])
time_1_32 = np.array([])
time_1_33 = np.array([])
time_1_34 = np.array([])
time_1_35 = np.array([])
time_1_41 = np.array([])
time_1_42 = np.array([])
time_1_43 = np.array([])
time_1_44 = np.array([])
time_1_45 = np.array([])
time_1_51 = np.array([])
time_1_52 = np.array([])
time_1_53 = np.array([])
time_1_54 = np.array([])
time_1_55 = np.array([])
time_1_61 = np.array([])
time_1_62 = np.array([])
time_1_63 = np.array([])
time_1_64 = np.array([])
time_1_65 = np.array([])

# PLANE T2

time_2_11 = np.array([])
time_2_12 = np.array([])
time_2_13 = np.array([])
time_2_14 = np.array([])
time_2_15 = np.array([])
time_2_21 = np.array([])
time_2_22 = np.array([])
time_2_23 = np.array([])
time_2_24 = np.array([])
time_2_25 = np.array([])
time_2_31 = np.array([])
time_2_32 = np.array([])
time_2_33 = np.array([])
time_2_34 = np.array([])
time_2_35 = np.array([])
time_2_41 = np.array([])
time_2_42 = np.array([])
time_2_43 = np.array([])
time_2_44 = np.array([])
time_2_45 = np.array([])
time_2_51 = np.array([])
time_2_52 = np.array([])
time_2_53 = np.array([])
time_2_54 = np.array([])
time_2_55 = np.array([])
time_2_61 = np.array([])
time_2_62 = np.array([])
time_2_63 = np.array([])
time_2_64 = np.array([])
time_2_65 = np.array([])

# PLANE T3

time_3_11 = np.array([])
time_3_12 = np.array([])
time_3_13 = np.array([])
time_3_14 = np.array([])
time_3_15 = np.array([])
time_3_21 = np.array([])
time_3_22 = np.array([])
time_3_23 = np.array([])
time_3_24 = np.array([])
time_3_25 = np.array([])
time_3_31 = np.array([])
time_3_32 = np.array([])
time_3_33 = np.array([])
time_3_34 = np.array([])
time_3_35 = np.array([])
time_3_41 = np.array([])
time_3_42 = np.array([])
time_3_43 = np.array([])
time_3_44 = np.array([])
time_3_45 = np.array([])
time_3_51 = np.array([])
time_3_52 = np.array([])
time_3_53 = np.array([])
time_3_54 = np.array([])
time_3_55 = np.array([])
time_3_61 = np.array([])
time_3_62 = np.array([])
time_3_63 = np.array([])
time_3_64 = np.array([])
time_3_65 = np.array([])

for i in range(mdat_rows):

    kx1 = mdat[i, ix1]
    ky1 = mdat[i, iy1]
    kx2 = mdat[i, ix2]
    ky2 = mdat[i, iy2]
    kx3 = mdat[i, ix3]
    ky3 = mdat[i, iy3]

    # PLANE T1

    if kx1 == 1 and ky1 == 1:
        time_1_11 = np.append(time_1_11, mdat[i, it1])
    elif kx1 == 1 and ky1 == 2:
        time_1_12 = np.append(time_1_12, mdat[i, it1])
    elif kx1 == 1 and ky1 == 3:
        time_1_13 = np.append(time_1_13, mdat[i, it1])
    elif kx1 == 1 and ky1 == 4:
        time_1_14 = np.append(time_1_14, mdat[i, it1])
    elif kx1 == 1 and ky1 == 5:
        time_1_15 = np.append(time_1_15, mdat[i, it1])


    elif kx1 == 2 and ky1 == 1:
        time_1_21 = np.append(time_1_21, mdat[i, it1])
    elif kx1 == 2 and ky1 == 2:
        time_1_22 = np.append(time_1_22, mdat[i, it1])
    elif kx1 == 2 and ky1 == 3:
        time_1_23 = np.append(time_1_23, mdat[i, it1])
    elif kx1 == 2 and ky1 == 4:
        time_1_24 = np.append(time_1_24, mdat[i, it1])
    elif kx1 == 2 and ky1 == 5:
        time_1_25 = np.append(time_1_25, mdat[i, it1])


    elif kx1 == 3 and ky1 == 1:
        time_1_31 = np.append(time_1_31, mdat[i, it1])
    elif kx1 == 3 and ky1 == 2:
        time_1_32 = np.append(time_1_32, mdat[i, it1])
    elif kx1 == 3 and ky1 == 3:
        time_1_33 = np.append(time_1_33, mdat[i, it1])
    elif kx1 == 3 and ky1 == 4:
        time_1_34 = np.append(time_1_34, mdat[i, it1])
    elif kx1 == 3 and ky1 == 5:
        time_1_35 = np.append(time_1_35, mdat[i, it1])


    elif kx1 == 4 and ky1 == 1:
        time_1_41 = np.append(time_1_11, mdat[i, it1])
    elif kx1 == 4 and ky1 == 2:
        time_1_42 = np.append(time_1_42, mdat[i, it1])
    elif kx1 == 4 and ky1 == 3:
        time_1_43 = np.append(time_1_43, mdat[i, it1])
    elif kx1 == 4 and ky1 == 4:
        time_1_44 = np.append(time_1_44, mdat[i, it1])
    elif kx1 == 4 and ky1 == 5:
        time_1_45 = np.append(time_1_45, mdat[i, it1])


    elif kx1 == 5 and ky1 == 1:
        time_1_51 = np.append(time_1_51, mdat[i, it1])
    elif kx1 == 5 and ky1 == 2:
        time_1_52 = np.append(time_1_52, mdat[i, it1])
    elif kx1 == 5 and ky1 == 3:
        time_1_53 = np.append(time_1_53, mdat[i, it1])
    elif kx1 == 5 and ky1 == 4:
        time_1_54 = np.append(time_1_54, mdat[i, it1])
    elif kx1 == 5 and ky1 == 5:
        time_1_55 = np.append(time_1_55, mdat[i, it1])

    elif kx1 == 6 and ky1 == 1:
        time_1_61 = np.append(time_1_61, mdat[i, it1])
    elif kx1 == 6 and ky1 == 2:
        time_1_62 = np.append(time_1_62, mdat[i, it1])
    elif kx1 == 6 and ky1 == 3:
        time_1_63 = np.append(time_1_63, mdat[i, it1])
    elif kx1 == 6 and ky1 == 4:
        time_1_64 = np.append(time_1_64, mdat[i, it1])
    elif kx1 == 6 and ky1 == 5:
        time_1_65 = np.append(time_1_65, mdat[i, it1])

    # PLANE T2

    if kx2 == 1 and ky2 == 1:
        time_2_11 = np.append(time_2_11, mdat[i, it2])
    elif kx2 == 1 and ky2 == 2:
        time_2_12 = np.append(time_2_12, mdat[i, it2])
    elif kx2 == 1 and ky2 == 3:
        time_2_13 = np.append(time_2_13, mdat[i, it2])
    elif kx2 == 1 and ky2 == 4:
        time_2_14 = np.append(time_2_14, mdat[i, it2])
    elif kx2 == 1 and ky2 == 5:
        time_2_15 = np.append(time_2_15, mdat[i, it2])

    elif kx2 == 2 and ky2 == 1:
        time_2_21 = np.append(time_2_21, mdat[i, it2])
    elif kx2 == 2 and ky2 == 2:
        time_2_22 = np.append(time_2_22, mdat[i, it2])
    elif kx2 == 2 and ky2 == 3:
        time_2_23 = np.append(time_2_23, mdat[i, it2])
    elif kx2 == 2 and ky2 == 4:
        time_2_24 = np.append(time_2_24, mdat[i, it2])
    elif kx2 == 2 and ky2 == 5:
        time_2_25 = np.append(time_2_25, mdat[i, it2])

    elif kx2 == 3 and ky2 == 1:
        time_2_31 = np.append(time_2_31, mdat[i, it2])
    elif kx2 == 3 and ky2 == 2:
        time_2_32 = np.append(time_2_32, mdat[i, it2])
    elif kx2 == 3 and ky2 == 3:
        time_2_33 = np.append(time_2_33, mdat[i, it2])
    elif kx2 == 3 and ky2 == 4:
        time_2_34 = np.append(time_2_34, mdat[i, it2])
    elif kx2 == 3 and ky2 == 5:
        time_2_35 = np.append(time_2_35, mdat[i, it2])

    elif kx2 == 4 and ky2 == 1:
        time_2_41 = np.append(time_2_41, mdat[i, it2])
    elif kx2 == 4 and ky2 == 2:
        time_2_42 = np.append(time_2_42, mdat[i, it2])
    elif kx2 == 4 and ky2 == 3:
        time_2_43 = np.append(time_2_43, mdat[i, it2])
    elif kx2 == 4 and ky2 == 4:
        time_2_44 = np.append(time_2_44, mdat[i, it2])
    elif kx2 == 4 and ky2 == 5:
        time_2_45 = np.append(time_2_45, mdat[i, it2])

    elif kx2 == 5 and ky2 == 1:
        time_2_51 = np.append(time_2_51, mdat[i, it2])
    elif kx2 == 5 and ky2 == 2:
        time_2_52 = np.append(time_2_52, mdat[i, it2])
    elif kx2 == 5 and ky2 == 3:
        time_2_53 = np.append(time_2_53, mdat[i, it2])
    elif kx2 == 5 and ky2 == 4:
        time_2_54 = np.append(time_2_54, mdat[i, it2])
    elif kx2 == 5 and ky2 == 5:
        time_2_55 = np.append(time_2_55, mdat[i, it2])

    elif kx2 == 6 and ky2 == 1:
        time_2_61 = np.append(time_2_61, mdat[i, it2])
    elif kx2 == 6 and ky2 == 2:
        time_2_62 = np.append(time_2_62, mdat[i, it2])
    elif kx2 == 6 and ky2 == 3:
        time_2_63 = np.append(time_2_63, mdat[i, it2])
    elif kx2 == 6 and ky2 == 4:
        time_2_64 = np.append(time_2_64, mdat[i, it2])
    elif kx2 == 6 and ky2 == 5:
        time_2_65 = np.append(time_2_65, mdat[i, it2])

    # PLANE T3

    if kx3 == 1 and ky3 == 1:
        time_3_11 = np.append(time_3_11, mdat[i, it3])
    elif kx3 == 1 and ky3 == 2:
        time_3_12 = np.append(time_3_12, mdat[i, it3])
    elif kx3 == 1 and ky3 == 3:
        time_3_13 = np.append(time_3_13, mdat[i, it3])
    elif kx3 == 1 and ky3 == 4:
        time_3_14 = np.append(time_3_14, mdat[i, it3])
    elif kx3 == 1 and ky3 == 5:
        time_3_15 = np.append(time_3_15, mdat[i, it3])

    elif kx3 == 2 and ky3 == 1:
        time_3_21 = np.append(time_3_21, mdat[i, it3])
    elif kx3 == 2 and ky3 == 2:
        time_3_22 = np.append(time_3_22, mdat[i, it3])
    elif kx3 == 2 and ky3 == 3:
        time_3_23 = np.append(time_3_23, mdat[i, it3])
    elif kx3 == 2 and ky3 == 4:
        time_3_24 = np.append(time_3_24, mdat[i, it3])
    elif kx3 == 2 and ky3 == 5:
        time_3_25 = np.append(time_3_25, mdat[i, it3])

    elif kx3 == 3 and ky3 == 1:
        time_3_31 = np.append(time_3_31, mdat[i, it3])
    elif kx3 == 3 and ky3 == 2:
        time_3_32 = np.append(time_3_32, mdat[i, it3])
    elif kx3 == 3 and ky3 == 3:
        time_3_33 = np.append(time_3_33, mdat[i, it3])
    elif kx3 == 3 and ky3 == 4:
        time_3_34 = np.append(time_3_34, mdat[i, it3])
    elif kx3 == 3 and ky3 == 5:
        time_3_35 = np.append(time_3_35, mdat[i, it3])

    elif kx3 == 4 and ky3 == 1:
        time_3_41 = np.append(time_3_41, mdat[i, it3])
    elif kx3 == 4 and ky3 == 2:
        time_3_42 = np.append(time_3_42, mdat[i, it3])
    elif kx3 == 4 and ky3 == 3:
        time_3_43 = np.append(time_3_43, mdat[i, it3])
    elif kx3 == 4 and ky3 == 4:
        time_3_44 = np.append(time_3_44, mdat[i, it3])
    elif kx3 == 4 and ky3 == 5:
        time_3_45 = np.append(time_3_45, mdat[i, it3])

    elif kx3 == 5 and ky3 == 1:
        time_3_51 = np.append(time_3_51, mdat[i, it3])
    elif kx3 == 5 and ky3 == 2:
        time_3_52 = np.append(time_3_52, mdat[i, it3])
    elif kx3 == 5 and ky3 == 3:
        time_3_53 = np.append(time_3_53, mdat[i, it3])
    elif kx3 == 5 and ky3 == 4:
        time_3_54 = np.append(time_3_54, mdat[i, it3])
    elif kx3 == 5 and ky3 == 5:
        time_3_55 = np.append(time_3_55, mdat[i, it3])

    elif kx3 == 6 and ky3 == 1:
        time_3_61 = np.append(time_3_61, mdat[i, it3])
    elif kx3 == 6 and ky3 == 2:
        time_3_62 = np.append(time_3_62, mdat[i, it3])
    elif kx3 == 6 and ky3 == 3:
        time_3_63 = np.append(time_3_63, mdat[i, it3])
    elif kx3 == 6 and ky3 == 4:
        time_3_64 = np.append(time_3_64, mdat[i, it3])
    elif kx3 == 6 and ky3 == 5:
        time_3_65 = np.append(time_3_65, mdat[i, it3])

plt.close('all')

lim = [-250, -100]
lim_pos = [100, 250]

# plane 1

plt.figure()
plt.hist(time_1_11, bins='auto')
# plt.hist(time_1_11)
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (1,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_11.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_12, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (1,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_12.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_13, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (1,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_13.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_14, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (1,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_14.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_15, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (1,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_15.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_21, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (2,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_21.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_22, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (2,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_22.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_23, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (2,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_23.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_24, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (2,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_24.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_25, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (2,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_25.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_31, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (3,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_31.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_32, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (3,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_32.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_33, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (3,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_33.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_34, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (3,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_34.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_35, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (3,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_35.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_41, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (4,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_41.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_42, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (4,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_42.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_43, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (4,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_43.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_44, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (4,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_44.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_45, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (4,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_45.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_51, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (5,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_51.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_52, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (5,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_52.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_53, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (5,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_53.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_54, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (5,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_54.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_55, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (5,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_55.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_61, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (6,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_61.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_62, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (6,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_62.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_63, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (6,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_63.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_64, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (6,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_64.png", bbox_inches='tight')

plt.figure()
plt.hist(time_1_65, bins='auto')
plt.xlabel('t (T1)')
plt.ylabel(' # Counts ')
plt.title('T1: (kx,ky) = (6,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T1_65.png", bbox_inches='tight')

# plane 2

plt.figure()
plt.hist(time_2_11, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (1,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_11.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_12, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (1,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_12.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_13, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (1,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_13.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_14, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (1,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_14.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_15, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (1,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_15.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_21, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (2,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_21.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_22, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (2,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_22.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_23, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (2,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_23.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_24, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (2,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_24.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_25, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (2,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_25.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_31, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (3,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_31.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_32, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (3,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_32.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_33, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (3,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_33.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_34, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (3,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_34.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_35, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (3,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_35.png", bbox_inches='tight')

# plt.close('all')

plt.figure()
plt.hist(time_2_41, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (4,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_41.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_42, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (4,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_42.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_43, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (4,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_43.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_44, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (4,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_44.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_45, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (4,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_45.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_51, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (5,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_51.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_52, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (5,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_52.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_53, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (5,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_53.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_54, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (5,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_54.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_55, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (5,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_55.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_61, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (6,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_61.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_62, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (6,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_62.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_63, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (6,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_63.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_64, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (6,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_64.png", bbox_inches='tight')

plt.figure()
plt.hist(time_2_65, bins='auto')
plt.xlabel('t (T2)')
plt.ylabel(' # Counts ')
plt.title('T2: (kx,ky) = (6,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T2_65.png", bbox_inches='tight')


# Plane 3

plt.figure()
plt.hist(time_3_11, bins='auto')
# plt.hist(time_3_11)
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (1,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_11.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_12, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (1,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_12.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_13, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (1,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_13.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_14, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (1,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_14.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_15, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (1,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_15.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_21, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (2,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_21.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_22, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (2,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_22.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_23, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (2,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_23.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_24, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (2,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_24.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_25, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (2,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_25.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_31, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (3,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_31.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_32, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (3,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_32.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_33, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (3,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_33.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_34, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (3,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_34.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_35, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (3,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_35.png", bbox_inches='tight')

# plt.close('all')

plt.figure()
plt.hist(time_3_41, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (4,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_41.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_42, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (4,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_42.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_43, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (4,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_43.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_44, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (4,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_44.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_45, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (4,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_45.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_51, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (5,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_51.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_52, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (5,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_52.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_53, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (5,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_53.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_54, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (5,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_54.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_55, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (5,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_55.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_61, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (6,1)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_61.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_62, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (6,2)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_62.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_63, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (6,3)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_63.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_64, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (6,4)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_64.png", bbox_inches='tight')

plt.figure()
plt.hist(time_3_65, bins='auto')
plt.xlabel('t (T3)')
plt.ylabel(' # Counts ')
plt.title('T3: (kx,ky) = (6,5)')
# plt.xlim(lim_pos)
# plt.savefig("time_T3_65.png", bbox_inches='tight')

plt.close('all')
