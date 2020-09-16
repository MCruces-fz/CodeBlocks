# -*- coding: utf-8 -*-
"""
- Time distribution histogram in each cell for each plane

@author: Sara Costa Faya

EDIT: Miguel Cruces
M-AILs:
  - mcsquared.fz@gmail.com
  - miguel.cruces.fernandez@usc.es
"""

import sys
import os
from os.path import join as join_path
import numpy as np
import matplotlib.pyplot as plt
import copy


class CookingData:
    """
    Class which prepares and plots histograms
    """
    def __init__(self):
        # Root Directory of the Project
        self.root_dir = os.path.abspath("./")

        # Add ROOT_DIR to $PATH
        if self.root_dir not in sys.path:
            sys.path.append(self.root_dir)

        # Read data
        data_path = join_path(self.root_dir, "dst_export_sara_3.txt")
        data = np.loadtxt(data_path, delimiter=',', usecols=range(186))  # , max_rows=500)

        self.mdat = self.set_mdat(data)

    def set_mdat(self, data, multi=None):
        """
        Function that returns an array with columns: [x coordinate, y coordinate, time, charge] x 3 planes
        :param data: Array with data taken from file.txt
        :param multi: Optional parameter which sets multiplicity of hits. Default [1, 1, 1]. Not implemented yet.
        :return: Array with all data specified above.
        """

        if multi is None:
            multi = [1, 1, 1]  # Default multiplicity

        # Cells position in Time & Charge matrices
        it1cel = 6
        it2cel = 66
        it3cel = 126

        iq1cel = 36
        iq2cel = 96
        iq3cel = 156

        coord_ix = {1: [1, 1], 2: [2, 1], 3: [3, 1], 4: [4, 1], 5: [5, 1], 6: [6, 1],
                    7: [1, 2], 8: [2, 2], 9: [3, 2], 10: [4, 2], 11: [5, 2], 12: [6, 2],
                    13: [1, 3], 14: [2, 3], 15: [3, 3], 16: [4, 3], 17: [5, 3], 18: [6, 3],
                    19: [1, 4], 20: [2, 4], 21: [3, 4], 22: [4, 4], 23: [5, 4], 24: [6, 4],
                    25: [1, 5], 26: [2, 5], 27: [3, 5], 28: [4, 5], 29: [5, 5], 30: [6, 5]}

        m_dat = np.zeros([0, 4 * len(multi)])
        for row in data:
            # Arrays with indexes of non-zero values
            t1, t2, t3 = row[it1cel:it1cel + 30], row[it2cel:it2cel + 30], row[it3cel:it3cel + 30]
            q1, q2, q3 = row[iq1cel:iq1cel + 30], row[iq2cel:iq2cel + 30], row[iq3cel:iq3cel + 30]
            t1_id = np.nonzero(t1)[0]
            t2_id = np.nonzero(t2)[0]
            t3_id = np.nonzero(t3)[0]
            q1_id = np.nonzero(q1)[0]
            q2_id = np.nonzero(q2)[0]
            q3_id = np.nonzero(q3)[0]

            multi_list = [len(t1_id), len(t2_id), len(t3_id), len(q1_id), len(q2_id), len(q3_id)]

            # Only hits with multiplicity "multi" on time and charge (multi*2) will be stored on m_dat
            if multi_list == multi * 2:
                kx1, ky1 = coord_ix[t1_id[0] + 1]
                kx2, ky2 = coord_ix[t2_id[0] + 1]
                kx3, ky3 = coord_ix[t3_id[0] + 1]

                new_row = np.hstack((kx1, ky1, t1[t1_id[0]], q1[q1_id[0]],
                                     kx2, ky2, t2[t2_id[0]], q2[q2_id[0]],
                                     kx3, ky3, t3[t3_id[0]], q3[q3_id[0]]))
                m_dat = np.vstack((m_dat, new_row))
        return m_dat

    def make_histograms(self):
        """
        This function sorts all data and creates 180 histograms for
        time and charge, one for each cell in each plane.
        :return: It is a void function.
        """
        # 3D array of shape (No. Planes, No. X cells, No. Y cells)
        data_iter = np.asarray(np.hsplit(self.mdat, 3))
        time_hist = empty_list([3, 6, 5])
        char_hist = empty_list([3, 6, 5])
        for p in range(data_iter.shape[0]):
            for row in data_iter[p]:
                x, y = int(row[0] - 1), int(row[1] - 1)
                time = row[2]
                charge = row[3]
                time_hist[p][x - 1][y - 1].append(time)
                char_hist[p][x - 1][y - 1].append(charge)

        # Time Histograms
        for p in range(len(time_hist)):
            for x in range(len(time_hist[p])):
                for y in range(len(time_hist[p][x])):
                    plt.figure(f"T_p{p + 1}x{x + 1}y{y + 1}")
                    plt.title(f"Plane {p + 1} - Cell ({x + 1}, {y + 1}) - Time")
                    arr = np.asarray(time_hist[p][x][y])
                    filtered = arr[~is_outlier(arr)]
                    plt.hist(filtered, bins="auto")
                    plt.xlabel(f"time (T{p + 1})")
                    plt.ylabel(f"# Counts")
                    plt.savefig(f"./histograms/time_T{p + 1}_{x + 1}{y + 1}.png")
                    plt.close(f"T_p{p + 1}x{x + 1}y{y + 1}")

        # Charge Histograms
        for p in range(len(char_hist)):
            for x in range(len(char_hist[p])):
                for y in range(len(char_hist[p][x])):
                    plt.figure(f"Q_p{p + 1}x{x + 1}y{y + 1}")
                    plt.title(f"Plane {p + 1} - Cell ({x + 1}, {y + 1}) - Charge")
                    arr = np.asarray(char_hist[p][x][y])
                    filtered = arr[~is_outlier(arr)]
                    plt.hist(filtered, bins="auto")
                    plt.xlabel(f"charge (T{p + 1})")
                    plt.ylabel(f"# Counts")
                    plt.savefig(f"./histograms/charge_T{p + 1}_{x + 1}{y + 1}.png")
                    plt.close(f"Q_p{p + 1}x{x + 1}y{y + 1}")


def empty_list(shape):
    """
    Function empty_list creates empty lists with a given shape
    :param shape: list with the wanted shape.
    :return: empty list of empty lists of empty lists...
    """
    if len(shape) == 1:
        return [[] for i in range(shape[0])]
    items = shape[0]
    newshape = shape[1:]
    sublist = empty_list(newshape)
    return [copy.deepcopy(sublist) for i in range(items)]


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


if __name__ == "__main__":
    CD = CookingData()
    mdat = CD.mdat
    CD.make_histograms()
