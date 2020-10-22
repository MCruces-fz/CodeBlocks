"""
@author: Miguel Cruces

mcsquared.fz@gmail.com
miguel.cruces.fernandez@usc.es
"""

import os
from os.path import join as join_path
import numpy as np
import matplotlib.pyplot as plt

# Root Directory of the Project
ROOT_DIR = os.path.abspath("./")

# =============================== S E T T I N G S - T A B L E : =============================== #

plot_cells = False  # If plot cells or not
num_size = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large',
            'xx-large'][2]  # Select sice for displayed number of hits on each cell with index

# Range of files for displaying data. Format: YYDDDHHMMSS
start_date = 20289214228  # Included
end_date = 20290011411  # Not included

# ============================================================================================= #


folder_name = [file for file in os.listdir(ROOT_DIR) if not file.endswith(".py")][0]  # First folder's name
abs_path_dir = join_path(ROOT_DIR, folder_name)

folder_files = sorted(os.listdir(abs_path_dir))
start_name = f"tr{start_date}.hld.png"
end_name = f"tr{end_date}.hld.png"
_from = folder_files.index(start_name)
_to = folder_files.index(end_name)

cell_entries = {}  # Dictionary for Cell Entries Data
plane_mult = {}  # Dictionary for Plane Multiplicity Data


def plot_cell_entries(d_name: str, cll_ent: dict):
    """
    Specific function to plot the dictionary cell_entries.

    :param cll_ent: Input dictionary
    :return: It is a void function, so that doesn't returns anything
    """
    for d_name in cll_ent:
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(f"File {d_name.replace('.hld', '')}")
        for ix, key in enumerate(cll_ent[d_name]):
            arr = cll_ent[d_name][key]
            im = axs[ix].imshow(arr)
            axs[ix].set_title(f"Plane {key}")
            for i in range(len(arr)):
                for j in range(len(arr[0])):
                    c = int(arr[i, j])
                    axs[ix].text(j, i, c, va='center', ha='center', size=num_size)


for filename in folder_files[_from:_to]:
    abs_path_file = join_path(abs_path_dir, filename)
    if filename.endswith("_cell_entries.dat"):
        day_name = f"{filename.replace('_cell_entries.dat', '')}"
        cell_entries[day_name] = {}
        with open(abs_path_file, 'r') as f:
            file = f.readlines()
            for line in file:
                if line[0][0] == "#":
                    plane_name = line.replace("\n", "")[1:]
                    cell_entries[day_name][plane_name] = np.zeros([0, 12])
                    continue
                row = [int(val) for val in line.replace("\n", "").split("\t")[:-1]]
                cell_entries[day_name][plane_name] = np.vstack((cell_entries[day_name][plane_name], row))
            if not plot_cells:
                continue
            plot_cell_entries(day_name, cell_entries)
    elif filename.endswith("_plane_mult.dat"):
        day_name = f"{filename.replace('_plane_mult.dat', '')}"
        plane_mult[day_name] = {}
        with open(abs_path_file, 'r') as f:
            file = f.readlines()
            for line in file:
                if line[0][0] == "#":
                    plane_name = line.replace("\n", "")[1:]
                    plane_mult[day_name][plane_name] = np.zeros([0, 10])
                    continue
                row = [int(val) for val in line.replace("\n", "").split("\t")[:-1]]
                plane_mult[day_name][plane_name] = np.vstack((plane_mult[day_name][plane_name], row))
        pass
