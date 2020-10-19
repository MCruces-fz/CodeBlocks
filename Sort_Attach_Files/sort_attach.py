"""
@author: Miguel Cruces

mcsquared.fz@gmail.com
miguel.cruces.fernandez@usc.es
"""

import os
from os.path import join as join_path
import sys


# Root Directory of the Project
ROOT_DIR = os.path.abspath("./")

# Add ROOT_DIR to $PATH
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

abs_path_dir = join_path(ROOT_DIR, "16October2020")
for filename in os.listdir(abs_path_dir):
    if filename.endswith("_cell_entries.dat"):
        abs_path_file = join_path(abs_path_dir, filename)
    elif filename.endswith("_plane_mult.dat"):
        abs_path_file = join_path(abs_path_dir, filename)
