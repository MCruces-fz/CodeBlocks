#! /usr/bin/python3

import sys
sys.path.append("../TRAGALDABAS-Kalman-Filter/")

from cosmic.hit import Hit
from simulation.clunky_sim import SimClunkyEvent
from simulation.efficiency import SimEvent
from reconstruction.track_reconstruction import TrackFinding
# from reconstruction.saeta import Saeta
from represent.represent_3d import Represent3D as r3d
from utils.const import NPLAN, VZ1, VC, PITCHX, PITCHY, DT
import numpy as np

# np.random.seed(42)

sim = SimEvent(tracks_number=2)  # Generate event here, with inputs

n_hits = sim.total_mult

def physical_speed(hit1: Hit, hit2: Hit) -> bool:
    d_time = abs(hit1.time - hit2.time) + DT
    z1 = VZ1[hit1.trb_num]
    z2 = VZ1[hit2.trb_num]
    dx = int(abs(hit1.col - hit2.col) - 0.5) * PITCHX
    dy = int(abs(hit1.row - hit2.row) - 0.5) * PITCHY
    dz = z1 - z2
    D = np.sqrt(dx**2 + dy**2 + dz**2)
    t_c = D / VC
    # print(f"t_c: {D / VC}")
    # print(f"t_v: {d_time}")
    return t_c < d_time

def nested_loops(range_list: list, hi: Hit, execute_function, ip=NPLAN - 2, hit_ids: list = []):
    """
    Exeute the execute_funtion for each combination of indices in range_list.

    :param range_list: Nested lists with indices of desired for loops
    :param execute_function: Final function to execute with all indices. It mus have
        as much parameters as len(range_list)
    :param current_index: Keep it zero (default)
    :param iter_list: Keep it None (default)
    """
    for idx in range(sim.total_mult):
        hj = sim.hits[idx]
        if hj.trb_num != ip: continue
        if not physical_speed(hi, hj): continue

        if ip == 0:
            execute_function(hit_ids + [idx])
        else:
            nested_loops(range_list[1:], hj, execute_function, ip=ip-1, hit_ids=hit_ids + [idx])


def foo(lis: list):
    print("Used -->", lis)

for k, hit in enumerate(sim.hits):
    print(f"{k}: {hit.values}")

for i in range(n_hits):
    hi = sim.hits[i]
    if hi.trb_num != NPLAN - 1: continue
    nested_loops(range(NPLAN - 1)[::-1], hi, foo, hit_ids=[i])

# print("Hits:")
# for hit in sim.hits:
#     print(hit.values)
# # sim.print_hits(size="small")
# print("")
# 
# find = TrackFinding(sim)

# represent = r3d(find.sim_evt, find.rec_evt)
# r3d.saetas(find.sim_evt, lbl="Sim.", frmt_marker='--')
# r3d.hits(find.sim_evt)
# r3d.saetas(find.rec_evt, lbl="Rec.", frmt_color="chi2", frmt_marker='-')
# r3d.show()


# print("Saetas:")
# for saeta in sim.saetas:
#     print(saeta.saeta)
# sim.print_saetas()

