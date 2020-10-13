import matplotlib.pyplot as plt
import numpy as np

plt.close("all")

with open('map_p_s_19_60.dat', newline='\n') as f:
    raw_lines = f.readlines()
    print("*** Len Data: ", len(raw_lines))
    data = np.zeros((0, 5))
    count = 0
    for line in raw_lines:
        count += 1
        # print(count, "/", len(raw_lines))
        row = [float(i) for i in line.split()]
        data = np.vstack((data, row))

ux = data[:, 3]
uy = data[:, 4]
uz = - np.sqrt(1 - ux**2 - uy**2)
angle = np.arctan(- np.sqrt(ux**2 + uy**2)/uz)

r = data[:, 1]
phi = data[:, 2]
plt.figure('Positions')
# plt.plot(r * np.cos(phi), r * np.sin(phi))
plt.plot(r * np.cos(phi), r * np.sin(phi), "k.")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position Hits at Ground')
plt.savefig('positions.png')

fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
hb = ax.hexbin(r, phi, bins='log', gridsize=50, cmap='inferno')
ax.set_title("Hexagon coordinates")
ax.set_xlabel('Distance from core / m')
ax.set_ylabel('Polar angle / rad')
cb = fig.colorbar(hb, ax=ax)
cb.set_label('counts')
fig.savefig('phi_r_coordinates.png')

fig, ax = plt.subplots(ncols=1, figsize=(7, 4))
hb = ax.hexbin(phi, angle, bins='log', gridsize=50, cmap='inferno')
ax.set_title("Hexagon energies")
ax.set_xlabel('Polar angle / rad')
ax.set_ylabel('Particle Energy / GeV')
cb = fig.colorbar(hb, ax=ax)
cb.set_label('counts')
fig.savefig('E_phi_coordinates.png')

codes = data[:, 0]
gamm_ids = np.where(codes == 1)
elec_ids = np.where(abs(codes) == 2)
muon_ids = np.where(abs(codes) == 3)

plt.figure('Gammas')
plt.hist(angle[gamm_ids], bins='auto')
plt.xlabel('Zenith Angle / rad')
plt.ylabel('Number of gammas at grid')
plt.xlim(0, 1.6)
plt.ylim(0, 1400)
plt.title('Gammas Histogram')
plt.savefig('gammas_histogram.png')

plt.figure('Electrons')
plt.hist(angle[elec_ids], bins='auto')
plt.xlabel('Zenith Angle / rad')
plt.ylabel('Number of electrons at grid')
plt.xlim(0, 1.6)
plt.ylim(0, 1400)
plt.title('Electrons Histogram')
plt.savefig('electrons_histogram.png')

plt.figure('Muons')
plt.hist(angle[muon_ids], bins='auto')
plt.xlabel('Zenith Angle / rad')
plt.ylabel('Number of muons at grid')
plt.xlim(0, 1.6)
plt.ylim(0, 1400)
plt.title('Muons Histogram')
plt.savefig('muons_histogram.png')

plt.figure('All Part')
plt.hist(angle, bins='auto')
plt.xlabel('Polar Angle / rad')
plt.ylabel('Number of particles at grid')
plt.xlim(0, 1.6)
plt.ylim(0, 1400)
plt.title('Particles Histogram')
plt.savefig('all_particles_histogram.png')

plt.show()
