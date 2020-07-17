import matplotlib.pyplot as plt
import csv
import numpy as np

with open('mu_datos_hexbin.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\n')  # , quotechar=',')
    titles = []
    data = []
    for row in spamreader:
        rowSplit = row[0].split(',')
        try:
            data.append(list(np.array(rowSplit, dtype=np.float)))
        except ValueError:
            titles = rowSplit

data = list(np.array(data).transpose())
pag = dict(zip(titles, data))

dt = pag['Tiempo_{f}'] - pag['Tiempo_{i}']
x = pag['DistGeomCen [m]'][dt > 0]
y = dt[dt > 0]


plt.figure(0)
plt.hexbin(x, y, mincnt=1, cmap='jet', bins='log', gridsize=100, xscale='log', yscale='log', extent=(-1, 4, -1, 3.5))
plt.colorbar(extend='max')
plt.title('Title')
plt.xlabel('Distancia al centro de la cascada / m')
plt.ylabel(r'$\Delta$t / ns')
plt.savefig('fig.png')
plt.show()
# plt.close()
