import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class RepresentAIRES:
    def __init__(self, file: str = 'learn1.t2505'):
        # Initialize constants
        self.file_name = file
        self.table_name = ''
        self.units = {}
        self.date = None
        self.num_sowers = 0
        self.col_titles = []
        self.data_frame = None

        # Invoke functions
        self.read_data()
        self.energy_units(_to='MeV')
        self.diagram()
        # self.histogram()

    def read_data(self):
        with open(self.file_name, 'r') as f:
            lin = f.readlines()
            lines = list(map(lambda s: s.strip(), lin))
            data = []
            for idx, line in enumerate(lines):
                if line[0] == '#':
                    if 'Task starting date:' in line:
                        self.date = line.split('date: ')[1]
                    if 'Number of showers' in line:
                        self.num_sowers = int(line[-5:])
                    if 'TABLE' in line:
                        self.table_name = line.replace('#   ', '')
                    if 'Units used' in line:
                        ix = idx + 2
                        newline = lines[ix]
                        while '# ' in newline:
                            magnitude, unit = newline.replace('#', '').replace(' ', '').split('---')
                            self.units[magnitude] = unit
                            ix += 1
                            newline = lines[ix]
                    if 'Columns' in line:
                        ix = idx + 2
                        newline = lines[ix]
                        title_str = ''
                        while '# ' in newline:
                            title_str += newline.replace('#         ', '').replace(', ', ',')
                            ix += 1
                            newline = lines[ix]
                        titles = title_str.split(',')
                        for tit in titles:
                            self.col_titles.append(tit[2:])
                else:
                    data.append(np.asarray(line.split(), dtype=np.float))

            dat_array = np.asarray(data)
            self.data_frame = pd.DataFrame(data=dat_array[:, 1:],
                                           index=dat_array[:, 0].astype(np.int),
                                           columns=self.col_titles[1:])

    def energy_units(self, _to='MeV'):
        unit_values = {'eV': 1, 'keV': 1e3, 'MeV': 1e6,
                       'GeV': 1e9, 'TeV': 1e12, 'PeV': 1e15}
        _from = self.units['Energy']
        factor = unit_values[_from] / unit_values[_to]
        self.data_frame['Energy'] = factor * self.data_frame['Energy']
        self.units['Energy'] = _to

    def diagram(self):
        table, title, particle = self.table_name.split(':')
        fig = plt.figure(table)
        ax = fig.add_subplot()
        plt.title(f'{title}: {particle}')

        x = self.data_frame['Energy']

        # Mean
        y = self.data_frame['Mean']
        plt.plot(x, y, color='#000000', label='Particles at Ground.')
        ax.fill_between(x=x, y1=y, y2=0, color='#00B5B8', alpha=0.5)

        # Minimum and Maximum
        ymin = self.data_frame['Minimum']
        ymax = self.data_frame['Maximum.']
        plt.plot(x, ymin, color='#74508D', alpha=0.25)
        plt.plot(x, ymax, color='#74508D', alpha=0.25)
        ax.fill_between(x=x, y1=ymin, y2=ymax, color='#74508D', alpha=0.3, label='Maximum and Minimum.')

        # Std. Dev.
        # ystd = self.data_frame['Std. Dev.'] / 2
        # plt.plot(x, y - ystd, color='#ED177A', alpha=0.25)
        # plt.plot(x, y + ystd, color='#ED177A', alpha=0.25)
        # ax.fill_between(x=x, y1=y - ystd, y2=y + ystd, color='#ED177A', alpha=0.2, label='Std. Dev.')

        # RMS Error.
        # yrms = self.data_frame['RMS Error'] / 2
        # plt.plot(x, y - yrms, color='#279F00', alpha=0.25)
        # plt.plot(x, y + yrms, color='#279F00', alpha=0.25)
        # ax.fill_between(x=x, y1=y - yrms, y2=y + yrms, color='#279F00', alpha=0.2, label='RMS Error')

        # Config
        ax.set_xlabel(f'Energy / {self.units["Energy"]}')
        ax.set_xscale('log')
        ax.set_ylabel('Particles at Ground')
        ax.legend(loc='best')
        ax.grid(which='both', alpha=0.25)

        # fig.savefig(f'{title} {particle}png')

    def histogram(self):
        table, title, particle = self.table_name.split(':')
        fig = plt.figure(table)
        ax = fig.add_subplot()
        plt.title(f'{title}: {particle}')
        x = self.data_frame['Energy']
        ax.hist(x, bins='auto')


RA = RepresentAIRES(file='learn1.t2501')
RB = RepresentAIRES(file='learn1.t2505')
RC = RepresentAIRES(file='learn1.t2507')
RD = RepresentAIRES(file='learn1.t2508')
