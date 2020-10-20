"""
Change Datetime Format

Tue 20 Oct 2020
"""

import numpy as np
import datetime

dias = np.array([  # Días en formato largo
    "2019-01-01 06:00:00",
    "2019-01-01 08:30:00",
    "2019-01-02 06:00:00",
    "2019-01-02 06:00:00",
    "2019-01-03 00:00:00",
    "2019-01-03 12:24:00",
    "2019-02-01 06:00:12",
    "2020-12-15 18:35:00",
    "2018-06-08 12:00:00",
])


def datetime_format(datime: np.array) -> np.array:
    """
    Función que cambia la fecha y hora de formato largo a
    formato comprimido.

    :param datime: Array con fechas en strings de formato
        largo "YYYY-DD-MM HH:MM:SS"
    :return: Array con fecha y hora en strings de formato
        reducido "YYDDD-HHMM"
    """
    short_datime = np.zeros([0, 1])
    for dat in datime:
        doy = datetime.date(int(dat[:4]), int(dat[5:7]), int(dat[8:10])).strftime('%j')
        sh_dat = f"{dat[2:4]}{doy}-{dat[11:13]}{dat[14:16]}"
        short_datime = np.vstack((short_datime, sh_dat))
    return short_datime


dias2 = datetime_format(dias)  # Días en formato corto
