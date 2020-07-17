import numpy as np

# Array con 3000 números con celdas impactadas del 1 al 30
ncells = np.random.randint(low=1, high=31, size=3000)  # From low (inclusive) to high (exclusive)
# Array con 3000 números de cargas medias aleatorias
Q_ = np.random.uniform(low=32.0, high=5000.0, size=3000)  # from 30 to 5000


def ordenar(ncells, Q_):
    '''
    Programa que devuelve ese diccionario del que hablábamos
    :param ncells: Array con los datos del eje Y (Numero de celdas impactadas)
    :param Q_: Array con los datos del eje X (cargas medias)
    :return: Dictionary con todas esas cargas medias
    '''
    # Se crea dictionary dew salida (out) con 30 listas vacías con keys del 1 al 30
    out = {}
    for i in range(30):
        out[f'{i+1}'] = []

    # Se llena out con las cargas medias qwue pertenecen al número de celdas impactadas
    i = 0
    for n in ncells:
        out[f'{n}'].append(Q_[i])
        i += 1

    # Python program to get average of a list
    def average(lst):
        return sum(lst) / len(lst)

    # Se hace la media de las listas
    out_average = {}
    for i in out:
        out_average[i] = average(out[i])

    return out_average


print(ordenar(ncells, Q_))
