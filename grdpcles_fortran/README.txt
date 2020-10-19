En el ejemplo grdpcles_reader.f hay muchas cosas que realmente 
no hacen falta, lo más importante está en las líneas que empiezan 
en (line 125):

c ####################################
c Read data from .grdpcles AIRES files

hasta la instrucción "end" (line 271).

Para compilarlo tienes que hacer link con las librerías de Aires 
que se instalan con Aires. Por ejemplo en mi caso:

$ gfortran -o grdpcles_map grdpcles_reader.f -L/home/mcruces/aires/19-04-00/lib/ -lAires -lgfortran 

Y lo puedes correr escribiendo un archivo ejecutable que contenga 
lo siguiente:

$ ./grdpcles_map << XX1
p_s_19_60.grdpcles
map_p_s_19_60.dat      ! Output file
10000. 10000.              ! Size of grid x and y (m)
25.                              ! Step (m)
5                                ! Number of showers
XX1

El p_s_19_60.grdpcles por si quieres probar, está en el mismo 
directorio. 

Tienes más información sobre cómo están estructurados los grdpcles 
files en el capítulo 4 del manual de Aires.

INFO: Jaime Álvarez Muñiz.
