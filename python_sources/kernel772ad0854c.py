# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
cupones=pd.read_csv("/kaggle/input/id-fecha-cupon/id_fecha_cupon.csv")
df=pd.DataFrame(cupones)
identi=df['id']
fecha=df['fecha']
nucup=df['cupon']
conop=0 #contador que nos indica el numero de operaciones totales de clientes repetidos
con1=0 #contador que indica numero de clientes, que repiten el servicio de un mismo cupon(158 repiten el cupon 1 y 238 repiten el cupon 2 por al menos una vez)
for a in range(len(nucup)):
    con=0 #contador que nos idica que por lo menos un cliente, repite la operacion una vez mas, con un mismo cupo
    for b in range(len(nucup)):
        if a!=b:
            if identi[a]==identi[b] and a<b: #and nucup[a]=='cupon 2':#aqui debemos de cambiar nucup[b] para cupon 1 y nucup[a] para cupon 2 por el orden de adquisicion de datos
                con +=1
                
                print(identi[b],fecha[b],nucup[b])
            if b==(len(nucup)-1) and con>=1:
                conop = conop + con +1
                con1+=1
                print(identi[a],fecha[a],nucup[a])
                print(con,con1,conop)
#enfocandonos al analisis del cupon 1 logramos ver que el cupon 1 lo utilizan 158 usuarios, y el numero de transacciones
#que cubren estos usuarios es de 316, aclarando, que los usuarios que repiten este cupoon, solo lo ocupan 1 vez mas
#lo mismo ocurre con el cupon dos con la diferencia que el numero de usuarios que ocupan el cupo 2 es de 238
#y el numero de operaciones es de 476
#ahora despreciando la etiqueta de que cupon se utiliza tenemos que 1080 usuarios utilizan entrel el cupon 1 y cupon 2
# el numero total de operaciones es de 2541
#analizando los datos
#tenemos que el cupon 1 tiene un 52% del total de operaciones
#lo que seria 2657*.52=1381.64 redondeando 1382
#las operaciones de usuarios que solo ocupan el cupon 1 una sola vez es de 
#1382-316=1066 que corresponde al 77.13% de las operaciones del cupon 1
#ahora con el cupon 2, que corresponde un 48% del total de operaciones
#que seria 2657*.48=1275.36 redondeando 1275
#para usuarios que ocuparon el cupon 2 una sola vez 
# 1275-476=799 que corresponde al 62.66% de las operaciones del cupon 2
#en analisis podemos decir que el numero de usuarios correspondientes al uso del cupon 1 es mayor aunque su uso
#llega a ser unico en mas porcentaje que el cupon 2 que llega a utilizarse en mayor porcentaje con respecto al cupon 1
#despreciando que cupon se utiliza tenemos que 1080 usuarios repiten operacion 
#las operaciones totales son 2657
#por lo tanto el nunero de usuarios que no realizan o repiten el uso de un cupo es 
# 2657-2541=116
#el numero de clientes que usaron un cupon es de 1080+116=1196(100%)
#la suma de los siguietnes valores no deberia ser el 100% por las coaliciones que hay,en un ejemplo donde un cliente pudo 
#haber usado el cupon 1 y el cupon 2 una sola ve
#el nunmero de clientes que usaron el cupon 1 una vez es 1066(89.13%)
#el numero de clientes que usaron el cupon 1 2 veces es 158(13.2%)
#el nunmero de clientes que usaron el cupon 2 una vez es 799(66.8%)
#el numero de clientes que usaron el cupon 2 2 veces es 238(19.89%)
#el cupon uno presenta un mayor impacto, ya que mas usuarios lo ocupan, pero pocos lo repiten, en cambio el cupon 2 lo ocupan 
#menos usuarios, pero llegan a usarlo nuevamente, enfocandonos  el tipo de transacciones,y en fechas desplegadas ,podriamos
#decir que es mayor el uso del cupon 1 y menor su frecuencia de repetitibilidad, por que los clientes  hacen uso de cargos por servicios
#de esto se puede concluir en un analisis sencillo, que el cupon 1 es mas atractivo a simple vistas, por que con 2 transacciones
# de 200 se esta ganando 100 y con el cupon 2 en una tranzaccion de 500 se gana 100, podria decirse que lleva mucha ventaja el cupoon 1
#y eso esta generando mas usuarios de este, sim embargo, la mayoria de los usuarios no vuelve a repetir el cupon, por lo que
#en teoria no estan aprovechando esta ventaja del cupon, y esto depende al tipo de operacion que efectue el cliente