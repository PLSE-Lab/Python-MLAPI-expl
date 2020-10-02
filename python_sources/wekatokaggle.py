#!/bin/bash

# Este script permite modificar el formato del archivo generado por Weka para que pueda ser subido a Kaggle.

# Descomentar a partir de esta linea

#file_aux=$1".aux__"
#touch $file_aux

#sed 's/inst#,actual,predicted,error,prediction/ID,Class_WVHT/' $1 > $file_aux
#sed -r 's/^([0-9]+),.*,[1234]:([0123]),,.*/echo "$((\1-1)),\2"/ge' $file_aux > $1

#rm $file_aux