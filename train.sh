#!/bin/sh

METHODS="EDOF_CNN_max EDOF_CNN_3D EDOF_CNN_fast EDOF_CNN_pairwise"
#METHODS="EDOF_CNN_max EDOF_CNN_3D EDOF_CNN_fast"
KS="3 5"
FOLDS=`seq 0 4`


for K in $KS; do
	for M in $METHODS; do
		for F in $FOLDS; do
			python3 -u train.py --method $M --Z $K --fold $F
		done
	done
done

