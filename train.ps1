$MODELS = @("EDOF_CNN_modified","EDOF_CNN_max","EDOF_CNN_3D","EDOF_CNN_pairwise")
foreach($MODEL in $MODELS) {
    for($FOLD=0; $FOLD -le 2; $FOLD++) {
		python train.py --method $MODEL --Z 3 --fold $FOLD --cudan 0
        python train.py --method $MODEL --Z 5 --fold $FOLD --cudan 0
		python train.py --method $MODEL --Z 7 --fold $FOLD --cudan 0
        python train.py --method $MODEL --Z 9 --fold $FOLD --cudan 0
    }
}

