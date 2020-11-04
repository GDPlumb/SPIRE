
SETUP=false
TRAIN=false
EVAL=false
SEARCH=true

#
for mode in 'initial-transfer' 'spurious-transfer' 'spurious-paint-transfer' 'both-transfer' 'initial-tune' 'spurious-tune' 'spurious-paint-tune'
do
    echo ''
    echo ''
    echo ''
    echo $mode
    for i in 'bottle person'
#            'bowl person'\
#            'car person' \
#            'chair person' \
#            'cup person' \
#            'dining+table person' \
#            'bottle cup' \
#            'bowl cup' \
#            'chair cup' \
#            'bottle dining+table' \
#            'bowl dining+table' \
#            'chair dining+table' \
#            'cup dining+table'
    do
        set -- $i
        main=$1
        spurious=$2
        
        echo ''
        echo $main $spurious
        
        if $SETUP ; then
            echo 'Setting Up'
            ./SetupPair.sh $main $spurious
        fi
            
        for p in 0.5 0.6 0.4 0.7 0.3 0.8 0.2 0.9 0.1
        do
            echo $p
            if $TRAIN ; then
                echo 'Training'
                ./Train.sh $mode $main $spurious $p
            fi
            
            if $EVAL ; then
                echo 'Evaluating'
                ./Evaluate.sh $mode $main $spurious $p
            fi
            
            if $SEARCH ; then
                echo 'Searching'
                ./Search.sh $mode $main $spurious $p
            fi
        done
        
        echo 'Plotting'
        ./Plot.sh $main $spurious
        echo ''
    done
done
