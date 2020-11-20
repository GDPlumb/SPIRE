
SETUP=false
TRAIN=false
EVAL=false
SEARCH=false
PLOT=true

# 'bottle person' 'bowl person' 'car person' 'chair person' 'cup person' 'dining+table person' 'bottle cup' 'bowl cup' 'chair cup' 'bottle dining+table' 'bowl dining+table' 'chair dining+table' 'cup dining+table'
for i in 'bottle person' 'bowl person' 'car person' 'chair person' 'cup person' 'dining+table person'
do
    set -- $i
    main=$1
    spurious=$2
        
    echo ''
    echo ''
    echo ''
    echo $main $spurious
    
    for mode in 'careful-paint-tune'
    do
        echo ''
        echo $mode
        
        if $SETUP ; then
            echo 'Setting Up'
            ./SetupPair.sh $main $spurious
        fi
            
        for p in 0.6 0.8 0.9 0.95 0.975
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
        
        if $PLOT ; then
            echo 'Plotting'
            ./Plot.sh $main $spurious
            echo ''
        fi
    done
done
