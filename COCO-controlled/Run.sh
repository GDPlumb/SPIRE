main='cup'
spurious='bowl'

./SetupPair.sh $main $spurious

for p in 0.1 0.5 0.8 0.9 0.95 0.99
do
    ./TrainPair.sh $main $spurious $p 0.1
    ./EvaluatePair.sh $main $spurious $p 0.1
done
