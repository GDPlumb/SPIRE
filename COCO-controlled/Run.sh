main='chair'
spurious='dining+table'

./SetupPair.sh $main $spurious

for p in 0.05 0.1 0.25 0.5 0.75 0.9 0.95
do
    ./TrainPair.sh $main $spurious $p 0.5
    ./EvaluatePair.sh $main $spurious $p 0.5
done
