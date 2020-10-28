mode=$1
main=$2
spurious=$3

for p in 0.5 0.6 0.4 0.7 0.3 0.8 0.2 0.9 0.1
do
    ./Train.sh $mode $main $spurious $p
    ./Evaluate.sh $mode $main $spurious $p
done
