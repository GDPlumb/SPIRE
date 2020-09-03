
for ((i=0;i<$2;i++))
do
    CUDA_VISIBLE_DEVICES=$i python Run_models.py $1 $i $3 &
done
wait
