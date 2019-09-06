learningRates=( 0.005 0.01 0.02 )
batchSizes=( 10000 30000 50000 )

for i in "${learningRates[@]}"
do
    for j in "${batchSizes[@]}"
    do
        python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b $j -lr $i -rtg --nn_baseline --exp_name hc_b"$j"_r"$i"
    done
done