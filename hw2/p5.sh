
learningRates=( 0.03 0.02 0.01 )
batchSizes=( 300 400 500 )

for i in "${learningRates[@]}"
do
    for j in "${batchSizes[@]}"
    do
        python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b $j -lr $i -rtg --exp_name hc_b"$j"_r"$i"
    done
done