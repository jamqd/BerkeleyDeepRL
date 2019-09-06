
learningRate=0
batchSize=0

python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $batchSize -lr $learningRate --exp_name hc_b"$batchSize"_r"$learningRate"
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $batchSize-lr $learningRate -rtg --exp_name hc_b"$batchSize"_r"$learningRate"
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $batchSize -lr $learningRate --nn_baseline --exp_name hc_b"$batchSize"_r"$learningRate"
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b $batchSize -lr $learningRate -rtg --nn_baseline --exp_name hc_b"$batchSize"_r"$learningRate"