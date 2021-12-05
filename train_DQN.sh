for seed in {1..2000}
do
    python train_DQN.py -config 'configs/normal.json' -seed $seed
done