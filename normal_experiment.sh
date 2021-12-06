for i in 0.1 0.5 1 1.2 1.4 1.6 1.8 2 4 6 8 10 20
do
    python train_attacker.py -attack_config configs/attack.json -agent_config configs/normal.json -norm_bound $i
done