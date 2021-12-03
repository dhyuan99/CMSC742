for i in 0.1 0.5 1 1.2 1.4 1.6 1.8 2 3 4 5 6 7 8 9 10 20
do
    python train_attacker.py -attack_config configs/attack.json -agent_config configs/robust.json -norm_bound $i
done