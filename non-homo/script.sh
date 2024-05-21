
python3 main.py --dataset genius --epochs 1000 --seed 51290 --K 8 --patience 200 --bias bn --hid 128 --nlayers 4 --tau 0.9 --lr 0.001 --weight_decay 5e-05 --dropout 0 --model gcn

python3 3main.py --dataset penn94 --K 4 --hid 256 --lr1 0.001 --lr2 0.05 --wd1 0 --wd2 0.00005 --dpC 0 --dpM 0.7 --nlayers 3 --tau 0.1 --tau1 1.1 --tau2 0.6

