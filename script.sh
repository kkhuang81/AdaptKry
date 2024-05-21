python3 m3training.py --dataset cora --epochs 1000 --seed 51290 --patience 200 --hid 256 --nlayers 3 --K 10 --lr1 0.1 --lr2 0.0005 --wd1 0.001 --wd2 0 --dpC 0.2 --dpM 0.3 --tau 0.5 --tau1 1.1 --tau2 0.8

python training.py --dataset citeseer --epochs 1000 --seed 51290 --patience 200 --hid 128 --nlayers 4 --K 10 --lr1 0.01 --lr2 0.05 --wd1 5e-5 --wd2 0.0001 --dpC 0.9 --dpM 0.8 --tau 0.1

python training.py --dataset pubmed --epochs 1000 --seed 51290 --patience 200 --hid 128 --nlayers 4 --K 10 --lr1 0.1 --lr2 0.05 --wd1 0.0005 --wd2 0.0001 --dpC 0.4 --dpM 0 --tau 0.5

python3 m3training.py --dataset actor --epochs 1000 --seed 51290 --patience 200 --hid 256 --nlayers 4 --K 10 --lr1 0.1 --lr2 0.01 --wd1 0.0005 --wd2 0 --dpC 0 --dpM 0.4 --tau 0.6 --tau1 1.7 --tau2 1.8

python3 mtraining.py --dataset chameleon --epochs 1000 --seed 51290 --patience 400 --hid 256 --nlayers 4 --K 10 --lr1 0.015 --lr2 0.05 --wd1 0.0001 --wd2 0.0001 --dpC 0 --dpM 0.7 --tau 0.5 --tau1 0.8

python3 training.py --dataset squirrel --epochs 1000 --seed 51290 --patience 200 --hid 256 --nlayers 2 --K 10 --lr1 0.01 --lr2 0.1 --wd1 0.0001 --wd2 0.0005 --dpC 0 --dpM 0.6 --bias 'bn' --tau 0.8

python3 mtraining.py --dataset ogbnarxiv --epochs 1000 --patience 250 --K 10 --hid 256 --seed 25190 --lr 0.05 --dropout 0.7  --weight_decay 0 --nlayers 2 --model gcn --tau 0.6 --tau1 1.2
