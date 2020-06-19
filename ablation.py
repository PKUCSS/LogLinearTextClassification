import os 

for lr in [0.001,0.01,0.1]:
    for bs in [10000]:
        for lemma in [0,1e-4,1e-2,1e-1]:
            os.system('python main.py --batch_size {} --lr {} --lemma {}'.format(bs,lr,lemma))
            # os.system('python3 main.py --batchsize {} --lr {} --lemma {}'.format(bs,lr,lemma))