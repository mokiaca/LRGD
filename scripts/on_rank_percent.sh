dataset=set14

python src/main.py -m dataset.name=${dataset} transmitter.sampling.rate=0.01,0.04,0.1,0.25 receiver.stable_diffusion.rank_percent=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1

# seperately
for i in 0.01 0.04 0.1 0.25
do
    for j in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
    do
        python src/main.py dataset.name=${dataset} transmitter.sampling.rate=${i} receiver.stable_diffusion.rank_percent=${j}
    done
done