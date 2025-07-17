# multi run
dataset=set14
python src/main.py -m dataset.name=${dataset} transmitter.sampling.rate=0.01,0.04,0.1,0.25

# or
dataset=set14
python src/main.py dataset.name=${dataset} transmitter.sampling.rate=0.01 \
&& python src/main.py dataset.name=${dataset} transmitter.sampling.rate=0.04 \
&& python src/main.py dataset.name=${dataset} transmitter.sampling.rate=0.1 \
&& python src/main.py dataset.name=${dataset} transmitter.sampling.rate=0.25

# run all
python src/main.py -m transmitter.sampling.rate=0.01,0.04,0.1,0.25 dataset.name=set14,bsd100,div2k,flickr8k

# wht-cs
python src/main.py -m transmitter.sampling.rate=0.01,0.04,0.1,0.25 dataset.name=set14,bsd100,div2k,flickr8k transmitter.cs_method=walsh transmitter.clip_model.enable=false transmitter.contour.enable=false receiver.method=walsh receiver.stable_diffusion.enable=false

# bilinear
python src/main.py -m transmitter.sampling.rate=0.01,0.04,0.1,0.25 dataset.name=set14,bsd100,div2k,flickr8k transmitter.cs_method=pixel transmitter.clip_model.enable=false transmitter.contour.enable=false receiver.method=bi receiver.stable_diffusion.enable=false

# sd-inpaint
python src/main.py -m transmitter.sampling.rate=0.01,0.04,0.1,0.25 dataset.name=set14,bsd100,div2k,flickr8k transmitter.cs_method=pixel receiver.method=sd receiver.stable_diffusion.method=inpaint transmitter.clip_model.enable=true transmitter.contour.enable=true receiver.stable_diffusion.enable=true

# lrgd
python src/main.py -m transmitter.sampling.rate=0.01,0.04,0.1,0.25 dataset.name=set14,bsd100,div2k,flickr8k transmitter.cs_method=pixel receiver.method=sd receiver.stable_diffusion.method=lowrank transmitter.clip_model.enable=true transmitter.contour.enable=true receiver.stable_diffusion.enable=true 