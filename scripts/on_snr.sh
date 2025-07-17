# any
dataset=set14
python src/main.py -m dataset.name=${dataset} channel.snr=5,6.25,7.5,8.75,10

# jpeg
python src/main.py -m dataset.name=${dataset} transmitter.cs_method=jpeg transmitter.clip_model.enable=false transmitter.contour.enable=false receiver.method=jpeg receiver.stable_diffusion.enable=false channel.snr=5,6.25,7.5,8.75,10

# wht-cs
python src/main.py -m dataset.name=${dataset} transmitter.cs_method=walsh transmitter.clip_model.enable=false transmitter.contour.enable=false receiver.method=walsh receiver.stable_diffusion.enable=false channel.snr=5,6.25,7.5,8.75,10

# bilinear
python src/main.py -m dataset.name=${dataset} transmitter.cs_method=pixel transmitter.clip_model.enable=false transmitter.contour.enable=false receiver.method=bi receiver.stable_diffusion.enable=false channel.snr=5,6.25,7.5,8.75,10

# sd-inpaint
python src/main.py -m dataset.name=${dataset} transmitter.cs_method=pixel receiver.method=sd receiver.stable_diffusion.method=inpaint channel.snr=5,6.25,7.5,8.75,10

# lrgd
python src/main.py -m dataset.name=${dataset} transmitter.cs_method=pixel receiver.method=sd receiver.stable_diffusion.method=lowrank channel.snr=5,6.25,7.5,8.75,10