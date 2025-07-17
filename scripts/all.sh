dataset=set14

# jpeg
python src/main.py dataset.name=${dataset} transmitter.cs_method=jpeg transmitter.clip_model.enable=false transmitter.contour.enable=false receiver.method=jpeg receiver.stable_diffusion.enable=false

# bilinear
python src/main.py dataset.name=${dataset} transmitter.cs_method=pixel transmitter.clip_model.enable=false transmitter.contour.enable=false receiver.method=bi receiver.stable_diffusion.enable=false

# wht-cs
python src/main.py dataset.name=${dataset} transmitter.cs_method=walsh transmitter.clip_model.enable=false transmitter.contour.enable=false receiver.method=walsh receiver.stable_diffusion.enable=false

# sd-inpaint
python src/main.py dataset.name=${dataset} transmitter.cs_method=pixel receiver.method=sd receiver.stable_diffusion.method=inpaint

# lrgd
python src/main.py dataset.name=${dataset} transmitter.cs_method=pixel receiver.method=sd receiver.stable_diffusion.method=lowrank