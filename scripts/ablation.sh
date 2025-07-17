# Full
python src/main.py

# w/o Vis.
python src/main.py receiver.stable_diffusion.rank_percent=0

# w/o Smoo.
python src/main.py receiver.stable_diffusion.smooth_transition=false

# w/o Ortho.
python src/main.py receiver.stable_diffusion.ortho_projection=false

# w/o Cond.
python src/main.py transmitter.contour.enable=false

# w/o Txt.
python src/main.py transmitter.clip_model.enable=false

# Vis. only
python src/main.py receiver.stable_diffusion.smooth_transition=false transmitter.contour.enable=false receiver.stable_diffusion.ortho_projection=false transmitter.clip_model.enable=false

# Vis. + Smoo.
python src/main.py transmitter.contour.enable=false receiver.stable_diffusion.ortho_projection=false transmitter.clip_model.enable=false

# Vis. + Smoo. + Ortho.
python src/main.py transmitter.contour.enable=false transmitter.clip_model.enable=false

# Vis. + Smoo. + Cond.
python src/main.py receiver.stable_diffusion.ortho_projection=false transmitter.clip_model.enable=false

# All in one
dataset=bsd100 \
&& python src/main.py dataset.name=${dataset} \
&& python src/main.py dataset.name=${dataset} receiver.stable_diffusion.rank_percent=0 \
&& python src/main.py dataset.name=${dataset} receiver.stable_diffusion.smooth_transition=false \
&& python src/main.py dataset.name=${dataset} receiver.stable_diffusion.ortho_projection=false \
&& python src/main.py dataset.name=${dataset} transmitter.contour.enable=false \
&& python src/main.py dataset.name=${dataset} transmitter.clip_model.enable=false \
&& python src/main.py dataset.name=${dataset} receiver.stable_diffusion.smooth_transition=false transmitter.contour.enable=false receiver.stable_diffusion.ortho_projection=false transmitter.clip_model.enable=false \
&& python src/main.py dataset.name=${dataset} transmitter.contour.enable=false receiver.stable_diffusion.ortho_projection=false transmitter.clip_model.enable=false \
&& python src/main.py dataset.name=${dataset} transmitter.contour.enable=false transmitter.clip_model.enable=false \
&& python src/main.py dataset.name=${dataset} receiver.stable_diffusion.ortho_projection=false transmitter.clip_model.enable=false
