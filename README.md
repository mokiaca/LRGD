# LRGD: Low-Rank Guided Diffusion for Robust Image Transmission in Semantic Communication

This repository contains the official implementation of [LRGD: Low-Rank Guided Diffusion for Robust Image Transmission in Semantic Communication](https://ieeexplore.ieee.org/document/11078451).

![framework](figure/framework.svg)

## Usage

### Installation

List all dependencies and how to install them. For example:

```bash
# create a new conda env
conda create -n lrgd python=3.12.2
conda activate lrgd

# clone this repo
git clone https://github.com/mokiaca/LRGD.git
cd LRGD

# install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Run following commands to download datasets in `~/datasets/`:

```bash
# Flickr8k
mkdir -p ~/datasets/flickr8k
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -P ~/datasets/flickr8k
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -P ~/datasets/flickr8k

# CBSD68 (not necessary)
mkdir -p ~/datasets/cbsd68
wget https://huggingface.co/datasets/deepinv/CBSD68/resolve/main/CBSD68.tar.gz -P ~/datasets/cbsd68
cd ~/datasets/cbsd68 && tar -xvf CBSD68.tar.gz && rm CBSD68.tar.gz && mv CBSD68/0/* . && rm -r CBSD68
```

Other datasets could be downloaded automatically when running the code.

### Run

To run LRGD on different datasets:

```bash
dataset=set14  # set14, bsd100, div2k, flickr8k
python src/main.py dataset.name=${dataset} transmitter.cs_method=pixel receiver.method=sd receiver.stable_diffusion.method=lowrank
```

For detailed parameter settings, please refer to the [configuration file](conf/config.yaml).

- To run baseline methods (JPEG, Bilinear, WHT-CS, SD-Inpaint), check the commands in [scripts/all.sh](scripts/all.sh)
- For experiments with different CS sampling ratios, see [scripts/on_sampling_ratio.sh](scripts/on_sampling_ratio.sh)
- For robustness experiments under different SNR settings, see [scripts/on_snr.sh](scripts/on_snr.sh)
- For ablation studies, see [scripts/ablation.sh](scripts/ablation.sh)
- For experiments with different inference steps, see [scripts/on_inference_step.sh](scripts/on_inference_step.sh).
- For Experiments with different rank percent, see [scripts/on_rank_percent.sh](scripts/on_rank_percent.sh).

You can also modify the parameters directly in the [configuration file](conf/config.yaml) according to your needs and run `python src/main.py`.

## Citation

If you find this project useful, please consider citing:

```text
@article{zhao2025lrgd,
  title={LRGD: Low-Rank Guided Diffusion for Robust Image Transmission in Semantic Communication},
  author={Zhao, Zengrui and Wu, Celimuge and Lin, Yangfei and Zhong, Lei and Ji, Yusheng and Ohtsuki, Tomoaki and Bennis, Mehdi},
  journal={IEEE Transactions on Cognitive Communications and Networking},
  year={2025},
  publisher={IEEE}
}
```
