import os
import sys
import time
from collections import defaultdict

import hydra
import lightning as L
import torch
import wandb
from accelerate import Accelerator
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from data.dataset import get_test_dataset
from models import Receiver, Transmitter
from models.channel import Channel
from utils.logger import setup_logger
from utils.metrics import calculate_all_metrics, calculate_fid
from utils.visualization import arrange_images, arrange_images_side_by_side

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


# @logger.catch
@hydra.main(config_path="../conf", config_name="config")
@logger.catch
def main(cfg: DictConfig):

    # 1. Initialize
    setup_logger("INFO")
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    accelerator = Accelerator()

    # 2. Load data
    dataset = get_test_dataset(cfg)
    logger.info(f"Dataset {cfg.dataset.name}: {len(dataset)} images")

    # 3. Wandb logger
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True)
        )
    else:
        wandb.init(mode="disabled")

    # 4. Model
    transmitter = Transmitter(cfg.transmitter)
    channel = Channel(cfg.channel)
    receiver = Receiver(cfg.receiver)

    # 5. Prepare models
    transmitter, receiver, dataset = accelerator.prepare(transmitter, receiver, dataset)

    # 6. Main loop
    all_metrics = []
    all_origi_images = []
    all_recon_images = []
    vis_images = defaultdict(list)
    pbar = tqdm(dataset, desc="Processing images")
    for i, data in enumerate(pbar):
        if i >= cfg.num_images > 0:
            logger.warning(f"Stopping after processing {cfg.num_images} images")
            break

        if "image_path" in data.keys():
            image_path = data["image_path"]
        elif "hr" in data.keys():
            image_path = data["hr"]
        else:
            raise ValueError("Image path not found in data")

        original_image = Image.open(image_path).convert("RGB")

        # Transmit and receive images
        start_time = time.time()
        transmitted, tx_times = transmitter(original_image)
        elapsed_transmit_time = time.time() - start_time

        orig_size = channel.calculate_size_KB(original_image)
        transmitted, trans_sizes = channel(transmitted)

        start_time = time.time()
        recon_image, inpaint, rx_times = receiver(transmitted, get_inpaint=True)
        elapsed_receive_time = time.time() - start_time

        all_origi_images.append(original_image)
        all_recon_images.append(recon_image)

        # Save images for visualization, if needed
        if cfg.save_images:
            if "sampled_pixels" in transmitted:
                transmitted["sampled_pixels"].save("sampled_pixels.png")
            if transmitted["contour"] is not None:
                transmitted["contour"].save("contour.png")
            original_image.save(f"original_image_{i}.png")
            inpaint.save("inpaint.png")
            recon_image.save(f"reconstructed_image_{i}.png")
        logger.info(f"Txt. {i:03d}: {transmitted['clip_description']}")

        # Visualize the first 4 images
        if i <= 16 and i <= len(dataset) - 1:
            vis_images[0].append(original_image)
            vis_images[1].append(recon_image)
            # vis_images[2].append(transmitted["sampled_pixels"])
            # vis_images[3].append(transmitted["contour"])
        if i == 16 or i < 16 and i == len(dataset) - 1:
            image = arrange_images_side_by_side(
                [
                    arrange_images(vis_images[i], resize_to=512)
                    for i in range(len(vis_images))
                ]
            )
            del vis_images
            if cfg.save_images:
                image.save("vis_images.png")
            wandb.log({"visualization": wandb.Image(image)})

        # Calculate metrics
        metrics = calculate_all_metrics(original_image, recon_image)
        metrics.update(
            {
                "tx_time": elapsed_transmit_time,
                "rx_time": elapsed_receive_time,
                "compression_ratio": trans_sizes["total"] / orig_size,
                # "compression_ratio_sparse": trans_sizes["sparse"] / orig_size,
                # "compression_ratio_edge": trans_sizes["edge"] / orig_size,
                # "compression_ratio_text": trans_sizes["text"] / orig_size,
            }
        )
        metrics.update({f"tx_{k}_time": v for k, v in tx_times.items()})
        metrics.update({f"rx_{k}_time": v for k, v in rx_times.items()})
        all_metrics.append(metrics)
        metrics_log = {k: f"{v:.4g}" for k, v in metrics.items()}
        pbar.set_postfix(metrics_log)
        logger.info(f"Metrics {i:03d}: {metrics_log}")

        # Log metrics
        wandb.log(metrics)

    # 7. Calculate and log average metrics
    avg_metrics = {
        k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]
    }
    avg_metrics["fid"] = calculate_fid(all_origi_images, all_recon_images)
    for metric_name, value in avg_metrics.items():
        wandb.log({f"avg_{metric_name}": value})
        logger.info(f"Average {metric_name}: {value:.6g}")

    logger.info(
        f"Results[{cfg.dataset.name}|{cfg.transmitter.sampling.rate}]: "
        + "|".join(
            [
                f"{avg_metrics[k]:.6g}"
                for k in [
                    "psnr",
                    "ssim",
                    "ms_ssim",
                    "lpips",
                    "fid",
                    "compression_ratio",
                ]
            ]
        )
    )
    time_keys = [i for i in avg_metrics.keys() if "time" in i]
    logger.info(
        "Time Results: "
        + "|".join(time_keys)
        + "\t"
        + "|".join([f"{avg_metrics[k]:.6g}" for k in time_keys])
    )

    wandb.finish()


if __name__ == "__main__":
    main()
