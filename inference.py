import argparse
import os

import typer
from pathlib import Path
from enum import Enum

from misc import set_seed


def main(
    config_path: Path = typer.Option(
        "configs/inference_yaml/inference_universal.yaml", 
        help="Path to the config file"
    ),
    checkpoint_path: Path = typer.Option(
        "", 
        help="Path to the checkpoint"
    ),
    img_path: Path = typer.Option(
        "demo_images/universal/0000.png", 
        help="Path to the image"
    ),
    output_folder: Path = typer.Option(
        "outputs/", 
        help="Output folder"
    ),
    num_output_frames: int = typer.Option(
        150, 
        help="Number of output latent frames"
    ),
    seed: int = typer.Option(0, help="Random seed"),
    pretrained_model_path: Path = typer.Option(
        "Matrix-Game-2.0", 
        help="Path to the VAE model folder"
    ),
    enable_profile: bool = typer.Option(
        False, 
        help="Enable torch profiling",
    ),

    vae_compile_mode: VAECompileMode = typer.Option(
        VAECompileMode.AUTO,
        help="VAE decoder compile mode",
        case_sensitive=False
    )
):
    set_seed(seed)
    os.makedirs(output_folder, exist_ok=True)


    

if __name__ == "__main__":
    typer.run(main)