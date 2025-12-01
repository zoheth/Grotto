import os

os.environ["PYTHONBREAKPOINT"] = "0"
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"
from pathlib import Path

import numpy as np
import typer
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.loading_utils import load_image

from grotto.generator import VAECompileMode, VideoGenerator
from grotto.misc import set_seed


def process_video(input_video, output_video):
    fps = 12
    frame_count = len(input_video)

    out_video = []
    frame_idx = 0
    for frame in input_video:
        out_video.append(frame / 255)
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{frame_count}", end="\r")
    export_to_video(out_video, output_video, fps=fps)
    print("\nProcessing complete!")


def main(
    config_path: Path = typer.Option(
        "configs/universal/inference.yaml", help="Path to the config file"
    ),
    checkpoint_path: Path = typer.Option("", help="Path to the checkpoint"),
    img_path: Path = typer.Option("demo_images/universal/0000.png", help="Path to the image"),
    output_folder: Path = typer.Option("outputs/", help="Output folder"),
    num_output_frames: int = typer.Option(150, help="Number of output latent frames"),
    seed: int = typer.Option(0, help="Random seed"),
    pretrained_model_path: Path = typer.Option(
        "Matrix-Game-2.0", help="Path to the VAE model folder"
    ),
    enable_profile: bool = typer.Option(
        False,
        help="Enable torch profiling",
    ),
    vae_compile_mode: VAECompileMode = typer.Option(
        VAECompileMode.NONE,
        help="VAE decoder compile mode: auto (use cache if available), force (recompile), none (no compile)",
    ),
):
    set_seed(seed)
    os.makedirs(output_folder, exist_ok=True)

    generator = VideoGenerator(
        config_path, checkpoint_path, pretrained_model_path, vae_compile_mode=vae_compile_mode
    )

    image = load_image(str(img_path))
    video = generator.generate(image, num_output_frames)

    process_video(video.astype(np.uint8), output_folder / "demo.mp4")

    print("Done")


if __name__ == "__main__":
    typer.run(main)
