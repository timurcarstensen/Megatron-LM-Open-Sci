#!/usr/bin/env python3
"""
Checkpoint Converter Script

This script extracts and processes checkpoint paths from SLURM log files.
It identifies completed training runs and submits batch jobs to convert the checkpoints.
"""

import logging
import os
import re
import subprocess
from pathlib import Path

from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def extract_checkpoints_from_logs(
    slurm_log_dir: Union[str, Path],
) -> List[Tuple[Path, Path]]:
    """
    Extract checkpoint paths from SLURM log files.
    Looks for files that have completed training and extracts the checkpoint path.

    Args:
        slurm_log_dir: Directory containing SLURM log files

    Returns:
        List of tuples containing (checkpoint_path, log_file_path)
    """
    slurm_log_dir = Path(slurm_log_dir)
    checkpoint_paths_and_logs = []

    for slurm_log_file in slurm_log_dir.glob("open-sci-ref*.out"):
        with open(slurm_log_file, "r") as f:
            content = f.read()
            if "[after training is done]" in content:
                # Extract checkpoint path from the line after [after training is done]
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "[after training is done]" in line and i + 1 < len(lines):
                        # Look for checkpoint path in the next line
                        next_lines = lines[i + 1 :]
                        for next_line in next_lines:
                            if "/leonardo_work/EUHPC_E03_068/" in next_line:
                                checkpoint_path = next_line.strip()
                                checkpoint_path = (
                                    checkpoint_path.split(" to ")[-1]
                                    .strip()
                                    .split(" ")[0]
                                )
                                if os.path.exists(checkpoint_path):
                                    checkpoint_paths_and_logs.append(
                                        (Path(checkpoint_path), slurm_log_file)
                                    )
                                    break
                                else:
                                    logging.info(
                                        f"Checkpoint path {checkpoint_path} does not exist"
                                    )
                                    break

    return checkpoint_paths_and_logs


def extract_model_size(log_path: Path) -> str:
    with open(log_path, "r") as f:
        log_file = f.read()

    if "Total number of parameters in billions" not in log_file:
        raise ValueError("billions not found")

    with open(log_path, "r") as f:
        log_file_lines = f.readlines()

    for line in log_file_lines:
        if "Total number of parameters in billions" in line:
            num = line.strip().split("billions: ")[-1]
            if "1.3" in num:
                return "1.3"
            elif "0.4" in num:
                return "0.4"
            elif "1.7" in num:
                return "1.7"
            elif "0.13" in num:
                return "0.13"


def get_model_config_from_command_line(log_path: Path) -> Optional[Dict[str, int]]:
    """
    Extract model configuration by parsing the pretrain_gpt.py command line in the log file.

    Args:
        log_path: Path to the log file

    Returns:
        dict: Model configuration parameters or None if not found
    """
    # defaults = {"1.3b": {"FFN_HIDDEN_SIZE": 5440}, "1.7b": {"FFN_HIDDEN_SIZE": 8192}}
    defaults = {
        "0.13": {"FFN_HIDDEN_SIZE": 2256},
        "0.4": {"FFN_HIDDEN_SIZE": 3840},
        "1.3": {"FFN_HIDDEN_SIZE": 5440},
        "1.7": {"FFN_HIDDEN_SIZE": 8192},
    }

    try:
        with open(log_path, "r") as f:
            content = f.read()

        # Find the pretrain_gpt.py command line
        match = re.search(r"pretrain_gpt\.py\s+(.+?)(?:\n|$)", content)
        if not match:
            logging.info(f"No pretrain_gpt.py command found in {log_path}")
            return None

        command_line = match.group(1)

        # Extract relevant parameters
        config = {}

        # Map of parameter names to their config keys
        param_map = {
            "--num-layers": "NUM_LAYERS",
            "--hidden-size": "HIDDEN_SIZE",
            "--ffn-hidden-size": "FFN_HIDDEN_SIZE",
            "--num-attention-heads": "NUM_ATTN_HEADS",
            "--seq-length": "SEQ_LENGTH",
            "--max-position-embeddings": "MAX_POSITION_EMBEDDINGS",
        }

        for param, config_key in param_map.items():
            param_match = re.search(f"{param}\\s+(\\d+)", command_line)
            if param_match:
                config[config_key] = int(param_match.group(1))

        model_size = extract_model_size(log_path=log_path)
        logging.info(f"model size is {model_size}")

        try:
            for v in param_map.values():
                if v not in config:
                    config[v] = defaults[model_size][v]
                    print(f"setting {v} as :{defaults[model_size][v]}")
        except Exception as e:
            logging.error(e)

        if not config:
            logging.info(
                f"No model configuration parameters found in command line: {log_path}"
            )
            return None

        return config

    except Exception as e:
        logging.error(f"Error parsing command line from log file {log_path}: {e}")
        return None


def convert_checkpoint(
    log_path: Path,
    conversion_script_path: Path,
    iterations: List[str],
    save_checkpoints_dir: str,
    opensci_megatron_path: str,
    open_sci_hf_path: str,
    convert_logs_dir: str,
    account: str,
    partition: str,
    container_image: str,
    model_config: Optional[Dict[str, int]] = None,
) -> None:
    """
    Submit a job to convert a checkpoint.

    Args:
        log_path: Path to the log file
        conversion_script_path: Path to the conversion script
        iterations: List of iterations to convert
        save_checkpoints_dir: Directory to save converted checkpoints
        opensci_megatron_path: Path to Megatron-LM-Open-Sci repository
        open_sci_hf_path: Path to Open-Sci-hf repository
        convert_logs_dir: Directory to save conversion logs
        account: Account to use for conversion
        partition: Partition to use for conversion
        container_image: Container image to use for conversion
        model_config: Model configuration parameters (optional)
    """
    model_config = get_model_config_from_command_line(log_path)

    if not model_config:
        logging.info(
            f"Skipping {log_path.name}, could not determine model configuration"
        )
        return

    # Check if already converted
    log_base_name = log_path.name.split(".out")[0]
    if os.path.exists(f"{save_checkpoints_dir}/{log_base_name}"):
        # logging.info(f"Skipping {log_base_name}, checkpoint already converted")
        return

    with open(conversion_script_path, "r") as f:
        conversion_script = f.read()

    try:
        conversion_script = conversion_script.format(
            train_logs=log_path,
            iters_to_convert=" ".join(iterations),
            opensci_megatron_path=opensci_megatron_path,
            num_layers=model_config["NUM_LAYERS"],
            num_attn_heads=model_config["NUM_ATTN_HEADS"],
            ffn_hidden_size=model_config["FFN_HIDDEN_SIZE"],
            max_seq_length=model_config["MAX_POSITION_EMBEDDINGS"],
            open_sci_hf_path=open_sci_hf_path,
            save_checkpoints_dir=save_checkpoints_dir,
            convert_logs_dir=convert_logs_dir,
            account=account,
            partition=partition,
            container_image=container_image,
        )

        temp_script_path = Path("/tmp/convert_script.sh")
        with open(temp_script_path, "w") as f:
            f.write(conversion_script)

        # Make the script executable
        os.chmod(temp_script_path, 0o755)

        # Run the bash script
        logging.info("Running script")
        subprocess.run(["bash", temp_script_path])
    except Exception as e:
        logging.error(f"Error running script: {e}")
        logging.info(f"Skipping checkpoint: {log_path}")


def get_iterations_from_checkpoint(checkpoint_path: Path) -> List[str]:
    """
    Determine iterations by scanning the checkpoint directory structure.
    Looks for directories in the format 'iter_0002000', etc.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        List of iteration strings (e.g., ['0002000', '0004000', ...])
    """
    iterations = []

    # Check if checkpoint_path exists and is a directory
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        logging.warning(
            f"Warning: Checkpoint path {checkpoint_path} does not exist or is not a directory"
        )
        raise ValueError(
            f"Checkpoint path {checkpoint_path} does not exist or is not a directory"
        )

    # Look for iteration directories (format: iter_XXXXXXX)
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith("iter_"):
            iter_num = item.name.split("_")[1]
            if iter_num.isdigit():
                iterations.append(iter_num)

    # Sort iterations numerically
    iterations.sort(key=int)

    if not iterations:
        logging.info(f"No iterations found in {checkpoint_path}")
    return iterations


def process_all_checkpoints(
    slurm_log_dir: Union[str, Path],
    conversion_script_path: str,
    save_checkpoints_dir: str,
    opensci_megatron_path: str,
    open_sci_hf_path: str,
    convert_logs_dir: str,
    account: str,
    partition: str,
    container_image: str,
) -> None:
    """
    Process all checkpoints found in the logs.

    Args:
        base_checkpoint_dir: Directory containing checkpoints
        slurm_log_dir: Directory containing SLURM log files
        save_checkpoints_dir: Directory to save converted checkpoints
        opensci_megatron_path: Path to Megatron-LM-Open-Sci repository
        iterations: List of iterations to convert. If None, iterations will be
                    determined from checkpoint directories.
    """
    # Extract checkpoints
    checkpoint_paths_and_logs = extract_checkpoints_from_logs(slurm_log_dir)

    # Process each checkpoint
    for checkpoint_path, log_path in checkpoint_paths_and_logs:
        # If iterations not provided, determine them from the checkpoint directory
        checkpoint_iterations = get_iterations_from_checkpoint(checkpoint_path)
        # If still no iterations found, use a fallback method
        if not len(checkpoint_iterations) > 0:
            logging.info(
                f"No iterations found in {checkpoint_path}, using default range"
            )
            continue

        convert_checkpoint(
            log_path=log_path,
            iterations=checkpoint_iterations,
            conversion_script_path=conversion_script_path,
            save_checkpoints_dir=save_checkpoints_dir,
            opensci_megatron_path=opensci_megatron_path,
            open_sci_hf_path=open_sci_hf_path,
            convert_logs_dir=convert_logs_dir,
            account=account,
            partition=partition,
            container_image=container_image,
        )


if __name__ == "__main__":
    # Process all checkpoints with default settings

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slurm_log_dir",
        type=str,
        default="/leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/slurm_output/",
    )
    parser.add_argument(
        "--conversion_script_path",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/timur_megatron_open_sci/scripts/ckpt/convert_full/convert_full.sh",
    )
    parser.add_argument(
        "--save_checkpoints_dir",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/converted_checkpoints",
    )
    parser.add_argument(
        "--opensci_megatron_path",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/timur_megatron_open_sci",
    )
    parser.add_argument(
        "--open_sci_hf_path",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/Open-Sci-hf",
    )

    parser.add_argument(
        "--convert_logs_dir",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/slurm_logs",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="EUHPC_E03_068",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="boost_usr_prod",
    )
    parser.add_argument(
        "--container_image",
        type=str,
        default="/leonardo_work/EUHPC_E03_068/shared/container_images/pytorch_24.09-py3_leonardo.sif",
    )

    args = parser.parse_args()
    process_all_checkpoints(
        slurm_log_dir=args.slurm_log_dir,
        conversion_script_path=args.conversion_script_path,
        save_checkpoints_dir=args.save_checkpoints_dir,
        opensci_megatron_path=args.opensci_megatron_path,
        open_sci_hf_path=args.open_sci_hf_path,
        convert_logs_dir=args.convert_logs_dir,
        account=args.account,
        partition=args.partition,
        container_image=args.container_image,
    )
