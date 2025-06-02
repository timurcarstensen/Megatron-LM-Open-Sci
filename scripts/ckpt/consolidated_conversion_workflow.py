#!/usr/bin/env python3
"""
Consolidated Checkpoint Converter Script

This script consolidates the functionality of:
- checkpoint_conversion_workflow.py
- convert_full.sh
- converter.py

It extracts and processes checkpoint paths from SLURM log files,
identifies completed training runs, and submits batch jobs to convert the checkpoints.
"""

import argparse
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

# Import functions from the original checkpoint_conversion_workflow.py
from checkpoint_conversion_workflow import (
    extract_checkpoints_from_logs,
    get_model_config_from_command_line,
    get_iterations_from_checkpoint,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def convert_checkpoint_consolidated(
    log_path: Path,
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
    Convert a checkpoint using the consolidated approach.
    This replaces the bash script + converter.py approach with direct Python calls.

    Args:
        log_path: Path to the log file
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

    import pdb

    pdb.set_trace()

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

    # Create necessary directories
    os.makedirs(convert_logs_dir, exist_ok=True)
    os.makedirs(save_checkpoints_dir, exist_ok=True)

    # Load the SBATCH template
    sbatch_template_path = os.path.join(
        opensci_megatron_path, "scripts/ckpt/convert_full/template.sbatch"
    )

    try:
        with open(sbatch_template_path, "r") as f:
            sbatch_template = f.read()
            # escape ${} in f-strings with double curly braces
            # escape cat <<EOF > ../config.json\n{...}\nEOF
            cat_eof_data = re.search(
                r"cat <<EOF.*?EOF", sbatch_template, re.DOTALL
            ).group()
            sbatch_template = sbatch_template.replace(cat_eof_data, "<cat_eof_data>")
            sbatch_template = re.sub(
                r"\$\{(.+?)\}", r"\${{\1}}", sbatch_template
            ).replace("\$", "$")

        # Process each iteration
        for iteration in iterations:
            logging.info(f"Converting iteration {iteration}")
            sbatch_script = sbatch_template.format(
                account=account,
                partition=partition,
                container_image=container_image,
                opensci_megatron_path=opensci_megatron_path,
                open_sci_hf_path=open_sci_hf_path,
                train_logs_path=str(log_path),
                save_checkpoints_dir=save_checkpoints_dir,
                convert_logs_dir=convert_logs_dir,
                pre_run_cmd="",  # No pre-run command by default
                iteration_to_convert=iteration,
                num_layers=model_config["NUM_LAYERS"],
                num_attn_heads=model_config["NUM_ATTN_HEADS"],
                ffn_hidden_size=model_config["FFN_HIDDEN_SIZE"],
                max_seq_length=model_config["MAX_POSITION_EMBEDDINGS"],
            )

            sbatch_script = sbatch_script.replace("<cat_eof_data>", cat_eof_data)

            sbatch_script_path = os.path.join(
                convert_logs_dir, f"convert_{log_path.name}_{iteration}.sbatch"
            )
            with open(sbatch_script_path, "w") as f:
                f.write(sbatch_script)

            subprocess.run(["sbatch", sbatch_script_path])
            logging.info(f"Submitted {sbatch_script_path}")
            logging.info(f"Submitted {sbatch_script_path}")

    except Exception as e:
        logging.error(f"Error converting checkpoint: {e}")
        logging.info(f"Skipping checkpoint: {log_path}")


def process_all_checkpoints_consolidated(
    slurm_log_dir: Union[str, Path],
    save_checkpoints_dir: str,
    opensci_megatron_path: str,
    open_sci_hf_path: str,
    convert_logs_dir: str,
    account: str,
    partition: str,
    container_image: str,
) -> None:
    """
    Process all checkpoints found in the logs using the consolidated approach.

    Args:
        slurm_log_dir: Directory containing SLURM log files
        save_checkpoints_dir: Directory to save converted checkpoints
        opensci_megatron_path: Path to Megatron-LM-Open-Sci repository
        open_sci_hf_path: Path to Open-Sci-hf repository
        convert_logs_dir: Directory to save conversion logs
        account: Account to use for conversion
        partition: Partition to use for conversion
        container_image: Container image to use for conversion
    """
    # Extract checkpoints
    checkpoint_paths_and_logs = extract_checkpoints_from_logs(slurm_log_dir)

    # Process each checkpoint
    for checkpoint_path, log_path in checkpoint_paths_and_logs:
        # Determine iterations from the checkpoint directory
        checkpoint_iterations = get_iterations_from_checkpoint(checkpoint_path)

        # If no iterations found, skip this checkpoint
        if not len(checkpoint_iterations) > 0:
            logging.info(f"No iterations found in {checkpoint_path}, skipping")
            continue

        convert_checkpoint_consolidated(
            log_path=log_path,
            iterations=checkpoint_iterations,
            save_checkpoints_dir=save_checkpoints_dir,
            opensci_megatron_path=opensci_megatron_path,
            open_sci_hf_path=open_sci_hf_path,
            convert_logs_dir=convert_logs_dir,
            account=account,
            partition=partition,
            container_image=container_image,
        )


def main():
    """Main function that handles command line arguments and runs the consolidated workflow."""
    parser = argparse.ArgumentParser(
        description="Consolidated checkpoint conversion workflow"
    )

    # Arguments from original checkpoint_conversion_workflow.py
    parser.add_argument(
        "--slurm_log_dir",
        type=str,
        default="/leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/slurm_output/",
        help="Directory containing SLURM log files",
    )
    parser.add_argument(
        "--save_checkpoints_dir",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/converted_checkpoints",
        help="Directory to save converted checkpoints",
    )
    parser.add_argument(
        "--opensci_megatron_path",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/timur_megatron_open_sci",
        help="Path to Megatron-LM-Open-Sci repository",
    )
    parser.add_argument(
        "--open_sci_hf_path",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/Open-Sci-hf",
        help="Path to Open-Sci-hf repository",
    )
    parser.add_argument(
        "--convert_logs_dir",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/slurm_logs",
        help="Directory to save conversion logs",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="EUHPC_E03_068",
        help="Account to use for conversion",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="boost_usr_prod",
        help="Partition to use for conversion",
    )
    parser.add_argument(
        "--container_image",
        type=str,
        default="/leonardo_work/EUHPC_E03_068/shared/container_images/pytorch_24.09-py3_leonardo.sif",
        help="Container image to use for conversion",
    )

    args = parser.parse_args()

    # Run the consolidated workflow
    process_all_checkpoints_consolidated(
        slurm_log_dir=args.slurm_log_dir,
        save_checkpoints_dir=args.save_checkpoints_dir,
        opensci_megatron_path=args.opensci_megatron_path,
        open_sci_hf_path=args.open_sci_hf_path,
        convert_logs_dir=args.convert_logs_dir,
        account=args.account,
        partition=args.partition,
        container_image=args.container_image,
    )


if __name__ == "__main__":
    main()
