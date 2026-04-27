"""Logging Configuration Utilities."""

import logging
import os
import sys

def setup_logger(output_dir: str, rank: int, filename: str = "train.log"):
    """
    Configures the logger.
    Only rank 0 logs to file and stdout. Others log warnings only.
    """
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, filename)
        
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[file_handler, stream_handler],
            force=True
        )
    else:
        logging.basicConfig(level=logging.WARNING)
