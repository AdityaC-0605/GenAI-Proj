#!/usr/bin/env python3
"""Data download script for Cross-Lingual QA datasets."""

import argparse
import logging
import os
from pathlib import Path
import urllib.request
import json
import zipfile
import tarfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DATASET_URLS = {
    'squad': {
        'train': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
        'dev': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
    },
    'xquad': {
        'url': 'https://github.com/deepmind/xquad/archive/refs/heads/master.zip',
        'type': 'zip'
    },
    'mlqa': {
        'url': 'https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip',
        'type': 'zip'
    },
    'tydiqa': {
        'url': 'https://storage.googleapis.com/tydiqa/v1.0/tydiqa-v1.0-train.jsonl.gz',
        'type': 'gz'
    }
}


def download_file(url: str, output_path: str):
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        output_path: Path to save file
    """
    logger.info(f"Downloading {url}")
    logger.info(f"Saving to {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info("Download complete")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def extract_zip(zip_path: str, extract_to: str):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info("Extraction complete")


def extract_tar_gz(tar_path: str, extract_to: str):
    """
    Extract a tar.gz file.
    
    Args:
        tar_path: Path to tar.gz file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting {tar_path}")
    
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)
    
    logger.info("Extraction complete")


def download_squad(data_dir: str):
    """
    Download SQuAD 2.0 dataset.
    
    Args:
        data_dir: Base data directory
    """
    squad_dir = Path(data_dir) / 'squad'
    squad_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading SQuAD 2.0 dataset")
    
    # Download training data
    train_path = squad_dir / 'train-v2.0.json'
    if not train_path.exists():
        download_file(DATASET_URLS['squad']['train'], str(train_path))
    else:
        logger.info(f"Training data already exists at {train_path}")
    
    # Download dev data
    dev_path = squad_dir / 'dev-v2.0.json'
    if not dev_path.exists():
        download_file(DATASET_URLS['squad']['dev'], str(dev_path))
    else:
        logger.info(f"Dev data already exists at {dev_path}")
    
    logger.info(f"SQuAD 2.0 dataset downloaded to {squad_dir}")


def download_xquad(data_dir: str):
    """
    Download XQuAD dataset.
    
    Args:
        data_dir: Base data directory
    """
    xquad_dir = Path(data_dir) / 'xquad'
    xquad_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading XQuAD dataset")
    
    # Download zip file
    zip_path = xquad_dir / 'xquad-master.zip'
    if not zip_path.exists():
        download_file(DATASET_URLS['xquad']['url'], str(zip_path))
    else:
        logger.info(f"XQuAD zip already exists at {zip_path}")
    
    # Extract
    extract_zip(str(zip_path), str(xquad_dir))
    
    logger.info(f"XQuAD dataset downloaded to {xquad_dir}")


def download_mlqa(data_dir: str):
    """
    Download MLQA dataset.
    
    Args:
        data_dir: Base data directory
    """
    mlqa_dir = Path(data_dir) / 'mlqa'
    mlqa_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading MLQA dataset")
    
    # Download zip file
    zip_path = mlqa_dir / 'MLQA_V1.zip'
    if not zip_path.exists():
        download_file(DATASET_URLS['mlqa']['url'], str(zip_path))
    else:
        logger.info(f"MLQA zip already exists at {zip_path}")
    
    # Extract
    extract_zip(str(zip_path), str(mlqa_dir))
    
    logger.info(f"MLQA dataset downloaded to {mlqa_dir}")


def download_tydiqa(data_dir: str):
    """
    Download TyDi QA dataset.
    
    Args:
        data_dir: Base data directory
    """
    tydiqa_dir = Path(data_dir) / 'tydiqa'
    tydiqa_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading TyDi QA dataset")
    
    # Download gz file
    gz_path = tydiqa_dir / 'tydiqa-v1.0-train.jsonl.gz'
    if not gz_path.exists():
        download_file(DATASET_URLS['tydiqa']['url'], str(gz_path))
    else:
        logger.info(f"TyDi QA file already exists at {gz_path}")
    
    # Extract gzipped jsonl file
    import gzip
    import shutil
    
    jsonl_path = tydiqa_dir / 'tydiqa-v1.0-train.jsonl'
    if not jsonl_path.exists():
        logger.info(f"Extracting {gz_path}")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(jsonl_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info("Extraction complete")
    else:
        logger.info(f"Extracted file already exists at {jsonl_path}")
    
    logger.info(f"TyDi QA dataset downloaded to {tydiqa_dir}")


def download_all(data_dir: str):
    """
    Download all datasets.
    
    Args:
        data_dir: Base data directory
    """
    logger.info("Downloading all datasets")
    
    download_squad(data_dir)
    download_xquad(data_dir)
    download_mlqa(data_dir)
    download_tydiqa(data_dir)
    
    logger.info("All datasets downloaded successfully")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download Cross-Lingual QA datasets"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['squad', 'xquad', 'mlqa', 'tydiqa', 'all'],
        default='all',
        help='Dataset to download'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory to save datasets'
    )
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Data directory: {data_dir.absolute()}")
    
    # Download requested dataset
    if args.dataset == 'squad':
        download_squad(str(data_dir))
    elif args.dataset == 'xquad':
        download_xquad(str(data_dir))
    elif args.dataset == 'mlqa':
        download_mlqa(str(data_dir))
    elif args.dataset == 'tydiqa':
        download_tydiqa(str(data_dir))
    elif args.dataset == 'all':
        download_all(str(data_dir))
    
    logger.info("Download complete!")
    logger.info(f"Datasets saved to: {data_dir.absolute()}")


if __name__ == "__main__":
    main()
