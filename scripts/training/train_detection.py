"""
YOLOv8 License Plate Detection Training Script

Fine-tune YOLOv8 detection model on GTA V dataset with training metrics logging,
visualization, and resume support.

Usage:
    # Basic training
    python scripts/training/train_detection.py
    
    # Custom parameters
    python scripts/training/train_detection.py --epochs 100 --batch 16 --device cuda
    
    # Resume from checkpoint (automatically loads training history)
    python scripts/training/train_detection.py --resume runs/detect/gta_v_lpr/weights/last.pt
    
    # Custom configuration
    python scripts/training/train_detection.py --config configs/pipeline_config.yaml --data datasets/lpr/data.yaml

Examples:
    # Quick test run
    python scripts/training/train_detection.py --epochs 1 --batch 4
    
    # Full training with visualization
    python scripts/training/train_detection.py --epochs 50 --batch 16 --save-plots
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# Ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable for progress bars
train_pbar = None
val_pbar = None


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 detection model on GTA V license plate dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Dataset arguments
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file (default: configs/pipeline_config.yaml)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='datasets/lpr/data.yaml',
        help='Path to dataset YAML file (default: datasets/lpr/data.yaml)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training (default: 640)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu/auto, overrides config)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience epochs (default: 10)'
    )
    
    # Output arguments
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Output project directory (default: runs/detect)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='gta_v_lpr',
        help='Experiment name (default: gta_v_lpr)'
    )
    
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Allow overwriting existing experiment'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Visualization arguments
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save training metric plots'
    )
    
    parser.add_argument(
        '--plots-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: <project>/<name>/plots)'
    )
    
    return parser.parse_args()


def load_configuration(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def load_training_history(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load training history from CSV file (for resume training).
    
    Args:
        csv_path: Path to training metrics CSV file
        
    Returns:
        DataFrame with training history or None if file doesn't exist
    """
    if not csv_path.exists():
        logger.info("No existing training history found")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded training history: {len(df)} epochs from {csv_path}")
        return df
    except Exception as e:
        logger.warning(f"Could not load training history: {e}")
        return None


def on_train_epoch_start(trainer):
    """
    Callback at the start of each training epoch.
    Creates a progress bar for training batches.
    """
    global train_pbar
    # Close previous progress bar if exists
    if train_pbar is not None:
        train_pbar.close()
    
    # Create new progress bar for this epoch
    epoch = trainer.epoch + 1
    total_epochs = trainer.epochs
    train_pbar = tqdm(
        total=len(trainer.train_loader),
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        unit="batch",
        leave=True,
        dynamic_ncols=True
    )


def on_train_batch_end(trainer):
    """
    Callback at the end of each training batch.
    Updates the progress bar with current metrics.
    """
    global train_pbar
    if train_pbar is not None:
        # Get current losses
        loss_items = trainer.loss_items if hasattr(trainer, 'loss_items') else None
        
        # Format postfix with losses
        if loss_items is not None and len(loss_items) >= 3:
            train_pbar.set_postfix({
                'box_loss': f'{loss_items[0]:.4f}',
                'cls_loss': f'{loss_items[1]:.4f}',
                'dfl_loss': f'{loss_items[2]:.4f}'
            })
        
        train_pbar.update(1)


def on_train_epoch_end(trainer):
    """
    Callback at the end of each training epoch.
    Closes the training progress bar and displays epoch summary.
    """
    global train_pbar
    if train_pbar is not None:
        train_pbar.close()
        train_pbar = None


def on_val_start(validator):
    """
    Callback at the start of validation.
    Creates a progress bar for validation batches.
    """
    global val_pbar
    # Close previous progress bar if exists
    if val_pbar is not None:
        val_pbar.close()
    
    # Create new progress bar for validation
    val_pbar = tqdm(
        total=len(validator.dataloader),
        desc="Validation",
        unit="batch",
        leave=True,
        dynamic_ncols=True
    )


def on_val_batch_end(validator):
    """
    Callback at the end of each validation batch.
    Updates the progress bar.
    """
    global val_pbar
    if val_pbar is not None:
        val_pbar.update(1)


def on_val_end(validator):
    """
    Callback at the end of validation.
    Closes the validation progress bar and displays metrics.
    """
    global val_pbar
    if val_pbar is not None:
        # Display validation metrics if available
        if hasattr(validator, 'metrics') and validator.metrics is not None:
            metrics = validator.metrics
            if hasattr(metrics, 'box'):
                box_metrics = metrics.box
                val_pbar.set_postfix({
                    'mAP50': f'{box_metrics.map50:.4f}' if hasattr(box_metrics, 'map50') else 'N/A',
                    'mAP50-95': f'{box_metrics.map:.4f}' if hasattr(box_metrics, 'map') else 'N/A',
                    'precision': f'{box_metrics.mp:.4f}' if hasattr(box_metrics, 'mp') else 'N/A',
                    'recall': f'{box_metrics.mr:.4f}' if hasattr(box_metrics, 'mr') else 'N/A'
                })
        
        val_pbar.close()
        val_pbar = None


def save_training_metrics(results_csv: Path, metrics_csv: Path, existing_history: Optional[pd.DataFrame] = None):
    """
    Extract and save training metrics from Ultralytics results CSV.
    
    Args:
        results_csv: Path to Ultralytics results.csv
        metrics_csv: Path to save cleaned metrics CSV
        existing_history: Previous training history (for resume)
    """
    if not results_csv.exists():
        logger.warning(f"Results CSV not found: {results_csv}")
        return
    
    try:
        # Load current training results
        df = pd.read_csv(results_csv)
        
        # Remove leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Select relevant metrics
        metrics_cols = ['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
                        'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']
        
        # Only keep columns that exist
        available_cols = [col for col in metrics_cols if col in df.columns]
        df_clean = df[available_cols].copy()
        
        # Rename columns for clarity
        rename_map = {
            'train/box_loss': 'train_box_loss',
            'train/cls_loss': 'train_cls_loss',
            'train/dfl_loss': 'train_dfl_loss',
            'metrics/precision(B)': 'precision',
            'metrics/recall(B)': 'recall',
            'metrics/mAP50(B)': 'mAP50',
            'metrics/mAP50-95(B)': 'mAP50_95',
            'val/box_loss': 'val_box_loss',
            'val/cls_loss': 'val_cls_loss',
            'val/dfl_loss': 'val_dfl_loss'
        }
        df_clean.rename(columns=rename_map, inplace=True)
        
        # If resuming, concatenate with existing history
        if existing_history is not None:
            # Adjust epoch numbers if needed
            if not df_clean.empty and not existing_history.empty:
                max_prev_epoch = existing_history['epoch'].max()
                df_clean['epoch'] = df_clean['epoch'] + max_prev_epoch
            
            # Concatenate
            df_combined = pd.concat([existing_history, df_clean], ignore_index=True)
        else:
            df_combined = df_clean
        
        # Save combined metrics
        df_combined.to_csv(metrics_csv, index=False)
        logger.info(f"Saved training metrics to {metrics_csv}")
        
        return df_combined
        
    except Exception as e:
        logger.error(f"Error saving training metrics: {e}")
        return None


def plot_training_metrics(metrics_csv: Path, output_dir: Path):
    """
    Plot training metrics from CSV file.
    
    Args:
        metrics_csv: Path to training metrics CSV
        output_dir: Directory to save plots
    """
    if not metrics_csv.exists():
        logger.warning(f"Metrics CSV not found: {metrics_csv}")
        return
    
    try:
        df = pd.read_csv(metrics_csv)
        
        if df.empty:
            logger.warning("No metrics to plot")
            return
        
        # Create plots directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Plot 1: Losses
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics - GTA V License Plate Detection', fontsize=16, fontweight='bold')
        
        # Box Loss
        if 'train_box_loss' in df.columns and 'val_box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['train_box_loss'], label='Train', linewidth=2, marker='o', markersize=3)
            axes[0, 0].plot(df['epoch'], df['val_box_loss'], label='Validation', linewidth=2, marker='s', markersize=3)
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Box Loss', fontsize=12)
            axes[0, 0].set_title('Bounding Box Loss', fontsize=14, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Class Loss
        if 'train_cls_loss' in df.columns and 'val_cls_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['train_cls_loss'], label='Train', linewidth=2, marker='o', markersize=3)
            axes[0, 1].plot(df['epoch'], df['val_cls_loss'], label='Validation', linewidth=2, marker='s', markersize=3)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Class Loss', fontsize=12)
            axes[0, 1].set_title('Classification Loss', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # DFL Loss
        if 'train_dfl_loss' in df.columns and 'val_dfl_loss' in df.columns:
            axes[1, 0].plot(df['epoch'], df['train_dfl_loss'], label='Train', linewidth=2, marker='o', markersize=3)
            axes[1, 0].plot(df['epoch'], df['val_dfl_loss'], label='Validation', linewidth=2, marker='s', markersize=3)
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('DFL Loss', fontsize=12)
            axes[1, 0].set_title('Distribution Focal Loss', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # mAP Metrics
        if 'mAP50' in df.columns and 'mAP50_95' in df.columns:
            axes[1, 1].plot(df['epoch'], df['mAP50'], label='mAP@0.5', linewidth=2, marker='o', markersize=3)
            axes[1, 1].plot(df['epoch'], df['mAP50_95'], label='mAP@0.5:0.95', linewidth=2, marker='s', markersize=3)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('mAP', fontsize=12)
            axes[1, 1].set_title('Mean Average Precision', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        plot_path = output_dir / 'training_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training metrics plot to {plot_path}")
        
        # Plot 2: Precision & Recall
        if 'precision' in df.columns and 'recall' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['epoch'], df['precision'], label='Precision', linewidth=2, marker='o', markersize=4)
            ax.plot(df['epoch'], df['recall'], label='Recall', linewidth=2, marker='s', markersize=4)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Precision and Recall', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            plt.tight_layout()
            plot_path = output_dir / 'precision_recall.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved precision/recall plot to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error plotting training metrics: {e}")


def train_model(args, config: Dict):
    """
    Main training logic for YOLOv8 detection model.
    
    Args:
        args: Parsed command line arguments
        config: Configuration dictionary
        
    Returns:
        Training results
    """
    try:
        # Determine device
        device = args.device if args.device is not None else config['detection']['device']
        logger.info(f"Using device: {device}")
        
        # Setup output directories
        output_dir = Path(args.project) / args.name
        plots_dir = Path(args.plots_dir) if args.plots_dir else output_dir / 'plots'
        metrics_csv = output_dir / 'training_metrics.csv'
        
        # Load existing training history if resuming
        existing_history = None
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            existing_history = load_training_history(metrics_csv)
        
        # Load model
        if args.resume:
            model = YOLO(args.resume)
            logger.info("Loaded checkpoint for resume training")
        else:
            model_path = config['detection']['model_path']
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = YOLO(model_path)
            logger.info(f"Loaded pre-trained model: {model_path}")
        
        # Configure training parameters
        augmentation = config.get('augmentation', {})
        
        train_args = {
            # Dataset
            'data': args.data,
            
            # Training params
            'epochs': args.epochs,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'device': device,
            'patience': args.patience,
            
            # Output
            'project': args.project,
            'name': args.name,
            'exist_ok': args.exist_ok,
            
            # Augmentation
            'hsv_h': 0.015,  # Hue augmentation
            'hsv_s': 0.7,    # Saturation augmentation
            'hsv_v': 0.4,    # Value augmentation
            'degrees': augmentation.get('rotation_limit', 10),
            'flipud': augmentation.get('vertical_flip', 0.0),
            'fliplr': augmentation.get('horizontal_flip', 0.5),
            'mosaic': 1.0 if augmentation.get('mosaic', True) else 0.0,
            
            # Performance
            'workers': 8,
            'cache': False,  # Set to True if enough RAM
            
            # Validation
            'val': True,
            'plots': True,
            'save': True,
            'save_period': -1,  # Save checkpoint every N epochs (-1 = only save last and best)
        }
        
        # Log training configuration
        logger.info("="*80)
        logger.info("Training Configuration:")
        logger.info("="*80)
        for key, value in train_args.items():
            logger.info(f"  {key}: {value}")
        logger.info("="*80)
        
        # Register custom callbacks for enhanced progress bars
        logger.info("Registering custom progress bar callbacks...")
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_val_start", on_val_start)
        model.add_callback("on_val_batch_end", on_val_batch_end)
        model.add_callback("on_val_end", on_val_end)
        
        # Start training
        logger.info("Starting training...")
        results = model.train(**train_args)
        
        # Get actual save directory from results (Ultralytics may auto-increment)
        actual_save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else output_dir
        logger.info(f"Actual output directory: {actual_save_dir}")
        
        # Post-training validation
        logger.info("="*80)
        logger.info("Running post-training validation...")
        val_results = model.val()
        
        # Save and process metrics (use actual save directory)
        results_csv = actual_save_dir / 'results.csv'
        actual_metrics_csv = actual_save_dir / 'training_metrics.csv'
        df_metrics = save_training_metrics(results_csv, actual_metrics_csv, existing_history)
        
        # Plot metrics if requested
        if args.save_plots and df_metrics is not None:
            logger.info("Generating training plots...")
            actual_plots_dir = actual_save_dir / 'plots'
            plot_training_metrics(actual_metrics_csv, actual_plots_dir)
        
        # Copy best model to models/detection/
        best_model_src = actual_save_dir / 'weights' / 'best.pt'
        best_model_dst = Path('models/detection/yolov8_finetuned_best.pt')
        
        if best_model_src.exists():
            best_model_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_model_src, best_model_dst)
            logger.info(f"✅ Best model copied to: {best_model_dst}")
        else:
            logger.warning("Best model not found for copying")
        
        # Print final summary
        logger.info("="*80)
        logger.info("Training Summary:")
        logger.info("="*80)
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"  Total epochs: {args.epochs}")
        logger.info(f"  Best mAP@0.5: {val_results.box.map50:.4f}")
        logger.info(f"  Best mAP@0.5:0.95: {val_results.box.map:.4f}")
        logger.info(f"  Precision: {val_results.box.mp:.4f}")
        logger.info(f"  Recall: {val_results.box.mr:.4f}")
        logger.info(f"\n📁 Output directory: {actual_save_dir}")
        logger.info(f"📊 Metrics CSV: {actual_metrics_csv}")
        if args.save_plots:
            logger.info(f"📈 Plots directory: {actual_plots_dir}")
        logger.info(f"🏆 Best model: {best_model_dst}")
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_configuration(args.config)
        
        # Verify dataset exists
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
        
        logger.info(f"Dataset configuration: {data_path}")
        
        # Train model
        train_model(args, config)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
