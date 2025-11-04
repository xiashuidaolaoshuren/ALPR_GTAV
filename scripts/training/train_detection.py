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


def save_training_metrics(results_csv: Path, metrics_csv: Path, existing_history: Optional[pd.DataFrame] = None, test_metrics: Optional[dict] = None):
    """
    Extract and save training metrics from Ultralytics results CSV.
    
    Args:
        results_csv: Path to Ultralytics results.csv
        metrics_csv: Path to save cleaned metrics CSV
        existing_history: Previous training history (for resume)
        test_metrics: Test set metrics dict (precision, recall, mAP50, mAP50_95)
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
        
        # Add test metrics columns (initialize as None for all rows)
        df_clean['test_precision'] = None
        df_clean['test_recall'] = None
        df_clean['test_mAP50'] = None
        df_clean['test_mAP50_95'] = None
        
        # Fill test metrics for the last epoch if provided
        if test_metrics is not None and not df_clean.empty:
            last_idx = df_clean.index[-1]
            df_clean.loc[last_idx, 'test_precision'] = test_metrics.get('precision')
            df_clean.loc[last_idx, 'test_recall'] = test_metrics.get('recall')
            df_clean.loc[last_idx, 'test_mAP50'] = test_metrics.get('mAP50')
            df_clean.loc[last_idx, 'test_mAP50_95'] = test_metrics.get('mAP50_95')
        
        # If resuming, concatenate with existing history
        if existing_history is not None:
            # Ensure existing history has test metric columns (backward compatibility)
            for col in ['test_precision', 'test_recall', 'test_mAP50', 'test_mAP50_95']:
                if col not in existing_history.columns:
                    existing_history[col] = None
            
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
        if test_metrics:
            logger.info(f"  Included test metrics: Precision={test_metrics['precision']:.4f}, Recall={test_metrics['recall']:.4f}, mAP50={test_metrics['mAP50']:.4f}")
        
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
            ax.plot(df['epoch'], df['precision'], label='Validation Precision', linewidth=2, marker='o', markersize=4)
            ax.plot(df['epoch'], df['recall'], label='Validation Recall', linewidth=2, marker='s', markersize=4)
            
            # Add test metrics if available (only shown at final epoch)
            if 'test_precision' in df.columns and 'test_recall' in df.columns:
                test_data = df[df['test_precision'].notna()]
                if not test_data.empty:
                    ax.scatter(test_data['epoch'], test_data['test_precision'], 
                              label='Test Precision', s=100, marker='*', c='red', zorder=5)
                    ax.scatter(test_data['epoch'], test_data['test_recall'], 
                              label='Test Recall', s=100, marker='X', c='orange', zorder=5)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Precision and Recall (Validation + Test)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            plt.tight_layout()
            plot_path = output_dir / 'precision_recall.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved precision/recall plot to {plot_path}")
        
        # Plot 3: Test Metrics (if available)
        if 'test_mAP50' in df.columns and 'test_mAP50_95' in df.columns:
            test_data = df[df['test_mAP50'].notna()]
            if not test_data.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot validation mAP as lines
                if 'mAP50' in df.columns and 'mAP50_95' in df.columns:
                    ax.plot(df['epoch'], df['mAP50'], label='Val mAP@0.5', 
                           linewidth=2, marker='o', markersize=3, alpha=0.7)
                    ax.plot(df['epoch'], df['mAP50_95'], label='Val mAP@0.5:0.95', 
                           linewidth=2, marker='s', markersize=3, alpha=0.7)
                
                # Overlay test metrics as stars at final epoch
                ax.scatter(test_data['epoch'], test_data['test_mAP50'], 
                          label='Test mAP@0.5', s=150, marker='*', c='red', zorder=5)
                ax.scatter(test_data['epoch'], test_data['test_mAP50_95'], 
                          label='Test mAP@0.5:0.95', s=150, marker='X', c='orange', zorder=5)
                
                # Annotate test values
                for _, row in test_data.iterrows():
                    ax.annotate(f"{row['test_mAP50']:.3f}", 
                               (row['epoch'], row['test_mAP50']),
                               textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                    ax.annotate(f"{row['test_mAP50_95']:.3f}", 
                               (row['epoch'], row['test_mAP50_95']),
                               textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
                
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('mAP', fontsize=12)
                ax.set_title('Mean Average Precision (Validation + Test)', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1])
                
                plt.tight_layout()
                plot_path = output_dir / 'test_metrics.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved test metrics plot to {plot_path}")
        
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
            'workers': 4,
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
        
        # Post-training test evaluation
        logger.info("="*80)
        logger.info("Running post-training test evaluation...")
        test_results = model.val(split='test')
        
        # Extract test metrics
        test_metrics = {
            'precision': float(test_results.box.mp),
            'recall': float(test_results.box.mr),
            'mAP50': float(test_results.box.map50),
            'mAP50_95': float(test_results.box.map)
        }
        
        # Save and process metrics (use actual save directory)
        results_csv = actual_save_dir / 'results.csv'
        actual_metrics_csv = actual_save_dir / 'training_metrics.csv'
        df_metrics = save_training_metrics(results_csv, actual_metrics_csv, existing_history, test_metrics)
        
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
        logger.info(f"\n📊 Validation Metrics:")
        logger.info(f"  mAP@0.5: {val_results.box.map50:.4f}")
        logger.info(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")
        logger.info(f"  Precision: {val_results.box.mp:.4f}")
        logger.info(f"  Recall: {val_results.box.mr:.4f}")
        logger.info(f"\n🧪 Test Metrics:")
        logger.info(f"  mAP@0.5: {test_metrics['mAP50']:.4f}")
        logger.info(f"  mAP@0.5:0.95: {test_metrics['mAP50_95']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
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
