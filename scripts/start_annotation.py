"""
Label Studio Startup Script

Convenience script to start Label Studio annotation server and automatically
open the web interface in the default browser.

Usage:
    python scripts/start_annotation.py [options]

Examples:
    # Start with default settings
    python scripts/start_annotation.py
    
    # Start on custom port
    python scripts/start_annotation.py --port 8090
    
    # Start without auto-opening browser
    python scripts/start_annotation.py --no-browser
"""

import argparse
import logging
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Start Label Studio annotation server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to run Label Studio on (default: 8080)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind to (default: localhost)'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Custom data directory for Label Studio (optional)'
    )
    
    return parser.parse_args()


def check_label_studio_installed() -> bool:
    """
    Check if Label Studio is installed.
    
    Returns:
        True if installed, False otherwise
    """
    try:
        result = subprocess.run(
            ['label-studio', 'version'],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_label_studio(port: int, host: str, data_dir: str = None) -> subprocess.Popen:
    """
    Start Label Studio server.
    
    Args:
        port: Port to run on
        host: Host to bind to
        data_dir: Custom data directory (optional)
        
    Returns:
        Subprocess object
    """
    cmd = ['label-studio', 'start', '--port', str(port), '--host', host]
    
    if data_dir:
        cmd.extend(['--data-dir', data_dir])
    
    logger.info(f"Starting Label Studio on {host}:{port}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process


def open_browser(url: str, delay: int = 5):
    """
    Open browser after a delay.
    
    Args:
        url: URL to open
        delay: Seconds to wait before opening
    """
    logger.info(f"Waiting {delay} seconds for server to start...")
    time.sleep(delay)
    
    logger.info(f"Opening browser at {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.warning(f"Could not open browser automatically: {e}")
        logger.info(f"Please manually open: {url}")


def print_instructions(url: str):
    """Print usage instructions."""
    logger.info("=" * 80)
    logger.info("Label Studio is now running!")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"📍 Access URL: {url}")
    logger.info("")
    logger.info("📚 Quick Start:")
    logger.info("  1. Create an account or login")
    logger.info("  2. Create a new project: 'GTA_V_License_Plate_Detection'")
    logger.info("  3. Choose 'Object Detection with Bounding Boxes' template")
    logger.info("  4. Add label class: 'license_plate'")
    logger.info("  5. Import images from: outputs/test_images/")
    logger.info("  6. Start annotating!")
    logger.info("")
    logger.info("📖 Documentation:")
    logger.info("  - Annotation Guide: docs/annotation_guide.md")
    logger.info("  - Label Studio Docs: https://labelstud.io/guide/")
    logger.info("")
    logger.info("⌨️  Press Ctrl+C to stop the server")
    logger.info("=" * 80)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Check if Label Studio is installed
        logger.info("Checking Label Studio installation...")
        if not check_label_studio_installed():
            logger.error("Label Studio is not installed!")
            logger.error("Please install it using: pip install label-studio")
            return 1
        
        logger.info("✓ Label Studio is installed")
        
        # Construct URL
        url = f"http://{args.host}:{args.port}"
        
        # Start Label Studio
        process = start_label_studio(args.port, args.host, args.data_dir)
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is not None:
            # Process ended, check for errors
            stdout, stderr = process.communicate()
            logger.error("Label Studio failed to start!")
            if stderr:
                logger.error(f"Error: {stderr}")
            return 1
        
        # Open browser if requested
        if not args.no_browser:
            open_browser(url)
        
        # Print instructions
        print_instructions(url)
        
        # Wait for process to complete (user will Ctrl+C)
        try:
            process.wait()
        except KeyboardInterrupt:
            logger.info("\n\nShutting down Label Studio...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing Label Studio process...")
                process.kill()
            logger.info("Label Studio stopped successfully")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        return 0
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
