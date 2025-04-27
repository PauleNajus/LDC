#!/usr/bin/env python
import os
import argparse
import glob
import sys
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.advanced_visualizations import generate_advanced_visualizations

def find_latest_files(base_dir):
    """Find the latest history and metrics files in a directory"""
    # Find all history files
    history_files = glob.glob(os.path.join(base_dir, "train_history_*.json"))
    # Find all metrics files
    metrics_files = glob.glob(os.path.join(base_dir, "test_metrics_*.json"))
    
    if not history_files or not metrics_files:
        return None, None
    
    # Sort by modification time (newest first)
    latest_history = max(history_files, key=os.path.getmtime)
    latest_metrics = max(metrics_files, key=os.path.getmtime)
    
    return latest_history, latest_metrics

def main():
    parser = argparse.ArgumentParser(description="Generate advanced visualizations for model metrics")
    parser.add_argument("--logs-dir", default="train_results_logs", 
                        help="Directory containing training logs (default: train_results_logs)")
    parser.add_argument("--history-file", help="Specific history file to use (optional)")
    parser.add_argument("--metrics-file", help="Specific metrics file to use (optional)")
    parser.add_argument("--output-dir", help="Output directory for visualizations (default: same as logs dir)")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.logs_dir
    
    # Get history and metrics files
    if args.history_file and args.metrics_file:
        history_path = args.history_file
        metrics_path = args.metrics_file
    else:
        history_path, metrics_path = find_latest_files(args.logs_dir)
        
        if not history_path or not metrics_path:
            print(f"Error: Could not find history and metrics files in {args.logs_dir}")
            return 1
    
    print(f"Using history file: {history_path}")
    print(f"Using metrics file: {metrics_path}")
    print(f"Output directory: {output_dir}")
    
    # Generate visualizations
    success = generate_advanced_visualizations(history_path, metrics_path, output_dir)
    
    if success:
        print("\nVisualization generation completed successfully!")
        return 0
    else:
        print("\nError occurred during visualization generation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 