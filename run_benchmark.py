#!/usr/bin/env python3
"""
Simple runner script for benchmarking label-shift estimation methods.

This script provides easy commands for common benchmarking tasks.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_single_evaluation(results_dir: str, model: str = "llama-1"):
    """Evaluate a single method"""
    cmd = [
        sys.executable, "benchmark_evaluation.py",
        "--results_dir", results_dir,
        "--ground_truth", f"bench/specs/{model}.yaml",
        "--output_dir", f"benchmark_output/{Path(results_dir).name}"
    ]
    
    print(f"Running evaluation for {results_dir}...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Evaluation completed successfully!")
        print(result.stdout)
    else:
        print("❌ Evaluation failed!")
        print(result.stderr)
        return False
    
    return True


def run_comparison(results_dirs: list, model: str = "llama-1"):
    """Compare multiple methods"""
    cmd = [
        sys.executable, "benchmark_evaluation.py",
        "--compare"
    ] + results_dirs + [
        "--ground_truth", f"bench/specs/{model}.yaml",
        "--output_dir", "benchmark_output/comparison"
    ]
    
    print(f"Running comparison for {len(results_dirs)} methods...")
    print(f"Methods: {', '.join([Path(d).name for d in results_dirs])}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Comparison completed successfully!")
        print(result.stdout)
    else:
        print("❌ Comparison failed!")
        print(result.stderr)
        return False
    
    return True


def list_available_results():
    """List available result directories"""
    out_dir = Path("out")
    if not out_dir.exists():
        print("No 'out' directory found. Run some experiments first!")
        return []
    
    result_dirs = [d for d in out_dir.iterdir() if d.is_dir() and (d / "summary.json").exists()]
    
    if not result_dirs:
        print("No result directories with summary.json found in 'out/'")
        return []
    
    print("Available result directories:")
    for i, d in enumerate(result_dirs, 1):
        print(f"  {i}. {d.name}")
    
    return [str(d) for d in result_dirs]


def main():
    parser = argparse.ArgumentParser(description="Easy benchmark runner for label-shift estimation")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available result directories")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a single method")
    eval_parser.add_argument("results_dir", help="Path to results directory")
    eval_parser.add_argument("--model", default="llama-1", help="Model specification to compare against")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple methods")
    compare_parser.add_argument("results_dirs", nargs="+", help="Paths to result directories")
    compare_parser.add_argument("--model", default="llama-1", help="Model specification to compare against")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode for selecting methods")
    interactive_parser.add_argument("--model", default="llama-1", help="Model specification to compare against")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_available_results()
    
    elif args.command == "eval":
        if not Path(args.results_dir).exists():
            print(f"Error: Results directory '{args.results_dir}' not found!")
            sys.exit(1)
        
        success = run_single_evaluation(args.results_dir, args.model)
        if not success:
            sys.exit(1)
    
    elif args.command == "compare":
        for results_dir in args.results_dirs:
            if not Path(results_dir).exists():
                print(f"Error: Results directory '{results_dir}' not found!")
                sys.exit(1)
        
        success = run_comparison(args.results_dirs, args.model)
        if not success:
            sys.exit(1)
    
    elif args.command == "interactive":
        available = list_available_results()
        if not available:
            sys.exit(1)
        
        print("\nEnter the numbers of methods you want to compare (space-separated):")
        print("Example: 1 3 5")
        
        try:
            choices = input("> ").strip().split()
            indices = [int(c) - 1 for c in choices]
            
            if not all(0 <= i < len(available) for i in indices):
                print("Error: Invalid selection!")
                sys.exit(1)
            
            selected = [available[i] for i in indices]
            
            if len(selected) == 1:
                success = run_single_evaluation(selected[0], args.model)
            else:
                success = run_comparison(selected, args.model)
            
            if not success:
                sys.exit(1)
                
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
