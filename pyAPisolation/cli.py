"""
Command Line Interface for pyAPisolation Analysis

This module provides a modern CLI interface for the analysis framework
with support for all analyzer types and batch processing.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

from ..analysis import registry, AnalysisParameters, AnalysisRunner


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser"""
    parser = argparse.ArgumentParser(
        description="pyAPisolation Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Spike analysis on a folder
  python -m pyAPisolation.cli spike /path/to/data --protocol "IC_STEPS"
  
  # Subthreshold analysis with custom parameters
  python -m pyAPisolation.cli subthreshold /path/to/data --time-after 75
  
  # Batch analysis with parallel processing
  python -m pyAPisolation.cli spike "/path/**/*.abf" --parallel --output /results
  
  # List available analyzers
  python -m pyAPisolation.cli --list-analyzers
        """
    )
    
    # Global options
    parser.add_argument('--list-analyzers', action='store_true',
                       help='List available analyzers and exit')
    parser.add_argument('--config', type=str,
                       help='JSON config file with analysis parameters')
    parser.add_argument('--output', '-o', type=str, default='.',
                       help='Output directory for results')
    parser.add_argument('--prefix', type=str, default='',
                       help='Prefix for output files')
    parser.add_argument('--format', choices=['csv', 'excel'], default='csv',
                       help='Output format for results')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing for batch analysis')
    parser.add_argument('--workers', type=int,
                       help='Number of worker processes (default: auto)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    # Create subparsers for different analyzer types
    subparsers = parser.add_subparsers(dest='analyzer', 
                                      help='Analysis type')
    
    # Spike analysis subparser
    spike_parser = subparsers.add_parser('spike',
                                        help='Spike detection and analysis')
    _add_spike_arguments(spike_parser)
    
    # Subthreshold analysis subparser  
    subthres_parser = subparsers.add_parser('subthreshold',
                                           help='Subthreshold analysis')
    _add_subthreshold_arguments(subthres_parser)
    
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser"""
    parser.add_argument('input', type=str,
                       help='Input file, directory, or glob pattern')
    parser.add_argument('--protocol', type=str, default='',
                       help='Protocol name filter')
    parser.add_argument('--start-time', type=float, default=0.0,
                       help='Start time for analysis (seconds)')
    parser.add_argument('--end-time', type=float, default=0.0,
                       help='End time for analysis (seconds, 0=auto)')


def _add_spike_arguments(parser: argparse.ArgumentParser) -> None:
    """Add spike-specific arguments"""
    _add_common_arguments(parser)
    
    parser.add_argument('--dv-cutoff', type=float, default=7.0,
                       help='dV/dt threshold for spike detection (mV/s)')
    parser.add_argument('--max-interval', type=float, default=0.010,
                       help='Max time from threshold to peak (seconds)')
    parser.add_argument('--min-height', type=float, default=2.0,
                       help='Min threshold-to-peak height (mV)')
    parser.add_argument('--min-peak', type=float, default=-10.0,
                       help='Min peak voltage (mV)')
    parser.add_argument('--thresh-frac', type=float, default=0.05,
                       help='Fraction of max dV/dt for threshold refinement')
    parser.add_argument('--bessel-filter', type=float, default=0.0,
                       help='Bessel filter frequency (Hz, 0=no filter)')
    parser.add_argument('--stim-find', action='store_true',
                       help='Search based on stimulus timing')


def _add_subthreshold_arguments(parser: argparse.ArgumentParser) -> None:
    """Add subthreshold-specific arguments"""
    _add_common_arguments(parser)
    
    parser.add_argument('--time-after', type=float, default=50.0,
                       help='Percentage of decay to analyze after step')
    parser.add_argument('--sweeps', type=str,
                       help='Comma-separated sweep numbers to analyze')
    parser.add_argument('--start-search', type=float,
                       help='Start time for analysis window (s)')
    parser.add_argument('--end-search', type=float,
                       help='End time for analysis window (s)')
    parser.add_argument('--savfilter', type=int, default=0,
                       help='Savitzky-Golay filter window (0=no filter)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate diagnostic plots')


def parse_config_file(config_path: str) -> Dict[str, Any]:
    """Parse JSON configuration file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_parameters_from_args(args: argparse.Namespace) -> AnalysisParameters:
    """Create AnalysisParameters from command line arguments"""
    parameters = AnalysisParameters(
        start_time=args.start_time,
        end_time=args.end_time,
        protocol_filter=getattr(args, 'protocol', '')
    )
    
    # Add analyzer-specific parameters
    if args.analyzer == 'spike':
        parameters.set('dv_cutoff', args.dv_cutoff)
        parameters.set('max_interval', args.max_interval)
        parameters.set('min_height', args.min_height)
        parameters.set('min_peak', args.min_peak)
        parameters.set('thresh_frac', args.thresh_frac)
        parameters.set('bessel_filter', args.bessel_filter)
        parameters.set('stim_find', args.stim_find)
    elif args.analyzer == 'subthreshold':
        parameters.set('time_after', args.time_after)
        parameters.set('savfilter', args.savfilter)
        parameters.set('bplot', getattr(args, 'plot', False))
        
        # Handle sweep specification
        if hasattr(args, 'sweeps') and args.sweeps:
            try:
                sweeps = [int(x.strip()) for x in args.sweeps.split(',')]
                parameters.set('subt_sweeps', sweeps)
            except ValueError:
                print("Warning: Invalid sweep specification, ignoring")
        
        # Handle search window
        if hasattr(args, 'start_search') and args.start_search is not None:
            parameters.set('start_sear', args.start_search)
        if hasattr(args, 'end_search') and args.end_search is not None:
            parameters.set('end_sear', args.end_search)
    
    return parameters


def print_progress(current: int, total: int) -> None:
    """Print progress bar"""
    percent = int(100 * current / total)
    bar_length = 50
    filled_length = int(bar_length * current / total)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: [{bar}] {percent}% ({current}/{total})', 
          end='', flush=True)
    if current == total:
        print()  # New line when complete


def main() -> int:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle list analyzers
    if args.list_analyzers:
        print("Available analyzers:")
        for name in registry.list_analyzers():
            info = registry.get_analyzer_info(name)
            print(f"  {name}: {info['type']} analysis")
        return 0
    
    # Validate analyzer selection
    if not args.analyzer:
        parser.error("Must specify an analyzer type or use --list-analyzers")
    
    if args.analyzer not in registry.list_analyzers():
        parser.error(f"Unknown analyzer: {args.analyzer}")
    
    # Load config file if specified
    if args.config:
        try:
            config = parse_config_file(args.config)
            # Override args with config values
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return 1
    
    # Create parameters
    try:
        parameters = create_parameters_from_args(args)
    except Exception as e:
        print(f"Error creating parameters: {e}")
        return 1
    
    # Create runner and run analysis
    runner = AnalysisRunner(registry)
    
    try:
        print(f"Running {args.analyzer} analysis on: {args.input}")
        
        # Setup progress callback if verbose
        progress_callback = print_progress if args.verbose else None
        
        results = runner.run_batch(
            file_pattern=args.input,
            analyzer_name=args.analyzer,
            parameters=parameters,
            parallel=args.parallel,
            max_workers=args.workers,
            progress_callback=progress_callback
        )
        
        # Print summary
        stats = runner.get_summary_stats()
        print(f"\nAnalysis complete:")
        print(f"  Files processed: {stats['total_files']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        
        if stats['total_warnings'] > 0:
            print(f"  Warnings: {stats['total_warnings']}")
        
        # Save results
        if results:
            saved_files = runner.save_results(
                output_dir=args.output,
                file_prefix=args.prefix,
                save_format=args.format
            )
            
            print(f"\nResults saved to:")
            for result_type, file_path in saved_files.items():
                print(f"  {result_type}: {file_path}")
        
        return 0 if stats['failed'] == 0 else 1
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
