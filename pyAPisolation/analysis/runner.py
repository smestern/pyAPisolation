"""
Analysis runner for batch processing

This module provides the main interface for running analyses on
multiple files with progress tracking and error handling.
"""

import os
import glob
from typing import List, Dict, Any, Optional, Callable
#from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
#from dataclasses import asdict#
import pandas as pd

from .base import AnalysisParameters, AnalysisResult
from .registry import AnalysisRegistry


class AnalysisRunner:
    """Main runner for batch analysis operations"""
    
    def __init__(self, registry: Optional[AnalysisRegistry] = None):
        self.registry = registry or AnalysisRegistry()
        self.results: List[AnalysisResult] = []
    
    def run_single_file(self, file_path: str, analyzer_name: str,
                       parameters: AnalysisParameters) -> AnalysisResult:
        """
        Run analysis on a single file
        
        Args:
            file_path: Path to file to analyze
            analyzer_name: Name of analyzer to use
            parameters: Analysis parameters
            
        Returns:
            AnalysisResult
        """
        analyzer = self.registry.get_analyzer(analyzer_name)
        return analyzer.analyze_file(file_path, parameters)
    
    def run_batch(self, file_pattern: str, analyzer_name: str,
                  parameters: AnalysisParameters,
                  protocol_filter: str = "",
                  parallel: bool = True,
                  max_workers: Optional[int] = None,
                  progress_callback: Optional[Callable[[int, int], None]] = None
                  ) -> List[AnalysisResult]:
        """
        Run analysis on multiple files
        
        Args:
            file_pattern: Glob pattern for files to analyze
            analyzer_name: Name of analyzer to use
            parameters: Analysis parameters
            protocol_filter: Filter files by protocol name
            parallel: Whether to use parallel processing
            max_workers: Max number of worker processes (None=auto)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of AnalysisResult objects
        """
        # Find files
        if os.path.isdir(file_pattern):
            files = glob.glob(os.path.join(file_pattern, "**/*.abf"), 
                            recursive=True)
        else:
            files = glob.glob(file_pattern)
        
        if not files:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
        
        # Filter files by protocol if specified
        if protocol_filter:
            files = self._filter_files_by_protocol(files, protocol_filter)
        
        # Set protocol filter in parameters
        if protocol_filter:
            parameters.protocol_filter = protocol_filter
        
        # Run analysis
        if parallel and len(files) > 1:
            results = self._run_parallel(files, analyzer_name, parameters,
                                       max_workers, progress_callback)
        else:
            results = self._run_sequential(files, analyzer_name, parameters,
                                         progress_callback)
        
        self.results.extend(results)
        return results
    
    def save_results(self, output_dir: str, file_prefix: str = "",
                    save_format: str = "csv") -> Dict[str, str]:
        """
        Save analysis results to files
        
        Args:
            output_dir: Directory to save results
            file_prefix: Prefix for output filenames
            save_format: Format to save ('csv' or 'excel')
            
        Returns:
            Dictionary mapping result type to saved file path
        """
        if not self.results:
            raise ValueError("No results to save")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine results by type
        summary_dfs = []
        detailed_dfs = []
        sweep_dfs = []
        
        for result in self.results:
            if result.success and result.summary_data is not None:
                summary_dfs.append(result.summary_data)
            if result.success and result.detailed_data is not None:
                detailed_dfs.append(result.detailed_data)
            if result.success and result.sweep_data is not None:
                sweep_dfs.append(result.sweep_data)
        
        saved_files = {}
        
        # Save summary data
        if summary_dfs:
            summary_df = pd.concat(summary_dfs, ignore_index=True)
            summary_path = os.path.join(output_dir, 
                                      f"{file_prefix}summary.{save_format}")
            self._save_dataframe(summary_df, summary_path, save_format)
            saved_files['summary'] = summary_path
        
        # Save detailed data
        if detailed_dfs:
            detailed_df = pd.concat(detailed_dfs, ignore_index=True)
            detailed_path = os.path.join(output_dir,
                                       f"{file_prefix}detailed.{save_format}")
            self._save_dataframe(detailed_df, detailed_path, save_format)
            saved_files['detailed'] = detailed_path
        
        # Save sweep data
        if sweep_dfs:
            sweep_df = pd.concat(sweep_dfs, ignore_index=True)
            sweep_path = os.path.join(output_dir,
                                    f"{file_prefix}sweeps.{save_format}")
            self._save_dataframe(sweep_df, sweep_path, save_format)
            saved_files['sweeps'] = sweep_path
        
        # Save error report
        error_report = self._generate_error_report()
        if error_report:
            error_path = os.path.join(output_dir, f"{file_prefix}errors.txt")
            with open(error_path, 'w') as f:
                f.write(error_report)
            saved_files['errors'] = error_path
        
        return saved_files
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the analysis run"""
        if not self.results:
            return {}
        
        total_files = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total_files - successful
        
        # Count warnings
        total_warnings = sum(len(r.warnings) for r in self.results)
        
        # Get analyzer types
        analyzer_types = set(r.analyzer_name for r in self.results)
        
        return {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_files if total_files > 0 else 0,
            'total_warnings': total_warnings,
            'analyzer_types': list(analyzer_types)
        }
    
    def _filter_files_by_protocol(self, files: List[str], 
                                protocol_filter: str) -> List[str]:
        """Filter files by protocol name"""
        import pyabf
        
        filtered_files = []
        for file_path in files:
            try:
                abf = pyabf.ABF(file_path, loadData=False)
                if protocol_filter in abf.protocol:
                    filtered_files.append(file_path)
            except Exception:
                # Skip files that can't be opened
                continue
        
        return filtered_files
    
    def _run_sequential(self, files: List[str], analyzer_name: str,
                       parameters: AnalysisParameters,
                       progress_callback: Optional[Callable[[int, int], None]]
                       ) -> List[AnalysisResult]:
        """Run analysis sequentially"""
        results = []
        
        for i, file_path in enumerate(files):
            result = self.run_single_file(file_path, analyzer_name, parameters)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(files))
        
        return results
    
    def _run_parallel(self, files: List[str], analyzer_name: str,
                     parameters: AnalysisParameters,
                     max_workers: Optional[int],
                     progress_callback: Optional[Callable[[int, int], None]]
                     ) -> List[AnalysisResult]:
        """Run analysis in parallel"""
        results = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = [
                executor.submit(self.run_single_file, file_path, 
                              analyzer_name, parameters)
                for file_path in files
            ]
            
            # Collect results as they complete
            for future in futures:
                result = future.result()
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(files))
        
        return results
    
    def _save_dataframe(self, df: pd.DataFrame, file_path: str, 
                       format_type: str) -> None:
        """Save dataframe to file"""
        if format_type.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format_type.lower() in ['excel', 'xlsx']:
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported save format: {format_type}")
    
    def _generate_error_report(self) -> str:
        """Generate a text report of errors and warnings"""
        failed_results = [r for r in self.results if not r.success]
        warning_results = [r for r in self.results if r.warnings]
        
        if not failed_results and not warning_results:
            return ""
        
        report_lines = ["Analysis Error Report", "=" * 50, ""]
        
        if failed_results:
            report_lines.extend([
                f"Failed Files ({len(failed_results)}):",
                "-" * 30
            ])
            
            for result in failed_results:
                report_lines.append(f"File: {result.file_path}")
                for error in result.errors:
                    report_lines.append(f"  Error: {error}")
                report_lines.append("")
        
        if warning_results:
            report_lines.extend([
                f"Files with Warnings ({len(warning_results)}):",
                "-" * 30
            ])
            
            for result in warning_results:
                if result.warnings:  # Only include if there are warnings
                    report_lines.append(f"File: {result.file_path}")
                    for warning in result.warnings:
                        report_lines.append(f"  Warning: {warning}")
                    report_lines.append("")
        
        return "\n".join(report_lines)
