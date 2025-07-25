"""
Modern GUI interface using the new analysis framework

This module provides an updated GUI that uses the modular analysis system
while maintaining compatibility with existing functionality.
"""

import os
import sys
from typing import Dict, Any, Optional, List
import copy

# Import Qt components
try:
    from PySide2.QtWidgets import QApplication, QWidget, QProgressDialog
    from PySide2.QtCore import Qt, QThread, pyqtSignal
    import PySide2.QtCore as QtCore
except ImportError:
    print("PySide2 not available, GUI functionality disabled")
    sys.exit(1)

# Import analysis framework
from ..analysis import registry, AnalysisParameters, AnalysisRunner, AnalysisResult

# Import original GUI for base functionality
from .spikeFinder import analysis_gui


class ModernAnalysisThread(QThread):
    """Thread for running analysis in background"""
    
    progress_updated = pyqtSignal(int, int)  # current, total
    analysis_complete = pyqtSignal(list)     # results
    error_occurred = pyqtSignal(str)         # error message
    
    def __init__(self, file_pattern: str, analyzer_name: str,
                 parameters: AnalysisParameters, parallel: bool = True):
        super().__init__()
        self.file_pattern = file_pattern
        self.analyzer_name = analyzer_name
        self.parameters = parameters
        self.parallel = parallel
        self.runner = AnalysisRunner(registry)
    
    def run(self):
        """Run the analysis in background thread"""
        try:
            results = self.runner.run_batch(
                file_pattern=self.file_pattern,
                analyzer_name=self.analyzer_name,
                parameters=self.parameters,
                parallel=self.parallel,
                progress_callback=self._progress_callback
            )
            self.analysis_complete.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _progress_callback(self, current: int, total: int):
        """Emit progress signal"""
        self.progress_updated.emit(current, total)


class ModernAnalysisGUI(analysis_gui):
    """Enhanced GUI using the new analysis framework"""
    
    def __init__(self):
        super().__init__()
        self.analysis_thread = None
        self.current_results: List[AnalysisResult] = []
        
        # Override some methods to use new framework
        self._setup_new_framework_integration()
    
    def _setup_new_framework_integration(self):
        """Setup integration with new analysis framework"""
        # Add analyzer selection if needed
        self.available_analyzers = registry.list_analyzers()
        
        # You could add UI elements here to select different analyzers
        # For now, we'll use the existing tab system
    
    def run_indiv_analysis_modern(self):
        """Modern version of individual analysis using new framework"""
        if not hasattr(self, 'abf') or self.abf is None:
            return
        
        # Get current analysis type and parameters
        analysis_type = self.get_current_analysis()
        parameters = self._create_analysis_parameters()
        
        try:
            # Get analyzer
            analyzer = registry.get_analyzer(analysis_type)
            
            # Run analysis
            result = analyzer.analyze_file(self.abf.abfFilePath, parameters)
            
            if result.success:
                # Store results in legacy format for compatibility
                if analysis_type == 'spike':
                    self.spike_df = self._convert_spike_results_to_legacy(result)
                    self.subthres_df = None
                elif analysis_type == 'subthreshold':
                    self.subthres_df = result.detailed_data
                    self.spike_df = None
                
                # Store modern results
                self.current_results = [result]
                
            else:
                print(f"Analysis failed: {'; '.join(result.errors)}")
                
        except Exception as e:
            print(f"Error in modern analysis: {e}")
            # Fallback to legacy method
            self.run_indiv_analysis()
    
    def run_batch_analysis_modern(self):
        """Modern batch analysis with progress tracking"""
        if not hasattr(self, 'selected_dir') or not self.selected_dir:
            return
        
        # Get parameters
        analysis_type = self.get_current_analysis()
        parameters = self._create_analysis_parameters()
        
        # Create and start analysis thread
        self.analysis_thread = ModernAnalysisThread(
            file_pattern=self.selected_dir,
            analyzer_name=analysis_type,
            parameters=parameters,
            parallel=self.actionEnable_Parallel.isChecked()
        )
        
        # Connect signals
        self.analysis_thread.progress_updated.connect(self._update_progress)
        self.analysis_thread.analysis_complete.connect(self._on_analysis_complete)
        self.analysis_thread.error_occurred.connect(self._on_analysis_error)
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog(
            "Running modern analysis...", "Cancel", 0, 100, self.main_widget
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Start analysis
        self.analysis_thread.start()
    
    def _create_analysis_parameters(self) -> AnalysisParameters:
        """Create AnalysisParameters from current GUI state"""
        # Get common parameters
        parameters = AnalysisParameters(
            start_time=float(self.start_time.text()) if hasattr(self, 'start_time') else 0.0,
            end_time=float(self.end_time.text()) if hasattr(self, 'end_time') else 0.0,
            protocol_filter=self.get_selected_protocol() if hasattr(self, 'get_selected_protocol') else ""
        )
        
        # Add analysis-specific parameters
        analysis_type = self.get_current_analysis()
        
        if analysis_type == 'spike':
            parameters.extra_params.update({
                'dv_cutoff': float(self.dvdt_thres.text()) if hasattr(self, 'dvdt_thres') else 7.0,
                'max_interval': float(self.thres_to_peak_time.text())/1000 if hasattr(self, 'thres_to_peak_time') else 0.010,
                'min_height': float(self.thres_to_peak_height.text()) if hasattr(self, 'thres_to_peak_height') else 2.0,
                'min_peak': float(self.min_peak_height.text()) if hasattr(self, 'min_peak_height') else -10.0,
                'thresh_frac': float(self.thres_per.text()) if hasattr(self, 'thres_per') else 0.05,
                'bessel_filter': float(self.bessel.text()) if hasattr(self, 'bessel') else 0.0,
                'stim_find': self.bstim.isChecked() if hasattr(self, 'bstim') else False
            })
        
        elif analysis_type == 'subthreshold':
            parameters.extra_params.update({
                'time_after': float(self.stimPer.text()) if hasattr(self, 'stimPer') else 50.0,
                'start_sear': float(self.startCM.text()) if hasattr(self, 'startCM') and self.startCM.text() else None,
                'end_sear': float(self.endCM.text()) if hasattr(self, 'endCM') and self.endCM.text() else None,
            })
            
            # Handle sweep specification
            if hasattr(self, 'subthresSweeps') and self.subthresSweeps.text():
                try:
                    import numpy as np
                    sweeps = np.fromstring(self.subthresSweeps.text(), dtype=int, sep=',')
                    if len(sweeps) > 0:
                        parameters.extra_params['subt_sweeps'] = sweeps.tolist()
                except:
                    pass
        
        return parameters
    
    def _convert_spike_results_to_legacy(self, result: AnalysisResult) -> Dict[int, Any]:
        """Convert modern spike results to legacy format for compatibility"""
        # This is a simplified conversion - you may need to adjust based on 
        # how the legacy GUI expects the data
        legacy_dict = {}
        
        if result.detailed_data is not None and not result.detailed_data.empty:
            # Group by sweep if sweep information is available
            # For now, just use sweep 0
            legacy_dict[0] = result.detailed_data
        
        return legacy_dict
    
    def _update_progress(self, current: int, total: int):
        """Update progress dialog"""
        if hasattr(self, 'progress_dialog'):
            progress_percent = int(100 * current / total) if total > 0 else 0
            self.progress_dialog.setValue(progress_percent)
            self.progress_dialog.setLabelText(f"Processing file {current} of {total}")
    
    def _on_analysis_complete(self, results: List[AnalysisResult]):
        """Handle analysis completion"""
        self.current_results = results
        
        # Hide progress dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.hide()
        
        # Convert results for legacy compatibility
        self._process_modern_results(results)
        
        # Update UI
        if hasattr(self, 'tableView') and results:
            # Combine summary data for display
            summary_dfs = [r.summary_data for r in results if r.success and r.summary_data is not None]
            if summary_dfs:
                import pandas as pd
                combined_df = pd.concat(summary_dfs, ignore_index=True)
                # Update table view (you'll need to implement PandasModel)
                # self.tableView.setModel(PandasModel(combined_df))
        
        print(f"Analysis complete. Processed {len(results)} files.")
        success_count = sum(1 for r in results if r.success)
        print(f"Successful: {success_count}, Failed: {len(results) - success_count}")
    
    def _on_analysis_error(self, error_message: str):
        """Handle analysis error"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.hide()
        
        print(f"Analysis error: {error_message}")
        # You could show an error dialog here
    
    def _process_modern_results(self, results: List[AnalysisResult]):
        """Process modern results for legacy compatibility"""
        # This method can be used to convert modern results back to
        # the format expected by the legacy GUI components
        
        # For now, just store the first successful result
        for result in results:
            if result.success:
                if result.analyzer_name == 'spike':
                    self.df = result.summary_data
                elif result.analyzer_name == 'subthreshold':
                    self.df = result.summary_data
                break
    
    def get_available_analyzers(self) -> List[str]:
        """Get list of available analyzers"""
        return self.available_analyzers
    
    def get_analyzer_info(self, analyzer_name: str) -> Dict[str, str]:
        """Get information about an analyzer"""
        return registry.get_analyzer_info(analyzer_name)
    
    def save_modern_results(self, output_dir: str, file_prefix: str = ""):
        """Save results using the modern framework"""
        if not self.current_results:
            print("No results to save")
            return
        
        # Create runner to use its save functionality
        runner = AnalysisRunner(registry)
        runner.results = self.current_results
        
        try:
            saved_files = runner.save_results(output_dir, file_prefix)
            print("Results saved:")
            for result_type, file_path in saved_files.items():
                print(f"  {result_type}: {file_path}")
        except Exception as e:
            print(f"Error saving results: {e}")


# Function to create the modern GUI
def create_modern_gui():
    """Create and return the modern GUI instance"""
    return ModernAnalysisGUI()
