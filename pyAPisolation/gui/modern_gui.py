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
    from PySide2.QtWidgets import QApplication, QWidget, QProgressDialog, QComboBox, QVBoxLayout
    from PySide2.QtCore import Qt, QThread
    from PySide2.QtCore import Signal as pyqtSignal
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
    
    def __init__(self, app):
        super().__init__(app)
        self.analysis_thread = None
        self.current_results: List[AnalysisResult] = []
        
        # Override some methods to use new framework
        self._setup_new_framework_integration()
    
    def _setup_new_framework_integration(self):
        """Setup integration with new analysis framework"""
        # Initialize modern analysis preferences
        self.use_modern_analysis = True
        
        # Add analyzer selection if needed
        self.available_analyzers = registry.list_modules()
        
        # Setup modern analysis integration with existing UI
        self.setup_modern_analysis_integration()
        
        # You could add UI elements here to select different analyzers
        # For now, we'll use the existing tab system
        for analyzer_name in self.available_analyzers:
            if hasattr(self, f"run_{analyzer_name}_analysis"):
                # Add a button or menu item to run this analysis
                # For simplicity, we'll just print available analyzers
                print(f"Available analyzer: {analyzer_name}")
        #add a tab
        # Add a tab for modern analysis if it doesn't exist
        if not hasattr(self, 'modern_analysis_tab'):
            self.modern_analysis_tab = QWidget()
            self.modern_analysis_tab.setObjectName("modern_analysis_tab")
            self.modern_analysis_tab.setWindowTitle("Analysis")
            self.tabselect.addTab(self.modern_analysis_tab, "Analysis")
            #add a dropdown to select analysis type
            self.analysis_tab_layout = QVBoxLayout(self.modern_analysis_tab)
            #top justification
            self.analysis_tab_layout.setAlignment(Qt.AlignTop)
            self.analysis_type_dropdown = QComboBox(self.modern_analysis_tab)
            self.analysis_type_dropdown.addItems(self.available_analyzers)
            self.analysis_type_dropdown.setCurrentIndex(-1)  # No selection by default
            self.analysis_type_dropdown.currentIndexChanged.connect(self._on_analysis_type_changed)
            self.analysis_tab_layout.addWidget(self.analysis_type_dropdown)
            self.modern_analysis_tab.setLayout(self.analysis_tab_layout)


    def _on_analysis_type_changed(self, index: int):
        """Handle analysis type selection change"""
        if index < 0 or index >= len(self.available_analyzers):
            return
        
        selected_analyzer = self.available_analyzers[index]
        print(f"Selected analyzer: {selected_analyzer}")
        
        # Update current analysis type
        self.current_analysis = selected_analyzer
        
        # Update UI elements based on selected analyzer
        self.update_ui_for_analyzer(selected_analyzer)


    def update_ui_for_analyzer(self, analyzer_name: str):
        """Update UI elements based on selected analyzer"""
        # This method can be used to show/hide specific UI elements
        # or update parameters based on the selected analyzer
        
        # For now, just print the analyzer name
        print(f"Updating UI for analyzer: {analyzer_name}")
        
        # You could add logic here to enable/disable specific fields
        # or change labels based on the selected analyzer type
        
        # Example: if the analyzer has specific parameters, show them
        if hasattr(self, 'analysis_parameters'):
            self.analysis_parameters.setText(f"Parameters for {analyzer_name}")

        #clear the tab if it exists
        if hasattr(self, 'modern_analysis_tab'):
            # Clear existing content in the tab
            if hasattr(self, 'parameter_widgets'):
                for i, item in self.parameter_widgets.items():
                    item.deleteLater()
        self.parameter_widgets = {}

        #now add the parameters for the selected analyzer
        if analyzer_name in self.available_analyzers:
            analyzer = registry.get_module(analyzer_name)
            parameters = analyzer.get_ui_elements()
            self._display_parameters(parameters)
    
    def _display_parameters(self, parameters):
        """Display parameter controls based on the analysis schema"""
        from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox
        
        if not hasattr(self, 'modern_analysis_tab'):
            return
        
        # Create layout if it doesn't exist
        if self.analysis_tab_layout is None:
            self.analysis_tab_layout = QVBoxLayout(self.modern_analysis_tab)
            self.analysis_tab_layout.addWidget(self.analysis_type_dropdown)
        else:
            layout = self.analysis_tab_layout

        # Store parameter widgets for later retrieval
        if not hasattr(self, 'parameter_widgets'):
            self.parameter_widgets = {}
        
        # Create widgets for each parameter
        for param_name, param_info in parameters.items():
            #if param_name in ['start_time', 'end_time', 'protocol_filter']:
                #continue  # Skip common parameters handled elsewhere
            
            param_layout = QHBoxLayout()
            label = QLabel(f"{param_name}:")
            param_layout.addWidget(label)
            
            widget = self._create_parameter_widget(param_info, param_info.value)
            param_layout.addWidget(widget)
            
            layout.addLayout(param_layout)
            self.parameter_widgets[param_name] = widget
            self.parameter_widgets[f"{param_name}_label"] = label
    
    def _create_parameter_widget(self, param_info, current_value):
        """Create appropriate widget based on parameter type"""
        from PySide2.QtWidgets import QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox
        
        param_type_str = str(param_info.param_type)
        param_type_raw = param_info.param_type
        default_value = param_info.get('default', current_value)

        if 'bool' in param_type_str or isinstance(current_value, bool):
            widget = QCheckBox()
            widget.setChecked(bool(current_value if current_value is not None else default_value))
            return widget

        elif 'int' in param_type_str or isinstance(current_value, int):
            widget = QSpinBox()
            # if 'min' in param_info:
            #     widget.setMinimum(int(param_info['min']))
            # if 'max' in param_info:
            #     widget.setMaximum(int(param_info['max']))
            widget.setValue(int(current_value if current_value is not None else default_value))
            return widget

        elif 'float' in param_type_str or isinstance(current_value, float):
            widget = QDoubleSpinBox()
            widget.setDecimals(4)
            # if 'min' in param_info:
            #     widget.setMinimum(float(param_info['min']))
            # if 'max' in param_info:
            #     widget.setMaximum(float(param_info['max']))
            widget.setValue(float(current_value if current_value is not None else default_value))
            return widget
        
        else:  # Default to string/text input
            widget = QLineEdit()
            widget.setText(str(current_value if current_value is not None else default_value))
            return widget

    def setup_modern_analysis_integration(self):
        """Setup modern analysis integration with existing tab system"""
        # This method integrates the modern analysis framework with the existing GUI
        # without needing to construct new tabs since they already exist in the UI
        
        # Override the existing run analysis methods to use modern framework when appropriate
        # Store original methods for fallback
        if not hasattr(self, '_original_run_indiv_analysis'):
            self._original_run_indiv_analysis = super().run_indiv_analysis
            self._original_run_analysis = super().run_analysis

        # Add modern analysis menu options or buttons if they don't exist
        self._add_modern_analysis_ui_elements()
    
    def _add_modern_analysis_ui_elements(self):
        """Add UI elements for modern analysis if they don't already exist"""
        # Add a checkbox or menu item to enable modern analysis mode
        if hasattr(self, 'tools_menu') and self.tools_menu:
            # Add separator if it doesn't exist
            actions = [action.text() for action in self.tools_menu.actions()]
            if "Use Modern Analysis Framework" not in actions:
                self.tools_menu.addSeparator()
                self.action_use_modern = self.tools_menu.addAction("Use Modern Analysis Framework")
                self.action_use_modern.setCheckable(True)
                self.action_use_modern.setChecked(True)  # Default to modern framework
                self.action_use_modern.triggered.connect(self._on_modern_analysis_toggled)
    
    def _on_modern_analysis_toggled(self, checked: bool):
        """Handle toggling between modern and legacy analysis"""
        self.use_modern_analysis = checked
        print(f"Modern analysis framework: {'Enabled' if checked else 'Disabled'}")
    
    def get_current_analysis(self) -> str:
        """Get the currently selected analysis type"""
        if hasattr(self, 'analysis_type_dropdown') and self.analysis_type_dropdown.currentIndex() >= 0:
            return self.analysis_type_dropdown.currentText()
        return ""


    def get_current_analyzer_config(self) -> Dict[str, Any]:
        """Get configuration for the current analyzer based on selected tab"""
        current_analysis = self.get_current_analysis()
        analyzer_info = {}
        
        if current_analysis in self.available_analyzers:
            analyzer_info = self.get_analyzer_info(current_analysis)
        
        return {
            'name': current_analysis,
            'info': analyzer_info,
            'available': current_analysis in self.available_analyzers
        }
    
    def run_indiv_analysis(self):
        """Override base class method to use modern framework when enabled"""
        if hasattr(self, 'use_modern_analysis') and self.use_modern_analysis:
            return self.run_indiv_analysis_modern()
        else:
            # Fallback to original method
            return self._original_run_indiv_analysis()
    
    def run_analysis(self):
        """Override base class method to use modern framework when enabled"""
        if hasattr(self, 'use_modern_analysis') and self.use_modern_analysis:
            return self.run_batch_analysis_modern()
        else:
            # Fallback to original method
            return self._original_run_analysis()
    
    def run_indiv_analysis_modern(self):
        """Modern version of individual analysis using new framework"""
        if not hasattr(self, 'abf') or self.abf is None:
            return
        
        # Get current analysis type and parameters
        analysis_type = self.get_current_analysis()
        parameters = self._create_analysis_parameters()
        
        #try:
        # Get analyzer
        analyzer = registry.get_analyzer(analysis_type)
        
        # Run analysis
        result = analyzer.analyze(file=self.abf.abfFilePath, parameters=parameters)

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
                
        #except Exception as e:
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
        analyzer = registry.get_analyzer(analysis_type)
        _internal = analyzer.parameters

        #legacy parameters
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
        
        else:
            # Get the widgets
            # Update the parameters with the internals
            parameters.extra_params.update(_internal.extra_params)
            for p, a in self.parameter_widgets.items():
                if "_label" in p:
                    continue  # Skip labels
                if p in _internal or p in parameters:
                    parameters[p] = a.value()
                if p in parameters.extra_params or p in _internal.extra_params:
                    parameters.extra_params[p] = a.value()

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
