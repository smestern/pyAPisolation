# -*- coding: utf-8 -*-

"""
Post-hoc Analysis Runner Wizard
A PySide2-based wizard for running statistical analysis on electrophysiological data.
Performs multiple one-way ANOVAs across features with user-selected categorical variables.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns


class PostAnalysisWizard(QWizard):
    """Main wizard class for post-hoc analysis"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Post-hoc Analysis Runner")
        self.setWindowIcon(QIcon())
        self.resize(800, 600)
        
        # Data storage
        self.data = None
        self.categorical_column = None
        self.selected_features = []
        self.results = {}
        
        # Add wizard pages
        self.addPage(FileSelectionPage())
        self.addPage(CategorySelectionPage())
        self.addPage(FeatureSelectionPage())
        self.addPage(AnalysisPage())
        self.addPage(ResultsPage())
        
        # Connect signals
        self.currentIdChanged.connect(self.on_page_changed)
        
    def on_page_changed(self, page_id):
        """Handle page changes to update data between pages"""
        if page_id == 1:  # Category selection page
            category_page = self.page(1)
            if hasattr(category_page, 'update_columns') and self.data is not None:
                category_page.update_columns(self.data.columns.tolist())
        elif page_id == 2:  # Feature selection page
            feature_page = self.page(2)
            if hasattr(feature_page, 'update_features') and self.data is not None:
                # Exclude the categorical column from features
                available_features = [col for col in self.data.columns 
                                    if col != self.categorical_column and 
                                    pd.api.types.is_numeric_dtype(self.data[col])]
                feature_page.update_features(available_features)


class FileSelectionPage(QWizardPage):
    """First page: File selection"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Select Data File")
        self.setSubTitle("Choose a CSV or Excel file containing your data for analysis.")
        
        layout = QVBoxLayout()
        
        # File selection group
        file_group = QGroupBox("Data File")
        file_layout = QVBoxLayout()
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a CSV or Excel file...")
        self.file_path_edit.textChanged.connect(self.validate_file)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_file)
        
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_path_edit)
        file_row.addWidget(browse_button)
        
        file_layout.addLayout(file_row)
        file_group.setLayout(file_layout)
        
        # Preview group
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_table)
        
        self.info_label = QLabel("No file selected")
        preview_layout.addWidget(self.info_label)
        
        preview_group.setLayout(preview_layout)
        
        layout.addWidget(file_group)
        layout.addWidget(preview_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Register field for validation
        self.registerField("file_path*", self.file_path_edit)
        
    def browse_file(self):
        """Open file dialog to select data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Data File", 
            "", 
            "Data files (*.csv *.xlsx *.xls);;CSV files (*.csv);;Excel files (*.xlsx *.xls)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)
            
    def validate_file(self):
        """Validate and load the selected file"""
        file_path = self.file_path_edit.text()
        if not file_path or not os.path.exists(file_path):
            self.preview_table.clear()
            self.info_label.setText("No file selected")
            return False
            
        try:
            # Load data based on file extension
            if file_path.lower().endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
                
            # Store data in wizard
            self.wizard().data = data
            
            # Update preview
            self.update_preview(data)
            self.info_label.setText(f"Loaded: {len(data)} rows, {len(data.columns)} columns")
            
            return True
            
        except Exception as e:
            self.preview_table.clear()
            self.info_label.setText(f"Error loading file: {str(e)}")
            return False
            
    def update_preview(self, data):
        """Update the preview table with data"""
        # Show first 5 rows and up to 10 columns
        preview_data = data.head(5)
        display_cols = min(10, len(preview_data.columns))
        
        self.preview_table.setRowCount(len(preview_data))
        self.preview_table.setColumnCount(display_cols)
        self.preview_table.setHorizontalHeaderLabels(
            [str(col) for col in preview_data.columns[:display_cols]]
        )
        
        for i, row in preview_data.iterrows():
            for j, value in enumerate(row[:display_cols]):
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(i, j, item)
                
        self.preview_table.resizeColumnsToContents()


class CategorySelectionPage(QWizardPage):
    """Second page: Select categorical column for grouping"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Select Grouping Variable")
        self.setSubTitle("Choose the column that contains the categories for your ANOVA analysis.")
        
        layout = QVBoxLayout()
        
        # Category selection group
        category_group = QGroupBox("Categorical Variable")
        category_layout = QVBoxLayout()
        
        self.category_combo = QComboBox()
        self.category_combo.currentTextChanged.connect(self.on_category_changed)
        
        category_layout.addWidget(QLabel("Select the column to use for grouping:"))
        category_layout.addWidget(self.category_combo)
        
        category_group.setLayout(category_layout)
        
        # Group preview
        preview_group = QGroupBox("Group Preview")
        preview_layout = QVBoxLayout()
        
        self.group_table = QTableWidget()
        self.group_table.setMaximumHeight(200)
        preview_layout.addWidget(self.group_table)
        
        self.group_info_label = QLabel("Select a categorical variable to see group information")
        preview_layout.addWidget(self.group_info_label)
        
        preview_group.setLayout(preview_layout)
        
        layout.addWidget(category_group)
        layout.addWidget(preview_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Register field
        self.registerField("categorical_column*", self.category_combo, "currentText")
        
    def update_columns(self, columns):
        """Update available columns in the combo box"""
        self.category_combo.clear()
        self.category_combo.addItems(columns)
        
    def on_category_changed(self):
        """Handle category selection change"""
        category_col = self.category_combo.currentText()
        if not category_col or self.wizard().data is None:
            return
            
        self.wizard().categorical_column = category_col
        self.update_group_preview(category_col)
        
        # Emit signal to update wizard page completion status
        self.completeChanged.emit()
        
    def update_group_preview(self, category_col):
        """Update the group preview table"""
        try:
            data = self.wizard().data
            if data is None or category_col not in data.columns:
                return
                
            # Get group counts
            group_counts = data[category_col].value_counts().sort_index()
            
            self.group_table.setRowCount(len(group_counts))
            self.group_table.setColumnCount(2)
            self.group_table.setHorizontalHeaderLabels(["Group", "Count"])
            
            for i, (group, count) in enumerate(group_counts.items()):
                self.group_table.setItem(i, 0, QTableWidgetItem(str(group)))
                self.group_table.setItem(i, 1, QTableWidgetItem(str(count)))
                
            self.group_table.resizeColumnsToContents()
            
            total_groups = len(group_counts)
            total_samples = group_counts.sum()
            self.group_info_label.setText(
                f"Found {total_groups} groups with {total_samples} total samples"
            )
            return True
            
        except Exception as e:
            self.group_info_label.setText(f"Error analyzing groups: {str(e)}")
    
    def isComplete(self):
        """Check if a categorical column is selected"""
        # Check both the combo box and the wizard's stored value
        current_text = self.category_combo.currentText()
        return bool(current_text and current_text.strip())


class FeatureSelectionPage(QWizardPage):
    """Third page: Select features for analysis"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Select Features for Analysis")
        self.setSubTitle("Choose which numeric columns to include in the ANOVA analysis.")
        
        layout = QVBoxLayout()
        
        # Feature selection group
        feature_group = QGroupBox("Available Features")
        feature_layout = QVBoxLayout()
        
        # Selection buttons
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_none_btn = QPushButton("Select None")
        self.select_all_btn.clicked.connect(self.select_all_features)
        self.select_none_btn.clicked.connect(self.select_no_features)
        
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.select_none_btn)
        button_layout.addStretch()
        
        # Feature list
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.feature_list.itemSelectionChanged.connect(self.on_selection_changed)
        
        feature_layout.addLayout(button_layout)
        feature_layout.addWidget(self.feature_list)
        
        feature_group.setLayout(feature_layout)
        
        # Selection info
        self.selection_label = QLabel("No features selected")
        
        layout.addWidget(feature_group)
        layout.addWidget(self.selection_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def update_features(self, features):
        """Update available features in the list"""
        self.feature_list.clear()
        for feature in features:
            item = QListWidgetItem(feature)
            self.feature_list.addItem(item)
            
    def select_all_features(self):
        """Select all features"""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setSelected(True)
            
    def select_no_features(self):
        """Deselect all features"""
        self.feature_list.clearSelection()
        
    def on_selection_changed(self):
        """Handle feature selection changes"""
        selected_items = self.feature_list.selectedItems()
        selected_features = [item.text() for item in selected_items]
        self.wizard().selected_features = selected_features
        
        count = len(selected_features)
        self.selection_label.setText(f"{count} feature(s) selected for analysis")
        # Emit signal to update wizard page completion status
        self.completeChanged.emit()
        
    def isComplete(self):
        """Check if page is complete"""
        return len(self.wizard().selected_features) > 0


class AnalysisPage(QWizardPage):
    """Fourth page: Run the analysis"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Running Analysis")
        self.setSubTitle("Performing one-way ANOVA for each selected feature...")
        
        layout = QVBoxLayout()
        
        # Progress group
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready to start analysis")
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        
        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        
        # Results preview
        results_group = QGroupBox("Analysis Summary")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        
        layout.addWidget(progress_group)
        layout.addWidget(self.run_button)
        layout.addWidget(results_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
        self.analysis_complete = False
        
    def run_analysis(self):
        """Run the ANOVA analysis"""
        try:
            self.run_button.setEnabled(False)
            self.progress_label.setText("Starting analysis...")
            self.progress_bar.setValue(0)
            
            wizard = self.wizard()
            data = wizard.data
            categorical_col = wizard.categorical_column
            features = wizard.selected_features
            
            if not all([data is not None, categorical_col, features]):
                raise ValueError("Missing required data for analysis")
                
            # Clean data - remove rows with NaN in categorical column
            clean_data = data.dropna(subset=[categorical_col])
            groups = clean_data[categorical_col].unique()
            
            results = {}
            total_features = len(features)
            
            for i, feature in enumerate(features):
                self.progress_label.setText(f"Analyzing {feature}...")
                self.progress_bar.setValue(int((i / total_features) * 100))
                QApplication.processEvents()
                
                # Skip if feature has too many NaN values
                feature_data = clean_data[[categorical_col, feature]].dropna()
                if len(feature_data) < 3:  # Need at least 3 data points
                    results[feature] = {
                        'f_statistic': np.nan,
                        'p_value': np.nan,
                        'error': 'Insufficient data points'
                    }
                    continue
                
                # Prepare data for ANOVA
                group_data = []
                for group in groups:
                    group_values = feature_data[feature_data[categorical_col] == group][feature]
                    if len(group_values) > 0:
                        group_data.append(group_values)
                
                # Run one-way ANOVA
                if len(group_data) >= 2 and all(len(g) > 0 for g in group_data):
                    f_stat, p_value = f_oneway(*group_data)
                    
                    # Calculate effect size (eta squared)
                    ss_between = sum(len(g) * (np.mean(g) - np.mean(feature_data[feature]))**2 
                                   for g in group_data)
                    ss_total = np.sum((feature_data[feature] - np.mean(feature_data[feature]))**2)
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    results[feature] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'eta_squared': eta_squared,
                        'groups': len(group_data),
                        'total_n': len(feature_data),
                        'group_means': {str(groups[i]): np.mean(group_data[i]) 
                                      for i in range(len(group_data))},
                        'group_stds': {str(groups[i]): np.std(group_data[i]) 
                                     for i in range(len(group_data))},
                        'error': None
                    }
                else:
                    results[feature] = {
                        'f_statistic': np.nan,
                        'p_value': np.nan,
                        'error': 'Insufficient groups or data'
                    }
            
            self.progress_bar.setValue(100)
            self.progress_label.setText("Analysis complete!")
            
            # Store results
            wizard.results = results
            
            # Update results preview
            self.update_results_preview(results)
            
            self.analysis_complete = True
            self.completeChanged.emit()
            
        except Exception as e:
            self.progress_label.setText(f"Error: {str(e)}")
            self.results_text.setText(f"Analysis failed: {str(e)}")
        finally:
            self.run_button.setEnabled(True)
            
    def update_results_preview(self, results):
        """Update the results preview text"""
        text = "ANOVA Results Summary:\n\n"
        
        significant_count = 0
        total_count = 0
        
        for feature, result in results.items():
            if result.get('error'):
                text += f"{feature}: {result['error']}\n"
            else:
                p_val = result.get('p_value', np.nan)
                f_stat = result.get('f_statistic', np.nan)
                
                if not np.isnan(p_val):
                    total_count += 1
                    if p_val < 0.05:
                        significant_count += 1
                        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                    else:
                        significance = "ns"
                    
                    text += f"{feature}: F={f_stat:.3f}, p={p_val:.4f} {significance}\n"
                else:
                    text += f"{feature}: Analysis failed\n"
        
        text += f"\nSummary: {significant_count}/{total_count} features showed significant differences (p < 0.05)"
        
        self.results_text.setText(text)
        
    def isComplete(self):
        """Check if analysis is complete"""
        return self.analysis_complete


class ResultsPage(QWizardPage):
    """Final page: Display detailed results"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Analysis Results")
        self.setSubTitle("Detailed results and export options")
        
        layout = QVBoxLayout()
        
        # Results table
        results_group = QGroupBox("Detailed Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        
        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QHBoxLayout()
        
        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_plots_btn = QPushButton("Generate Plots")
        
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_plots_btn.clicked.connect(self.generate_plots)
        
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_plots_btn)
        export_layout.addStretch()
        
        export_group.setLayout(export_layout)
        
        layout.addWidget(results_group)
        layout.addWidget(export_group)
        
        self.setLayout(layout)
        
    def initializePage(self):
        """Initialize the page with results"""
        self.update_results_table()
        
    def update_results_table(self):
        """Update the results table with detailed statistics"""
        results = self.wizard().results
        if not results:
            return
            
        # Prepare table headers
        headers = ['Feature', 'F-statistic', 'p-value', 'Eta²', 'Groups', 'N', 'Significance']
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        
        # Add results to table
        valid_results = [(k, v) for k, v in results.items() if not v.get('error')]
        self.results_table.setRowCount(len(valid_results))
        
        for i, (feature, result) in enumerate(valid_results):
            self.results_table.setItem(i, 0, QTableWidgetItem(feature))
            
            f_stat = result.get('f_statistic', np.nan)
            p_val = result.get('p_value', np.nan)
            eta_sq = result.get('eta_squared', np.nan)
            
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{f_stat:.4f}" if not np.isnan(f_stat) else "N/A"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{eta_sq:.4f}" if not np.isnan(eta_sq) else "N/A"))
            self.results_table.setItem(i, 4, QTableWidgetItem(str(result.get('groups', 'N/A'))))
            self.results_table.setItem(i, 5, QTableWidgetItem(str(result.get('total_n', 'N/A'))))
            
            # Significance
            if not np.isnan(p_val):
                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
            else:
                sig = "N/A"
            self.results_table.setItem(i, 6, QTableWidgetItem(sig))
            
        self.results_table.resizeColumnsToContents()
        
    def export_csv(self):
        """Export results to CSV file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "anova_results.csv", "CSV files (*.csv)"
        )
        
        if file_path:
            try:
                results = self.wizard().results
                
                # Create DataFrame from results
                export_data = []
                for feature, result in results.items():
                    if result.get('error'):
                        export_data.append({
                            'Feature': feature,
                            'F_statistic': 'Error',
                            'p_value': 'Error',
                            'eta_squared': 'Error',
                            'groups': 'Error',
                            'total_n': 'Error',
                            'error': result['error']
                        })
                    else:
                        export_data.append({
                            'Feature': feature,
                            'F_statistic': result.get('f_statistic', np.nan),
                            'p_value': result.get('p_value', np.nan),
                            'eta_squared': result.get('eta_squared', np.nan),
                            'groups': result.get('groups', np.nan),
                            'total_n': result.get('total_n', np.nan),
                            'error': None
                        })
                
                df = pd.DataFrame(export_data)
                df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Export Complete", f"Results exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
                
    def generate_plots(self):
        """Generate visualization plots"""
        try:
            wizard = self.wizard()
            data = wizard.data
            categorical_col = wizard.categorical_column
            results = wizard.results
            
            # Create plots for significant results
            significant_features = [
                feature for feature, result in results.items()
                if not result.get('error') and result.get('p_value', 1) < 0.05
            ]
            
            if not significant_features:
                QMessageBox.information(self, "No Plots", "No significant results to plot.")
                return
                
            # Ask user for save location
            save_dir = QFileDialog.getExistingDirectory(self, "Select Directory for Plots")
            if not save_dir:
                return
                
            # Generate plots
            for feature in significant_features[:10]:  # Limit to first 10 significant features
                plt.figure(figsize=(8, 6))
                
                clean_data = data[[categorical_col, feature]].dropna()
                
                # Box plot
                plt.subplot(1, 2, 1)
                groups = clean_data[categorical_col].unique()
                group_data = [clean_data[clean_data[categorical_col] == group][feature] 
                            for group in groups]
                plt.boxplot(group_data, labels=groups)
                plt.title(f'{feature} - Box Plot')
                plt.xlabel(categorical_col)
                plt.ylabel(feature)
                plt.xticks(rotation=45)
                
                # Bar plot with error bars
                plt.subplot(1, 2, 2)
                means = [np.mean(group) for group in group_data]
                stds = [np.std(group) for group in group_data]
                plt.bar(range(len(groups)), means, yerr=stds, capsize=5)
                plt.title(f'{feature} - Means ± SD')
                plt.xlabel(categorical_col)
                plt.ylabel(feature)
                plt.xticks(range(len(groups)), groups, rotation=45)
                
                plt.tight_layout()
                
                # Add p-value to title
                p_val = results[feature]['p_value']
                plt.suptitle(f'{feature} (p = {p_val:.4f})', y=1.02)
                
                # Save plot
                filename = f"{feature.replace(' ', '_').replace('/', '_')}_plot.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
            QMessageBox.information(
                self, "Plots Generated", 
                f"Generated plots for {len(significant_features)} significant features in {save_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Failed to generate plots: {str(e)}")


def main():
    """Main function to run the wizard"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Post-hoc Analysis Runner")
    app.setApplicationVersion("1.0")
    
    # Create and show wizard
    wizard = PostAnalysisWizard()
    wizard.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
