# Post-hoc Analysis Runner Wizard

A PySide2-based GUI wizard for performing statistical analysis on electrophysiological data. The wizard performs multiple one-way ANOVAs across features with user-selected categorical variables.

## Features

- **File Import**: Supports CSV and Excel (.xlsx, .xls) files
- **Data Preview**: Shows a preview of your data before analysis
- **Categorical Variable Selection**: Choose which column to use for grouping
- **Feature Selection**: Select which numeric columns to include in analysis
- **One-way ANOVA**: Performs ANOVA for each selected feature
- **Statistical Results**: Shows F-statistics, p-values, and effect sizes (eta²)
- **Export Options**: Export results to CSV and generate plots
- **Visualization**: Automatic generation of box plots and bar plots for significant results

## Installation

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

The wizard requires:
- PySide2 (Qt GUI framework)
- pandas (data manipulation)
- numpy (numerical operations)
- scipy (statistical functions)
- matplotlib (plotting)
- seaborn (statistical visualization)
- openpyxl (Excel file support)

## Usage

### Running the Wizard

You can run the wizard in several ways:

1. **Standalone launcher**:
   ```bash
   python run_analysis_wizard.py
   ```

2. **Direct import**:
   ```python
   from pyAPisolation.gui.postAnalysisRunner import PostAnalysisWizard
   from PySide2.QtWidgets import QApplication
   import sys
   
   app = QApplication(sys.argv)
   wizard = PostAnalysisWizard()
   wizard.show()
   app.exec_()
   ```

### Step-by-Step Guide

1. **File Selection**
   - Click "Browse..." to select your CSV or Excel file
   - Preview the first 5 rows and 10 columns of your data
   - Verify the file loaded correctly

2. **Category Selection**
   - Choose the column that contains your experimental groups/conditions
   - View the group counts and distribution
   - This column will be used as the independent variable for ANOVA

3. **Feature Selection**
   - Select which numeric columns to include in the analysis
   - Use "Select All" or "Select None" for quick selection
   - Only numeric columns are available for analysis

4. **Analysis**
   - Click "Run Analysis" to start the statistical analysis
   - Progress bar shows analysis progress
   - Preview shows a summary of significant results

5. **Results**
   - View detailed results table with F-statistics, p-values, and effect sizes
   - Export results to CSV for further analysis
   - Generate plots for significant results (box plots and bar plots with error bars)

## Understanding the Results

### Statistical Output

For each feature, the wizard provides:

- **F-statistic**: The F-ratio from the ANOVA test
- **p-value**: Statistical significance (< 0.05 indicates significant differences)
- **Eta² (eta squared)**: Effect size measure (0.01 = small, 0.06 = medium, 0.14 = large effect)
- **Groups**: Number of groups compared
- **N**: Total number of valid data points
- **Significance**: 
  - `***` p < 0.001
  - `**` p < 0.01  
  - `*` p < 0.05
  - `ns` not significant

### Interpretation

- **Significant results (p < 0.05)**: There are statistically significant differences between groups for this feature
- **Non-significant results (p ≥ 0.05)**: No evidence of differences between groups
- **Effect size (eta²)**: Indicates the practical significance of the differences

## Data Requirements

### File Format
- CSV files (.csv) or Excel files (.xlsx, .xls)
- First row should contain column headers
- Data should be in "tidy" format (one row per observation)

### Data Structure
- At least one categorical column for grouping (e.g., treatment, condition, genotype)
- Multiple numeric columns for analysis (e.g., spike count, amplitude, latency)
- Missing values (NaN) are automatically handled

### Example Data Structure
```
filename,group,spike_count,amplitude,threshold,latency
cell_001,Control,15,45.2,-40.1,0.25
cell_002,Treatment,8,42.1,-38.5,0.31
cell_003,Control,12,47.8,-41.2,0.23
...
```

## Testing

To test the wizard with sample data:

1. Generate test data:
   ```bash
   python create_test_data.py
   ```

2. This creates `sample_ephys_data.csv` with simulated electrophysiological data
3. Run the wizard and use this file to test functionality
4. Use "group" as the categorical variable

## Troubleshooting

### Common Issues

1. **"No file selected" error**
   - Make sure the file path is correct and the file exists
   - Check that the file format is supported (CSV or Excel)

2. **"Insufficient data points" error**
   - Ensure each group has at least 2 data points
   - Check for excessive missing values in your data

3. **"Analysis failed" for specific features**
   - Feature may contain only missing values
   - Feature may not be numeric
   - All values in the feature may be identical

4. **Import errors**
   - Verify all required packages are installed
   - Check Python version compatibility (tested with Python 3.7+)

### Performance Tips

- Large datasets (>10,000 rows) may take longer to process
- Consider selecting fewer features if analysis is slow
- Close other applications to free up memory for large datasets

## Extensions

The wizard can be extended to include:

- Post-hoc tests (Tukey's HSD, Bonferroni corrections)
- Two-way ANOVA for multiple factors
- Non-parametric alternatives (Kruskal-Wallis test)
- Multiple comparisons corrections
- Additional visualization options
- Integration with other analysis pipelines

## License

This software is part of the pyAPisolation package. See the main project license for details.
