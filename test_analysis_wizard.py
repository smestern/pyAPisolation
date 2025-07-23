#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete test and demonstration script for the Post-hoc Analysis Runner Wizard

This script:
1. Creates sample electrophysiological data
2. Launches the analysis wizard 
3. Demonstrates the complete workflow

Run this to test the wizard functionality end-to-end.
"""

import sys
import os
import pandas as pd
import numpy as np
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog

# Add the pyAPisolation package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyAPisolation.gui.postAnalysisRunner import PostAnalysisWizard


def create_comprehensive_test_data():
    """Create comprehensive test data with multiple experimental conditions"""
    
    np.random.seed(42)  # For reproducible results
    
    # Experimental design
    conditions = {
        'Wildtype_Control': {
            'n': 25,
            'spike_count_base': 15,
            'amplitude_base': 50,
            'threshold_base': -40,
            'latency_base': 0.35,
            'resistance_base': 200
        },
        'Wildtype_Drug': {
            'n': 25, 
            'spike_count_base': 10,  # Drug reduces firing
            'amplitude_base': 48,
            'threshold_base': -38,
            'latency_base': 0.45,    # Drug increases latency
            'resistance_base': 220
        },
        'Knockout_Control': {
            'n': 20,
            'spike_count_base': 8,   # KO has reduced firing
            'amplitude_base': 42,
            'threshold_base': -35,
            'latency_base': 0.40,
            'resistance_base': 250   # KO has higher resistance
        },
        'Knockout_Drug': {
            'n': 20,
            'spike_count_base': 3,   # KO + drug severely reduces firing
            'amplitude_base': 38,
            'threshold_base': -32,
            'latency_base': 0.55,    # Highest latency
            'resistance_base': 280
        }
    }
    
    data = []
    
    for condition, params in conditions.items():
        for i in range(params['n']):
            # Extract genotype and treatment
            genotype, treatment = condition.split('_')
            
            # Create realistic inter-animal variability
            animal_id = f"{genotype}_{treatment}_{i//5 + 1:02d}"  # 5 cells per animal
            cell_id = f"{animal_id}_cell_{i%5 + 1}"
            
            # Generate features with realistic correlations and noise
            spike_count = max(0, params['spike_count_base'] + np.random.normal(0, 3))
            amplitude = params['amplitude_base'] + np.random.normal(0, 6)
            threshold = params['threshold_base'] + np.random.normal(0, 4)
            latency = max(0.1, params['latency_base'] + np.random.normal(0, 0.08))
            resistance = max(50, params['resistance_base'] + np.random.normal(0, 40))
            
            # Correlated features (spike count affects other measures)
            width = 0.8 + (15 - spike_count) * 0.02 + np.random.normal(0, 0.15)  # Lower firing -> wider spikes
            adaptation = 0.85 - spike_count * 0.01 + np.random.normal(0, 0.12)   # Higher firing -> more adaptation
            
            # Create data record
            record = {
                # Identifiers
                'filename': f"{cell_id}.abf",
                'foldername': animal_id,
                'cell_id': cell_id,
                'animal_id': animal_id,
                'genotype': genotype,
                'treatment': treatment,
                'condition': condition,
                'protocol': 'IC1',
                
                # Primary spike features (should show group differences)
                'spike_count_rheobase': max(0, spike_count - 5 + np.random.normal(0, 1.5)),
                'spike_count_mean': spike_count,
                'spike_count_max': spike_count + np.random.normal(3, 1.5),
                
                # Amplitude features
                'amplitude_mean': amplitude,
                'amplitude_peak': amplitude + np.random.normal(8, 3),
                'amplitude_threshold': threshold,
                
                # Timing features
                'latency_first_spike': latency,
                'latency_mean': latency + np.random.normal(0.02, 0.01),
                'spike_width_mean': max(0.3, width),
                'isi_mean': 0.05 + np.random.exponential(0.02),
                'isi_cv': 0.3 + np.random.normal(0, 0.15),
                
                # Membrane properties
                'input_resistance': resistance,
                'membrane_capacitance': 120 + np.random.normal(0, 25),
                'time_constant': resistance * 0.1 + np.random.normal(0, 8),
                'resting_potential': -65 + np.random.normal(0, 8),
                
                # Adaptation and sag
                'adaptation_ratio': max(0, min(1, adaptation)),
                'sag_ratio': 0.15 + np.random.normal(0, 0.08),
                
                # Voltage-clamp features (if applicable)
                'holding_current': np.random.normal(-20, 15),
                'series_resistance': 15 + np.random.normal(0, 5),
                
                # Some features with no group effect (should be non-significant)
                'noise_feature_1': np.random.normal(100, 20),
                'noise_feature_2': np.random.normal(0, 1),
                'random_measure': np.random.uniform(0, 10),
                
                # QC measures
                'baseline_stability': np.random.uniform(0.8, 1.0),
                'recording_quality': np.random.uniform(0.7, 1.0),
                'temperature': 32 + np.random.normal(0, 1),
                
                # Additional derived measures
                'excitability_index': spike_count / max(1, abs(threshold) * 0.1),
                'efficiency_ratio': spike_count / max(1, latency * 100),
            }
            
            data.append(record)
    
    return pd.DataFrame(data)


def print_data_summary(df):
    """Print a summary of the generated data"""
    print("\n" + "="*60)
    print("GENERATED TEST DATA SUMMARY")
    print("="*60)
    
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    print("\nExperimental groups:")
    for condition, count in df['condition'].value_counts().sort_index().items():
        print(f"  {condition}: {count} cells")
    
    print(f"\nAnimals per group:")
    for condition in df['condition'].unique():
        n_animals = df[df['condition'] == condition]['animal_id'].nunique()
        print(f"  {condition}: {n_animals} animals")
    
    print("\nFeatures that SHOULD show significant differences:")
    significant_features = [
        'spike_count_rheobase', 'spike_count_mean', 'spike_count_max',
        'amplitude_mean', 'amplitude_peak', 'amplitude_threshold',
        'latency_first_spike', 'latency_mean', 'input_resistance',
        'excitability_index', 'efficiency_ratio'
    ]
    for feat in significant_features:
        if feat in df.columns:
            print(f"  - {feat}")
    
    print("\nFeatures that should NOT be significant (controls):")
    control_features = ['noise_feature_1', 'noise_feature_2', 'random_measure']
    for feat in control_features:
        if feat in df.columns:
            print(f"  - {feat}")
    
    print("\nSuggested grouping variables for analysis:")
    grouping_vars = ['condition', 'genotype', 'treatment']
    for var in grouping_vars:
        if var in df.columns:
            print(f"  - {var} ({len(df[var].unique())} groups)")
    
    print("\n" + "="*60)


def main():
    """Main function to create data and launch wizard"""
    
    print("Post-hoc Analysis Wizard - Test and Demonstration")
    print("=" * 55)
    
    # Create comprehensive test data
    print("1. Creating comprehensive test dataset...")
    df = create_comprehensive_test_data()
    
    # Save test data
    output_file = "comprehensive_test_data.csv"
    df.to_csv(output_file, index=False)
    print(f"   Saved to: {os.path.abspath(output_file)}")
    
    # Print summary
    print_data_summary(df)
    
    # Ask user if they want to launch the wizard
    print("\n2. Launching the Analysis Wizard...")
    print("   The wizard will open in a new window.")
    print("   Use the generated CSV file for testing.")
    print("\nRecommended testing workflow:")
    print("   1. Load the comprehensive_test_data.csv file")
    print("   2. Try different grouping variables:")
    print("      - 'condition' (4 groups) - most comprehensive")
    print("      - 'genotype' (2 groups) - Wildtype vs Knockout")  
    print("      - 'treatment' (2 groups) - Control vs Drug")
    print("   3. Select multiple features for analysis")
    print("   4. Review results and export if desired")
    
    # Create and run the application
    app = QApplication(sys.argv)
    app.setApplicationName("Post-hoc Analysis Wizard - Test")
    
    try:
        # Launch the wizard
        wizard = PostAnalysisWizard()
        wizard.show()
        
        # Show info message
        QMessageBox.information(
            wizard,
            "Test Data Created",
            f"Test data has been created: {os.path.abspath(output_file)}\n\n"
            "Use this file to test the wizard functionality.\n\n"
            "Recommended grouping variable: 'condition'\n"
            "Expected significant features: spike counts, amplitudes, latencies"
        )
        
        # Run the application
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"\nError launching wizard: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install PySide2 pandas numpy scipy matplotlib seaborn openpyxl")
        sys.exit(1)


if __name__ == "__main__":
    main()
