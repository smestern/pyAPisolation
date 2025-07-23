#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the Post-hoc Analysis Runner Wizard
Creates sample data to test the wizard functionality
"""

import pandas as pd
import numpy as np
import os

def create_sample_data():
    """Create sample electrophysiological data similar to the provided CSV"""
    
    np.random.seed(42)  # For reproducible results
    
    # Sample parameters
    n_samples = 100
    groups = ['Control', 'Treatment_A', 'Treatment_B', 'Treatment_C']
    
    data = []
    
    for i in range(n_samples):
        # Random group assignment
        group = np.random.choice(groups)
        
        # Create group-dependent baseline values
        if group == 'Control':
            base_spike_count = 15
            base_amplitude = 50
            base_threshold = -40
        elif group == 'Treatment_A':
            base_spike_count = 10  # Reduced firing
            base_amplitude = 45
            base_threshold = -35
        elif group == 'Treatment_B':
            base_spike_count = 20  # Increased firing
            base_amplitude = 55
            base_threshold = -45
        else:  # Treatment_C
            base_spike_count = 8   # Much reduced firing
            base_amplitude = 40
            base_threshold = -30
        
        # Add noise and create features
        sample = {
            'filename': f'sample_{i:03d}',
            'foldername': f'folder_{i//20}',
            'protocol': 'IC1',
            'group': group,
            
            # Spike count features
            'mean_spike_count': base_spike_count + np.random.normal(0, 3),
            'max_spike_count': base_spike_count + np.random.normal(5, 2),
            'rheobase_spike_count': max(0, base_spike_count - 5 + np.random.normal(0, 2)),
            
            # Amplitude features  
            'mean_amplitude': base_amplitude + np.random.normal(0, 8),
            'peak_amplitude': base_amplitude + np.random.normal(10, 5),
            'min_amplitude': base_amplitude + np.random.normal(-5, 3),
            
            # Threshold features
            'mean_threshold': base_threshold + np.random.normal(0, 5),
            'rheobase_threshold': base_threshold + np.random.normal(-2, 3),
            
            # Timing features
            'mean_latency': 0.3 + np.random.normal(0, 0.1),
            'mean_isi': 0.05 + np.random.normal(0, 0.02),
            'isi_cv': 0.2 + np.random.normal(0, 0.1),
            
            # Membrane properties
            'input_resistance': 200 + np.random.normal(0, 50),
            'membrane_capacitance': 100 + np.random.normal(0, 20),
            'time_constant': 20 + np.random.normal(0, 5),
            
            # Additional features with group effects
            'adaptation_ratio': 0.8 + (0.1 if group == 'Treatment_A' else 0) + np.random.normal(0, 0.15),
            'sag_ratio': 0.1 + (0.05 if group == 'Treatment_B' else 0) + np.random.normal(0, 0.03),
            
            # Some features with no group effect (should be non-significant)
            'noise_feature_1': np.random.normal(100, 15),
            'noise_feature_2': np.random.normal(50, 10),
            'noise_feature_3': np.random.normal(0, 1),
        }
        
        data.append(sample)
    
    return pd.DataFrame(data)


def main():
    """Create sample data and save to CSV"""
    
    # Create sample data
    print("Creating sample electrophysiological data...")
    df = create_sample_data()
    
    # Save to CSV
    output_file = "sample_ephys_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Sample data saved to: {os.path.abspath(output_file)}")
    print(f"Data shape: {df.shape}")
    print(f"Groups: {df['group'].value_counts().to_dict()}")
    print("\nFeatures that should show significant differences:")
    print("- mean_spike_count, max_spike_count, rheobase_spike_count")
    print("- mean_amplitude, peak_amplitude")
    print("- mean_threshold, rheobase_threshold") 
    print("- adaptation_ratio, sag_ratio")
    print("\nFeatures that should NOT be significant:")
    print("- noise_feature_1, noise_feature_2, noise_feature_3")
    print("\nYou can now test the Post-hoc Analysis Wizard with this data!")
    print("Use 'group' as the categorical variable for ANOVA analysis.")


if __name__ == "__main__":
    main()
