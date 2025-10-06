#!/usr/bin/env python3
"""
Test script for testing the from_dataframe method with demo data
"""

import sys
import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

# Add the parent directory to sys.path to import pyAPisolation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyAPisolation.database.tsDatabase import tsDatabase

def test_demo_dataframe_import():
    """Test importing the demo arbitrary data CSV"""
    
    # Load the demo data
    demo_path = os.path.join(os.path.dirname(__file__), 'test_data', 'demo_arb_data.csv')
    
    if not os.path.exists(demo_path):
        print(f"Demo data file not found: {demo_path}")
        return False
        
    df = pd.read_csv(demo_path, skiprows=2)
    print(f"Loaded demo data with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create database instance
    db = tsDatabase()
    
    # Test auto-detection of protocol columns
    result = db.from_dataframe(
        df, 
        cell_id_col='CELL_ID',
        metadata_cols=['DATE', 'drug', 'NOTE', 'Burst Adex', 'Burst Cadex']
    )
    
    if result:
        print("✅ Successfully imported demo data with auto-detection")
        print(f"Database contains {len(db.cellindex)} cells")
        print(f"Protocol columns: {len(db.cellindex.columns)} total columns")
        
        # Show some sample data
        print("\nSample cells:")
        for i, (cell_name, cell_data) in enumerate(db.getCells().items()):
            if i >= 3:  # Show first 3 cells
                break
            print(f"  {cell_name}: {len([v for v in cell_data.values() if v is not None and v != ''])} protocols")
        
        return True
    else:
        print("❌ Failed to import demo data")
        return False

def test_specific_protocols():
    """Test with manually specified protocol columns"""
    
    demo_path = os.path.join(os.path.dirname(__file__), 'test_data', 'demo_arb_data.csv')
    df = pd.read_csv(demo_path, skiprows=2)
    
    # Create database instance
    db = tsDatabase()
    
    # Test with specific protocol columns
    specific_protocols = ['IC1', 'CTRL_PULSE', 'NET_PULSE', 'DYN_CFG1_EXP3', 'DYN_CFG1_EXP4']
    
    result = db.from_dataframe(
        df,
        filename_cols=specific_protocols,
        cell_id_col='CELL_ID', 
        metadata_cols=['DATE', 'drug', 'NOTE']
    )
    
    if result:
        print("✅ Successfully imported demo data with specific protocols")
        print(f"Database contains {len(db.cellindex)} cells")
        
        # Check that our specific protocols are there
        for protocol in specific_protocols:
            if protocol in db.cellindex.columns:
                non_empty = db.cellindex[protocol].notna().sum()
                print(f"  {protocol}: {non_empty} cells have this protocol")
            else:
                print(f"  {protocol}: NOT FOUND in database")
        
        return True
    else:
        print("❌ Failed to import demo data with specific protocols")
        return False

def test_indiv_rows():
    """Test that individual rows are correctly imported into the database"""
    demo_path = os.path.join(os.path.dirname(__file__), 'test_data', 'demo_arb_data.csv')
    if not os.path.exists(demo_path):
        print(f"Demo data file not found: {demo_path}")
        return False
    df = pd.read_csv(demo_path, skiprows=2)
    db = tsDatabase()
    result = db.from_dataframe(
        df,
        cell_id_col='CELL_ID',
        metadata_cols=['DATE', 'drug', 'NOTE', 'Burst Adex', 'Burst Cadex']
    )
    if not result:
        print("❌ Failed to import demo data in test_indiv_rows")
        return False
    # Check a few individual rows
    sample_rows = df.head(3)
    all_passed = True
    for idx, row in sample_rows.iterrows():
        cell_id = row['CELL_ID']
        if cell_id not in db.cellindex.index:
            print(f"❌ Cell ID {cell_id} not found in database index")
            all_passed = False
        else:
            print(f"✅ Cell ID {cell_id} found in database index")
    return all_passed
if __name__ == "__main__":
    print("Testing tsDatabase.from_dataframe() with demo data...")
    print("=" * 60)
    
    # Test 1: Auto-detection
    print("Test 1: Auto-detection of protocol columns")
    success1 = test_demo_dataframe_import()
    
    print("\n" + "=" * 60)
    
    # Test 2: Specific protocols
    print("Test 2: Manually specified protocol columns")
    success2 = test_specific_protocols()
    
    print("\n" + "=" * 60)
    

