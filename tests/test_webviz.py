"""
Tests for pyAPisolation webViz functionality

Note: These tests focus on non-visual aspects like config loading, data validation,
and HTML structure. Visual plot rendering requires manual validation.
"""
from pyAPisolation.webViz.tsDatabaseViewer import tsDatabaseViewer
from pyAPisolation.webViz.ephysDatabaseViewer import ephysDatabaseViewer
from pyAPisolation.webViz.webVizConfig import webVizConfig
from pyAPisolation.webViz.flaskApp import tsServer
import pandas as pd
from joblib import load
import os
import tempfile
import json
import yaml
from bs4 import BeautifulSoup
import pytest


def test_tsDatabase():
    """Test the tsDatabase class instantiation"""
    data_file = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')
    db = tsDatabaseViewer(data_file, file_index='filename', folder_path='foldername')
    assert isinstance(db.database, pd.DataFrame)


def test_ephysDatabase():
    """Test the ephysDatabase class instantiation"""
    data_file = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')
    db = ephysDatabaseViewer(data_file, file_index='filename', folder_path='foldername')
    assert isinstance(db.database, pd.DataFrame)


def test_config_defaults():
    """Test webVizConfig default values"""
    config = webVizConfig()
    assert config.file_index == 'filename.1'
    assert config.output_path == './'
    assert config.ext == '.abf'
    assert isinstance(config.table_vars, list)
    assert isinstance(config.color_schemes, list)


def test_config_json_load():
    """Test loading configuration from JSON file"""
    # Create temporary JSON config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_config = {
            'file_index': 'test_file',
            'output_path': '/test/output',
            'table_vars': ['var1', 'var2']
        }
        json.dump(test_config, f)
        temp_path = f.name
    
    try:
        config = webVizConfig(file=temp_path)
        assert config.file_index == 'test_file'
        assert config.output_path == '/test/output'
        assert config.table_vars == ['var1', 'var2']
    finally:
        os.unlink(temp_path)


def test_config_yaml_load():
    """Test loading configuration from YAML file"""
    # Create temporary YAML config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        test_config = {
            'file_index': 'test_file_yaml',
            'output_path': '/test/output/yaml',
            'para_vars': ['p1', 'p2', 'p3']
        }
        yaml.safe_dump(test_config, f)
        temp_path = f.name
    
    try:
        config = webVizConfig(file=temp_path)
        assert config.file_index == 'test_file_yaml'
        assert config.output_path == '/test/output/yaml'
        assert config.para_vars == ['p1', 'p2', 'p3']
    finally:
        os.unlink(temp_path)


def test_config_save_json():
    """Test saving configuration to JSON"""
    config = webVizConfig(file_index='saved_test', table_vars=['a', 'b', 'c'])
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save_to_json(temp_path)
        # Load back and verify
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        assert loaded['file_index'] == 'saved_test'
        assert loaded['table_vars'] == ['a', 'b', 'c']
    finally:
        os.unlink(temp_path)


def test_config_save_yaml():
    """Test saving configuration to YAML"""
    config = webVizConfig(file_index='saved_yaml', para_vars=['x', 'y', 'z'])
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save_to_yaml(temp_path)
        # Load back and verify
        with open(temp_path, 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded['file_index'] == 'saved_yaml'
        assert loaded['para_vars'] == ['x', 'y', 'z']
    finally:
        os.unlink(temp_path)


def test_config_programmatic():
    """Test setting config values programmatically"""
    config = webVizConfig(
        file_index='programmatic_test',
        output_path='/custom/path',
        table_vars=['v1', 'v2'],
        custom_option='custom_value'
    )
    assert config.file_index == 'programmatic_test'
    assert config.output_path == '/custom/path'
    assert config.custom_option == 'custom_value'


def test_flask_server_initialization():
    """Test Flask server can be instantiated"""
    config = webVizConfig()
    server = tsServer(config=config, static=False)
    assert server.app is not None
    assert hasattr(server, 'setup_routes')


def test_flask_path_validation():
    """Test path validation prevents directory traversal"""
    config = webVizConfig()
    server = tsServer(config=config, static=False)
    
    # Test invalid paths
    assert server._validate_path('../../../etc', 'passwd') is None
    assert server._validate_path('', '') is None
    assert server._validate_path(None, 'test') is None


def test_csv_data_validation():
    """Test that database viewer can handle different CSV formats"""
    # Create test CSV
    test_df = pd.DataFrame({
        'filename': ['file1', 'file2', 'file3'],
        'foldername': ['/path1', '/path2', '/path3'],
        'value1': [1.0, 2.0, 3.0],
        'value2': [10, 20, 30]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_df.to_csv(f, index=False)
        temp_path = f.name
    
    try:
        # Test that viewer can load the CSV
        db = ephysDatabaseViewer(temp_path, file_index='filename', folder_path='foldername')
        assert isinstance(db.database, pd.DataFrame)
        assert len(db.database) == 3
    finally:
        os.unlink(temp_path)


# Note: Visual rendering tests (HTML plots, Plotly graphs) require manual validation
# The following aspects should be tested manually:
# - UMAP scatter plot rendering
# - Parallel coordinates plot rendering  
# - Bootstrap table display
# - Interactive filtering between plots
# - Trace loading in dynamic mode
# - F-I curve display


if __name__ == "__main__":
    print("Running webViz tests...")
    test_tsDatabase()
    print("✓ tsDatabase instantiation")
    test_ephysDatabase()
    print("✓ ephysDatabase instantiation")
    test_config_defaults()
    print("✓ Config defaults")
    test_config_json_load()
    print("✓ JSON config loading")
    test_config_yaml_load()
    print("✓ YAML config loading")
    test_config_save_json()
    print("✓ JSON config saving")
    test_config_save_yaml()
    print("✓ YAML config saving")
    test_config_programmatic()
    print("✓ Programmatic config")
    test_flask_server_initialization()
    print("✓ Flask server initialization")
    test_flask_path_validation()
    print("✓ Path validation")
    test_csv_data_validation()
    print("✓ CSV data validation")
    print("\n✓ All webViz tests passed successfully!")
    print("\nNote: Visual plot rendering requires manual validation.")