#!/usr/bin/env python3
"""
Convert JSON webViz configuration to YAML format

This utility helps migrate from legacy JSON configs to the more readable YAML format.
"""

import argparse
import json
import yaml
import os
import sys

def convert_json_to_yaml(json_path, yaml_path=None, overwrite=False):
    """Convert a JSON config file to YAML format
    
    Args:
        json_path: Path to input JSON file
        yaml_path: Path to output YAML file (default: same name with .yaml extension)
        overwrite: If True, overwrite existing YAML file
        
    Returns:
        Path to created YAML file
    """
    # Validate input file
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Determine output path
    if yaml_path is None:
        base = os.path.splitext(json_path)[0]
        yaml_path = base + '.yaml'
    
    # Check if output exists
    if os.path.exists(yaml_path) and not overwrite:
        response = input(f"File {yaml_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            return None
    
    # Load JSON
    try:
        with open(json_path, 'r') as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    
    # Save as YAML
    with open(yaml_path, 'w') as f:
        # Add header comment
        f.write("# pyAPisolation WebViz Configuration\n")
        f.write(f"# Converted from {os.path.basename(json_path)}\n\n")
        
        # Write YAML with nice formatting
        yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Successfully converted {json_path} to {yaml_path}")
    return yaml_path

def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON webViz config to YAML format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert config.json to config.yaml
  python convert_json_to_yaml.py config.json
  
  # Specify output filename
  python convert_json_to_yaml.py old_config.json -o new_config.yaml
  
  # Overwrite existing file without prompting
  python convert_json_to_yaml.py config.json --overwrite
        """
    )
    
    parser.add_argument('json_file', 
                        help='Path to JSON configuration file')
    parser.add_argument('-o', '--output', 
                        help='Output YAML file path (default: <input>.yaml)')
    parser.add_argument('--overwrite', 
                        action='store_true',
                        help='Overwrite existing YAML file without prompting')
    parser.add_argument('--validate', 
                        action='store_true',
                        help='Validate converted file by re-loading it')
    
    args = parser.parse_args()
    
    try:
        yaml_path = convert_json_to_yaml(
            args.json_file, 
            args.output, 
            args.overwrite
        )
        
        if yaml_path and args.validate:
            # Test loading the YAML file
            from pyAPisolation.webViz.webVizConfig import webVizConfig
            config = webVizConfig(file=yaml_path)
            print(f"✓ Validation successful: config loaded correctly")
            print(f"  - {len(vars(config))} configuration parameters")
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
