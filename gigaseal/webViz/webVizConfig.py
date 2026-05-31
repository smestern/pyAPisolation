"""
Configuration class for webViz visualization settings

Supports both JSON and YAML formats for backward compatibility and ease of use.
"""
import json
import yaml
import os

class webVizConfig():
    """Configuration for webViz electrophysiology visualization
    
    This class manages all settings for the web visualization including:
    - Table columns to display
    - UMAP embedding parameters  
    - Parallel coordinates variables
    - Color schemes
    - File paths and extensions
    
    Configuration can be loaded from JSON or YAML files, or set programmatically.
    
    Examples:
        # Load from YAML
        config = webVizConfig(file='config.yaml')
        
        # Load from JSON (backward compatible)
        config = webVizConfig(file='config.json')
        
        # Set programmatically
        config = webVizConfig(table_vars=['voltage', 'current'], output_path='./results')
    """
    
    def __init__(self, file = None, **kwargs):
        # Default configuration values
        self.file_index = 'filename.1'
        self.folder_path = 'foldername.1'
        self.file_path = "foldername.1"
        self.primary_label = None
        self.table_vars_rq = ['filename', 'foldername']
        self.table_vars = ["rheo", "QC", 'label_c']
        self.para_vars = ["rheo", 'CRH', 'label_c']
        self.para_var_colors = 'rheobase_width'
        self.umap_cols = ['Umap X', 'Umap Y']
        self.umap_labels = ['label', 'CRH', 'AVP', 'foldername.1']
        self.color_schemes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        self.para_vars_limit = 10
        self.table_vars_limit = 5
        self.plots_path = None
        self.output_path = './'
        self.ext='.abf'
        
        # Update with provided kwargs first
        self.__dict__.update(kwargs)
        
        # Load from file if provided
        if file:
            self.load_from_file(file)

        # Clean up col_rename if present
        if self.col_rename:
            self.col_rename = {k:v for k,v in self.col_rename.items() if v}

    def load_from_file(self, filepath):
        """Load configuration from JSON or YAML file
        
        Format is auto-detected based on file extension.
        
        Args:
            filepath: Path to .json or .yaml/.yml file
        """
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        with open(filepath, 'r') as f:
            if ext in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif ext == '.json':
                config_data = json.load(f)
            else:
                # Try JSON first, fallback to YAML
                f.seek(0)
                try:
                    config_data = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    config_data = yaml.safe_load(f)
        
        self.__dict__.update(config_data)
    
    def save_to_json(self, filepath):
        """Save configuration to JSON file
        
        Args:
            filepath: Path to output .json file
        """
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def save_to_yaml(self, filepath):
        """Save configuration to YAML file
        
        Args:
            filepath: Path to output .yaml file
        """
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(filepath, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    
    def update(self, kwargs):
        self.__dict__.update(kwargs)
        return self 
    
    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            return None
        

    def process_rename(self, df):
        """ Rename the columns of the dataframe according to the col_rename attribute. This function will also update the attributes of the class where necessary. 
        The function will return the dataframe with the columns renamed. 
        takes:
            df: pd.DataFrame
        """
        if self.col_rename is None:
            return df
        # Rename columns of the df
        if self.col_rename:
            df.rename(columns=self.col_rename, inplace=True)
        
        old_names = self.col_rename.keys()

        def recursive_update(value):
            if isinstance(value, str) and value in old_names:
                return self.col_rename[value]
            elif isinstance(value, list):
                return [recursive_update(item) for item in value]
            elif isinstance(value, dict):
                return {recursive_update(k): recursive_update(v) for k, v in value.items()}
            else:
                return value

        # Update our attributes where they need to be updated
        for k, v in self.__dict__.items():
            self.__dict__[k] = recursive_update(v)

        return df

