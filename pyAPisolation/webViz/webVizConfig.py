import json

class webVizConfig():
    def __init__(self, file = None, **kwargs):
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
        self.__dict__.update(kwargs)
        if file:
            with open(file, 'r') as f:
                self.__dict__.update(json.load(f))


        if self.col_rename:
            self.col_rename = {k:v for k,v in self.col_rename.items() if v}

    
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

