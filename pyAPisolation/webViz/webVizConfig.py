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
        self.color_schemes
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
        #rename columns of the df
        if self.col_rename:
            df.rename(columns=self.col_rename, inplace=True)
        #update our attributes where they need to be updated
        old_names = self.col_rename.keys()
        for k, v in self.__dict__.items():
            if isinstance(v, str) and v in old_names:
                self.__dict__[k] = self.col_rename[v]
            if isinstance(v, list): #only go one level deep, if we have a list of lists, we will not catch it could be a problem
                self.__dict__[k] = [self.col_rename[i] if i in old_names else i for i in v]
            if isinstance(v, dict):
                self.__dict__[k] = {self.col_rename[i] if i in old_names else i: j for i,j in v.items()}

        return df

