import json

class web_viz_config():
    def __init__(self, file = None, **kwargs):
        self.file_index = 'filename.1'
        self.file_path = "foldername.1"
        self.table_vars_rq = ['filename', 'foldername']
        self.table_vars = ["rheo", "QC", 'label_c']
        self.para_vars = ["rheo", 'CRH', 'label_c']
        self.para_var_colors = 'rheobase_width'
        self.umap_labels = ['label', 'CRH', 'AVP', 'foldername.1']
        self.para_vars_limit = 10
        self.table_vars_limit = 5
        self.plots_path = None
        self.__dict__.update(kwargs)
        if file:
            with open(file, 'r') as f:
                self.__dict__.update(json.load(f))

