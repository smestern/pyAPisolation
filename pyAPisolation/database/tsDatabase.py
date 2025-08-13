import numpy as np
import pandas as pd
import os
import glob
import sys
import anndata as ad
import logging
from ..patch_utils import df_select_by_col
from ..dataset import cellData
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DEFAULT_META_COLS = ['cell_type', 'time', 'cell_id', 'protocol', 'primary', 'filename', 'foldername', 'depolarizing_current', 'IC1_protocol_check', 'sample_rate']
DEFAULT_FEAT_COLS = ['']

class experimentalStructure:
    """
    A class to represent the experimental structure of a time series database. Each entry represents a protocol,
    and each protocol has a set of flags, (e.g. time, cell type, etc.).
    One flag is the primairy flag, which is used to flag the protocool the represents the primary time series (for computing features etc).

    """
    
    def __init__(self):
        """
        Constructor for the experimentalStructure class
        """
        self.protocols = pd.DataFrame(data=None, columns=['name', 'altnames'])
        self.primary = None

    def addProtocol(self, name, flags):
        """
        Add a protocol to the experimental structure
        :param name: Name of the protocol
        :param flags: Flags of the protocol
        """
        #check if the protocol already exists in the dataframe either by name or by altnames
        altnames = np.ravel([x for x in self.protocols['altnames'].values])
        if name in self.protocols['name'].values or name in altnames:
            logger.info(f'Protocol {name} already exists in the database')

            if name in self.protocols['altnames'].values:
                #if the name is in the altnames, we will update the name to the name in the dataframe
                name = self.protocols[self.protocols['altnames'] == name]['name'].values[0]
            #if any of the flags are different, we will make a new entry
            for i in range(len(self.protocols)):
                if self.protocols['name'][i] == name:
                    for key in flags.keys():
                        if key == 'name':
                            continue
                        if flags[key] != self.protocols[key][i]:
                            logger.info(f'Flags for protocol {name} are different, making a new entry')
                            self.protocols = self.protocols.append(pd.DataFrame(data={'name': name, **flags}))
                            
        else:
            self.protocols = self.protocols.append(pd.DataFrame(data={'name': name, 'altnames': [name], **flags}))

        #deep copy the dataframe to avoid any issues
        self.protocols = self.protocols.copy() #this is a bit of a hack, but it works for now
    
    def getProtocol(self, name):
        """
        Get the protocol by name
        :param name: Name of the protocol
        """
        #check if the protocol exists in the dataframe either by name or by altnames
        altnames = np.ravel([x for x in self.protocols['altnames'].values])
        if name in self.protocols['name'].values or name in altnames:
            if name in self.protocols['altnames'].values:
                #if the name is in the altnames, we will update the name to the name in the dataframe
                name = self.protocols[self.protocols['altnames'] == name]['name'].values[0]
            return self.protocols[self.protocols['name'] == name]
        else:
            logger.error(f'Protocol {name} does not exist in the database')
            return None

    def setPrimary(self, name):
        """
        Set the primary protocol
        :param name: Name of the primary protocol
        """
        self.primary = name


class tsDatabase:
    """
        A class to represent a time series database. Overall this class will a database style of indexing of files.
        The user will provide a experimental structure, and the class will provide a simple interface to load the data. 
        Each DB entry will be a single cell, it will have metadata describing the cell, and linking the different files to the cell.
        It will also have a table of features, and a table of time series data (hot-loaded from the files).
        
    """
    ## internally this looks something like
    ## Subject | Protocol 1 | Protocol 2 | ... | Protocol N | Metadata | Features | Time Series
    ## 1       | file1      | file2      | ... | fileN      | meta1    | feat1    | ts1
    ## 2       | file1      | file2      | ... | fileN      | meta2    | feat2    | ts2
    ## ...     | ...        | ...        | ... | ...        | ...      | ...      | ...
    #internally we will use annData to store the data, externally we use a excel sheet to store the metadata and features. This is not ideal but end users want to be able to edit the metadata and features in excel.
    #we could use a sqlite database to store the metadata and features, but this would require a lot of extra code to handle the database, for now we will use this simple solution.


    def __init__(self, path=None, exp=None, dataframe = None, **kwargs):
        """
        Constructor for the tsDatabase class
        :param path: Path to the database
        :param exp: Experimental structure of the database
        """
        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path

        if exp is None:
            self.exp = experimentalStructure()
        else:
            self.exp = exp

        #we have a cellindex df representing the df for cell index
        self.cellindex = pd.DataFrame()
        self.data = {}
        #if a dataframe is passed in here, we will use it to populate the database
        if dataframe is not None:
            self.dataframe = dataframe
            self.fromDataFrame(dataframe, self=self, **kwargs)
        else:
            self.dataframe = None

        


    def load(self, path):
        """
        Load the database from a path
        :param path: Path to the database
        """
        pass

    @classmethod
    def fromDataFrame(cls, dataframe, **kwargs):
        """
        Load the database from a dataframe
        :param dataframe: Dataframe to load from
        """
        #if self is in kwargs, we will load the data into that object, otherwise we will create a new object
        if 'self' in kwargs:
            logger.info('Loading data into existing object')
            self = kwargs.pop('self')
            #self = kwargs['self']
        else:
            logger.info('Creating new object')
            self = cls()
        
        data_obj = self.parseDataFrame(dataframe, **kwargs)

        #this will be our primary data object

        self.exp.addProtocol(data_obj.obs['protocol'][0], {'altnames': np.unique(data_obj.obs['protocol'])})
        self.exp.primary = data_obj.obs['protocol'][0]

        self.data = {data_obj.obs['protocol'][0]: data_obj}

        #create our cell index, cell names will be CELL_X_{primary_protocol_file_name}
        cell_names = [f'CELL_{i}_{data_obj.obs_names[i]}' for i in range(len(data_obj.obs))]

        self.cellindex = pd.DataFrame(index=cell_names, columns=['protocol', 'filename', 'foldername', data_obj.obs['protocol'][0]], data={'protocol': data_obj.obs['protocol'][0], 'filename': data_obj.obs_names, 'foldername': data_obj.obs['foldername'], data_obj.obs['protocol'][0]: data_obj.obs_names})


    def parseDataFrame(self, dataframe, **kwargs):
        """
        Parse the dataframe in a format that can be used by the database
        :param dataframe: Dataframe to parse
        """

        #load the metadata and features from the dataframe
        if 'id_col' in kwargs:
            id_col = kwargs['id_col']
        else:
            id_col = None
        
        if 'meta_cols' in kwargs:
            meta_cols = kwargs['meta_cols']
        else:
            meta_cols = DEFAULT_META_COLS

        if 'feature_cols' in kwargs:
            feature_cols = kwargs['feature_cols']
        else:
            feature_cols = DEFAULT_FEAT_COLS

        if 'time_series_cols' in kwargs:
            time_series_cols = kwargs['time_series_cols']
        else:
            time_series_cols = None

        dataframe.set_index(id_col, inplace=True) if id_col is not None else None

        #load the metadata
        meta = df_select_by_col(dataframe, meta_cols)
        
        #in our case features must not be in the metadata

        #load the features
        features = df_select_by_col(dataframe, feature_cols)
        
        #drop meta features from the features
        features = features.drop(meta.columns.values, axis=1)

        #time series will be hotloaded from the files, but for now we want to make the parsing of the dataframe as simple as possible
        if time_series_cols is not None:
            time_series = df_select_by_col(dataframe, time_series_cols)
        else:
            time_series = None
        
        #var is gonna be features name and protocol source
        feature_name = features.columns
        protocol_source = [meta['protocol'] for i in range(len(features.columns))]
        var = pd.DataFrame(index=features.columns, data={'feature_name': feature_name, 'protocol_source': protocol_source})


        data = ad.AnnData(X=features, obs=meta, var=var)
        #data.obs_names = meta.columns.values
        #data.var_names = features.columns
        return data

    @staticmethod
    def parseFile(file):
        #use celldata to parse the file
        cell = cellData(file)
        #now build a dict
        file_dict = {'filename': cell.fileName, 'protocol': cell.protocol}
        return file_dict

    def addEntry(self, name, paths=None):
        """
        Add an entry to the database
        :param name: Name of the entry
        :param path: Path to the entry
        """
        #create a row for adding
        row = pd.DataFrame(index=[name], columns=['name'], data=[name])
        logger.info(f'Adding entry {name} to the database')
        if paths is not None:
            logger.info(f'Adding files {paths} to the database')
            if isinstance(paths, str) or isinstance(paths, os.PathLike):
                file_dicts = [self.parseFile(paths)]
            elif isinstance(paths, list):
                file_dicts = [self.parseFile(path) for path in paths]
            else:
                logger.error(f'Invalid path type {type(paths)}')
                
            #with the diles

        else:
            #update cellIndex
            self.cellindex = pd.concat([self.cellindex, row]).copy()
        
    def addEntries(self, path):
        """
        Add multiple entries to the database
        :param path: Path to the entries
        """
        pass

    def loadEntry(self, name):
        """
        Load an entry from the database
        :param name: Name of the entry
        """
        pass
            
    def updateEntry(self, name, **kwargs):
        """
        Update an entry in the database
        :param name: Name of the entry
        """
        #this will be a bit tricky, we will have to update the cell index and the data, assuming the individual passes in the columns to update via kwargs
        #we will also have to update the experimental structure
        #update the cell index
        if name in self.cellindex.index:
            #update the cell index
            for key in kwargs.keys():
                if key in self.cellindex.columns:
                    self.cellindex.loc[name, key] = kwargs[key]
                else:
                    logger.error(f'Key {key} not found in cell index')
        else:
            logger.error(f'Entry {name} not found in the database')

    def getEntries(self):
        """
        Get all entries from the database
        """
        return self.cellindex.to_dict(orient='records')
    
    def getCells(self):
        """
        Get all cells from the database
        """
        return self.cellindex.to_dict(orient='index')
    
    def addProtocol(self, cell, protocol, **kwargs):
        """
        Add a protocol to a cell
        :param cell: Cell to add the protocol to
        :param protocol: Protocol to add
        """
        #check if the protocol exists in the database
        path = kwargs.pop('path', None) #pop the path from the kwargs
        if self.exp.getProtocol(protocol) is None:
            logger.info(f'Protocol {protocol} does not exist in the database')
            self.exp.addProtocol(protocol, kwargs)
            #make a new column in the cell index
            self.cellindex[protocol] = None
        else:
            logger.info(f'Protocol {protocol} exists in the database')
        
        #update the cell index
        if path is not None:
            self.cellindex.loc[cell, protocol] = path
        else:
            self.cellindex.loc[cell, protocol] = None

    def __getitem__(self, key, protocol, column):
        return self.data[key]
    
    def __setitem__(self, key, protocol, column, value):
        self.data[key] = value

    def __delitem__(self, key, protocol, column):
        del self.data[key]

    def save(self, path):
        """
        Save the database to an Excel file with multiple sheets
        :param path: Path to save the Excel file
        """
        import pandas as pd
        
        # Ensure the path has .xlsx extension
        if not path.endswith('.xlsx'):
            path = path + '.xlsx'
        
        # Create Excel writer object
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # Save main cell index data
            if not self.cellindex.empty:
                self.cellindex.to_excel(writer, sheet_name='CellIndex',
                                        index=True)
            else:
                # Create empty sheet with headers if no data
                empty_df = pd.DataFrame(columns=['Cell Name', 'Notes'])
                empty_df.to_excel(writer, sheet_name='CellIndex',
                                  index=False)
            
            # Save experimental structure/protocols as a separate sheet
            if not self.exp.protocols.empty:
                self.exp.protocols.to_excel(writer, sheet_name='Protocols',
                                            index=False)
            else:
                # Create empty protocols sheet
                protocol_cols = ['name', 'altnames', 'description',
                                 'pharma', 'temp']
                empty_protocols = pd.DataFrame(columns=protocol_cols)
                empty_protocols.to_excel(writer, sheet_name='Protocols',
                                         index=False)
            
            # Save configuration information
            config_data = {
                'version': ['1.0'],
                'created_by': ['pyAPisolation'],
                'database_type': ['tsDatabase'],
                'path': [self.path]
            }
            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name='_cdb_config',
                               index=False)
            
            # Save metadata about the database structure
            metadata = {
                'total_cells': [len(self.cellindex)],
                'total_protocols': [len(self.exp.protocols)],
                'columns': [list(self.cellindex.columns)],
                'index_name': [self.cellindex.index.name or 'Cell']
            }
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_excel(writer, sheet_name='_cdb_metadata',
                                 index=False)
        
        logger.info(f'Database saved to {path}')
        return path

    def load_from_excel(self, path):
        """
        Load the database from an Excel file
        :param path: Path to the Excel file
        """
        import pandas as pd
        
        try:
            # Read the main cell index
            self.cellindex = pd.read_excel(path, sheet_name='CellIndex',
                                           index_col=0)
            
            # Read protocols if they exist
            try:
                protocols_df = pd.read_excel(path, sheet_name='Protocols')
                self.exp.protocols = protocols_df
            except Exception as e:
                logger.warning(f'No Protocols sheet found: {e}')
            
            # Read config if it exists
            try:
                config_df = pd.read_excel(path, sheet_name='_cdb_config')
                if 'path' in config_df.columns and len(config_df) > 0:
                    self.path = config_df['path'].iloc[0]
            except Exception as e:
                logger.warning(f'No config sheet found: {e}')
                
            logger.info(f'Database loaded from {path}')
            
        except Exception as e:
            logger.error(f'Error loading database from {path}: {e}')
            raise e

    def import_spike_data(self, csv_path):
        """
        Import spike analysis data from a CSV file (e.g., spike_count_.csv)
        Each row represents a file with features from spike finder output.
        
        :param csv_path: Path to the CSV file containing spike analysis data
        """
        try:
            # Read the CSV file
            spike_df = pd.read_csv(csv_path)
            logger.info(f"Loading spike analysis data from {csv_path}")
            logger.info(f"Found {len(spike_df)} files with spike features")
            
            # Extract key columns for mapping to database entries
            # Based on CSV structure: foldername, filename, protocol columns
            if 'filename' not in spike_df.columns:
                raise ValueError("CSV must contain a 'filename' column")
            
            if 'protocol' not in spike_df.columns:
                raise ValueError("CSV must contain a 'protocol' column")
            
            # Get the feature columns (excluding metadata columns)
            metadata_cols = ['foldername', 'filename', 'protocol']
            feature_cols = [col for col in spike_df.columns
                            if col not in metadata_cols]
            
            # Track successful imports and issues
            imported_count = 0
            issues = []
            
            # Process each row in the CSV
            for idx, row in spike_df.iterrows():
                filename = row['filename']
                protocol = row['protocol']
                folder = row.get('foldername', '')
                
                # Find matching entry in cellindex
                # Look for entries where the filename matches
                matching_entries = []
                
                if not self.cellindex.empty:
                    # Try to match by filename in any protocol column
                    for cell_name in self.cellindex.index:
                        cell_row = self.cellindex.loc[cell_name]
                        
                        # Check if filename appears in any protocol column
                        for col in self.cellindex.columns:
                            if col in ['protocol', 'filename', 'foldername']:
                                continue
                            cell_value = cell_row.get(col, '')
                            if (isinstance(cell_value, str) and
                                    filename in cell_value):
                                matching_entries.append(cell_name)
                                break
                        
                        # Also check the filename column directly
                        if cell_row.get('filename', '') == filename:
                            matching_entries.append(cell_name)
                
                if not matching_entries:
                    # No matching entry found, create a new cell entry
                    cell_name = f'CELL_{imported_count}_{filename}'
                    
                    # Add to cellindex
                    new_row = {
                        'protocol': protocol,
                        'filename': filename,
                        'foldername': folder,
                        protocol: filename
                    }
                    
                    # Add all feature columns as additional metadata/features
                    for feat_col in feature_cols:
                        new_row[f'spike_{feat_col}'] = row[feat_col]
                    
                    # Add the new entry to cellindex
                    if self.cellindex.empty:
                        self.cellindex = pd.DataFrame([new_row],
                                                      index=[cell_name])
                    else:
                        self.cellindex.loc[cell_name] = new_row
                    
                    imported_count += 1
                    logger.info(f"Created new cell entry: {cell_name}")
                    
                else:
                    # Update existing entries with spike features
                    for cell_name in matching_entries:
                        for feat_col in feature_cols:
                            col_name = f'spike_{feat_col}'
                            self.cellindex.loc[cell_name, col_name] = (
                                row[feat_col])
                    
                    imported_count += len(matching_entries)
                    logger.info(f"Updated {len(matching_entries)} existing "
                                f"entries for {filename}")
            
            # Add spike analysis protocol to experimental structure
            spike_protocol_name = 'spike_analysis'
            protocol_names = [p for p in self.exp.protocols['name']
                              if isinstance(p, str)]
            if spike_protocol_name not in protocol_names:
                self.exp.addProtocol(spike_protocol_name, {
                    'altnames': ['spike_count', 'spike_finder'],
                    'description': 'Spike analysis features from spike output'
                })
            
            logger.info("Successfully imported spike data:")
            logger.info(f"  - {imported_count} entries processed")
            logger.info(f"  - {len(feature_cols)} feature columns added")
            logger.info("  - Features prefixed with 'spike_' in database")
            
            if issues:
                logger.warning("Issues encountered during import:")
                for issue in issues[:10]:  # Show first 10 issues
                    logger.warning(f"  - {issue}")
                if len(issues) > 10:
                    msg = f"  - ... and {len(issues) - 10} more issues"
                    logger.warning(msg)
            
            return {
                'imported_count': imported_count,
                'feature_cols': feature_cols,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Error importing spike data from {csv_path}: {e}")
            raise

    def import_spike_data_row(self, row_data, import_options):
        """
        Import a single row of spike analysis data
        
        Args:
            row_data (dict): Dictionary containing the mapped data fields
            import_options (dict): Import configuration options
            
        Returns:
            bool: True if import was successful, False otherwise
        """
        try:
            # Extract essential fields
            recording_path = row_data.get('Recording Path')
            if not recording_path:
                return False
            
            # Extract cell name (try multiple strategies)
            cell_name = row_data.get('Cell Name')
            if not cell_name:
                # Try to extract from file path
                filename = os.path.basename(recording_path)
                cell_name = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
            
            # Extract protocol name
            protocol_name = row_data.get('Protocol')
            if not protocol_name:
                protocol_name = import_options.get('default_protocol', 'Unknown')
            
            # Create cell if it doesn't exist
            if import_options.get('create_cells', True):
                if cell_name not in self.cellindex:
                    self.addEntry(cell_name)
            
            # Check if cell exists
            if cell_name not in self.cellindex:
                return False
            
            # Create or update protocol
            if import_options.get('create_protocols', True):
                if protocol_name not in self.cellindex[cell_name]:
                    self.addProtocol(cell_name, protocol_name, path=recording_path)
                elif import_options.get('update_existing', True):
                    # Update existing protocol with new recording
                    current_path = self.cellindex[cell_name].get(protocol_name)
                    if isinstance(current_path, list):
                        if recording_path not in current_path:
                            current_path.append(recording_path)
                    else:
                        if current_path != recording_path:
                            self.cellindex[cell_name][protocol_name] = [current_path, recording_path]
            
            # Add spike analysis metadata
            spike_data = {}
            field_mappings = {
                'Spike Count': 'spike_count',
                'Spike Rate': 'spike_rate',
                'ISI Mean': 'isi_mean',
                'ISI CV': 'isi_cv',
                'First Spike Latency': 'first_spike_latency',
                'Sweep Number': 'sweep_number'
            }
            
            for field_name, data_key in field_mappings.items():
                if field_name in row_data and row_data[field_name] is not None:
                    try:
                        # Convert to appropriate numeric type
                        value = row_data[field_name]
                        if pd.notna(value):
                            spike_data[data_key] = float(value) if '.' in str(value) else int(value)
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        pass
            
            # Store spike analysis data
            if spike_data:
                if 'spike_analysis' not in self.cellindex[cell_name]:
                    self.cellindex[cell_name]['spike_analysis'] = {}
                
                if protocol_name not in self.cellindex[cell_name]['spike_analysis']:
                    self.cellindex[cell_name]['spike_analysis'][protocol_name] = {}
                
                # Use sweep number as key if available, otherwise use recording path
                analysis_key = spike_data.get('sweep_number', recording_path)
                self.cellindex[cell_name]['spike_analysis'][protocol_name][analysis_key] = spike_data
            
            return True
            
        except Exception as e:
            print(f"Error importing spike data row: {e}")
            return False
        

    def from_xlsx(self, file_path, filename_cols, filepath_cols, protocol_col):
        #read the 
        pass

    def from_csv(self, file_path, filename_cols, filepath_cols, protocol_col):
        """
        Create database from CSV file with arbitrary structure
        """
        import pandas as pd
        df = pd.read_csv(file_path)
        return self.from_dataframe(df, filename_cols, filepath_cols, protocol_col)

    def from_dataframe(self, df, filename_cols=None, filepath_cols=None, protocol_file_col=None, 
                      cell_id_col='CELL_ID', metadata_cols=None, skip_empty=True):
            """
            Create database from arbitrary dataframe structure where:
            - Each row represents a cell/recording session
            - Columns represent protocols with file IDs or paths
            
            Parameters:
            -----------
            df : pandas.DataFrame
                Input dataframe with cell data
            filename_cols : list, optional
                List of column names that contain filenames/file IDs for protocols
                If None, will auto-detect columns that contain file-like data
            filepath_cols : dict, optional  
                Dictionary mapping protocol names to file path columns
                Format: {'protocol_name': 'filepath_column_name'}
            protocol_file_col : str, optional
                Column name containing base file paths (if different from filename columns)
            cell_id_col : str, default 'CELL_ID'
                Column name containing cell identifiers
            metadata_cols : list, optional
                List of column names to store as cell metadata (e.g., DATE, drug, NOTE)
            skip_empty : bool, default True
                Whether to skip empty cells in protocol columns
                
            Returns:
            --------
            bool : True if successful, False otherwise
                
            Example usage:
            -------------
            # Auto-detect protocol columns
            db.from_dataframe(df, cell_id_col='CELL_ID')
            
            # Specify specific protocol columns  
            db.from_dataframe(df, 
                            filename_cols=['IC1', 'CTRL_PULSE', 'NET_PULSE'],
                            cell_id_col='CELL_ID',
                            metadata_cols=['DATE', 'drug', 'NOTE'])
            """
            #try:
            logger.info("Creating database from arbitrary dataframe structure")
            
            # Validate input dataframe
            if df.empty:
                logger.error("Input dataframe is empty")
                return False
                
            if cell_id_col not in df.columns:
                logger.error(f"Cell ID column '{cell_id_col}' not found in dataframe")
                return False
            
            # Clean the dataframe - remove rows where cell_id is empty/null or contains only underscores
            df_clean = df[df[cell_id_col].notna() & (df[cell_id_col] != '') & (df[cell_id_col] != '_')].copy()
            
            if df_clean.empty:
                logger.error("No valid cell data found after cleaning")
                return False
                
            logger.info(f"Processing {len(df_clean)} cells from dataframe")
            
            # Auto-detect filename columns if not provided
            if filename_cols is None:
                filename_cols = []
                # Look for columns that contain file-like data (numbers, letters, underscores)
                # Exclude obvious metadata columns
                exclude_cols = [cell_id_col] + (metadata_cols or [])
                exclude_patterns = ['DATE', 'drug', 'NOTE', 'UNIQUE_ID', 'Burst', 'YES', 'NO']
                
                for col in df_clean.columns:
                    if col in exclude_cols:
                        continue
                    if any(pattern.lower() in col.upper() for pattern in exclude_patterns):
                        continue
                    
                    # Check if column contains file-like data (has non-empty string values)
                    non_empty_values = df_clean[col].dropna()
                    if len(non_empty_values) > 0:
                        # Check if values look like file IDs (contain numbers/letters/underscores)
                        sample_vals = non_empty_values.head(10).astype(str)
                        if any(val for val in sample_vals if 
                               val not in ['', 'nan', 'None'] and 
                               any(c.isalnum() or c == '_' for c in val)):
                            filename_cols.append(col)
                            
                logger.info(f"Auto-detected {len(filename_cols)} protocol columns: {filename_cols[:10]}{'...' if len(filename_cols) > 10 else ''}")
            
            # Initialize database if needed
            if self.cellindex.empty:
                self.cellindex = pd.DataFrame()
            
            # Process each cell
            processed_cells = 0
            for idx, row in df_clean.iterrows():
                cell_name = str(row[cell_id_col]) if cell_id_col in row else None
                if cell_id_col in row and pd.isna(cell_name):
                    logger.warning(f"Cell ID is NaN for row {idx}")
                    #making a cell ID?
                    cell_name = f"Cell_{idx}"

                if not cell_name or cell_name in ['nan', 'None', '_']:
                    continue
                    
                logger.debug(f"Processing cell: {cell_name}")
                
                # Add cell to database if it doesn't exist
                if cell_name not in self.cellindex.index:
                    self.addEntry(cell_name)
                    
                # Add metadata columns
                if metadata_cols:
                    for meta_col in metadata_cols:
                        if meta_col in df_clean.columns and pd.notna(row[meta_col]):
                            # Add metadata column to cellindex if it doesn't exist
                            if meta_col not in self.cellindex.columns:
                                self.cellindex[meta_col] = None
                            self.cellindex.loc[cell_name, meta_col] = row[meta_col]
                
                # Process protocol columns
                protocols_added = 0
                for protocol_col in filename_cols:
                    if protocol_col in row.index and pd.notna(row[protocol_col]):
                        file_value = str(row[protocol_col]).strip()
                        
                        # Skip empty values if requested
                        if skip_empty and (not file_value or file_value in ['', 'nan', 'None']):
                            continue
                            
                        # Determine file path
                        file_path = file_value
                        
                        # Check if there's a specific filepath column for this protocol
                        if filepath_cols and protocol_col in filepath_cols:
                            filepath_col = filepath_cols[protocol_col]
                            if filepath_col in row.index and pd.notna(row[filepath_col]):
                                file_path = str(row[filepath_col]).strip()
                        elif protocol_file_col and pd.notna(row[protocol_file_col]):
                            # Use base file path if provided
                            base_path = str(row[protocol_file_col]).strip()
                            if base_path:
                                file_path = os.path.join(base_path, file_value)
                        
                        # Add protocol to experimental structure and cell index
                        try:
                            self.addProtocol(cell_name, protocol_col, path=file_path)
                            protocols_added += 1
                            logger.debug(f"Added protocol '{protocol_col}' with path '{file_path}' to cell '{cell_name}'")
                        except Exception as e:
                            logger.warning(f"Failed to add protocol '{protocol_col}' to cell '{cell_name}': {e}")
                            continue
                
                if protocols_added > 0:
                    processed_cells += 1
                    logger.debug(f"Added {protocols_added} protocols to cell {cell_name}")
            
            logger.info(f"Successfully processed {processed_cells} cells with protocol data")
            
            # Update experimental structure
            logger.info(f"Database now contains {len(self.cellindex)} cells with {len(self.cellindex.columns)} total columns")
            
            return True
            
            # except Exception as e:
            #     logger.error(f"Error creating database from dataframe: {e}")
            #     import traceback
            #     traceback.print_exc()
            #     return False
        