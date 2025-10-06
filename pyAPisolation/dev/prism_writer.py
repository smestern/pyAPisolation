import numpy as np
import xml.etree.ElementTree as ET

import os

import shutil
import pandas as pd
import time
import copy
import logging

#set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#get the path of this script
path = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(path, 'prism_template2.pzfx')
def register_all_namespaces(filename):
    namespaces = dict([node for _, node in ET.iterparse(filename, events=['start-ns'])])
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])
register_all_namespaces(template_path)


group_str_template = {
"table_def": "<Table ID=\"TABLE_NAME\" XFormat=\"none\" YFormat=\"replicates\" Replicates=\"NUM_REPLICATES\" TableType=\"TwoWay\" EVFormat=\"AsteriskAfterNumber\"></Table> \n ",
"ycolumn": f"<YColumn Width=\"243\" Decimals=\"10\" Subcolumns=\"NUM_REPLICATES\"></YColumn> \n ",
"ycolumn_title": "<Title>COLUMN_TITLE</Title> \n ", 
"subcolumn": "<Subcolumn></Subcolumn> \n ",
"data_point": "<d>DATA_POINT</d> \n ", 
"ycolumn_end": "</YColumn> \n ",
"table_end": "</Table> \n ",
"table_sequence": "<Ref ID=\"TABLE_NAME\"/> \n ",
"row_title_decl":"<RowTitlesColumn Width=\"125\"><Subcolumn></Subcolumn></RowTitlesColumn> \n ",
"row_title": "<d>ROW_TITLE</d> \n ",
"subcol_title_decl" : "<SubColumnTitles OwnSet=\"1\"></SubColumnTitles> \n ",
"subcol_title": "<d><TextAlign align=\"Center\">SUBCOL_TITLE</TextAlign></d> \n "
}



def backup_prism_file(file_path):
    if os.path.exists(file_path):
        shutil.copy(file_path, file_path + f'.backup{time.time()}')
    else:
        raise FileNotFoundError(f'File {file_path} does not exist')
    return None

def load_prism_file(file_path, backup=True):
    if backup:
        backup_prism_file(file_path)
    with open(file_path, 'r') as f:
        #load the file with lxml / etree, not bs4
        tree = ET.parse(f)
    return tree

#load the template
template_file = load_prism_file(template_path, backup=False)
#add the namespace to the root
ns = {'pz': 'http://www.graphpad.com/prism/Prism.htm'}
#get the table templates
table_templates = template_file.findall('{http://graphpad.com/prism/Prism.htm}Table')
template_dict = {'mv_example': None, 'group_example': None, 'col_example': None, 'xy_example': None}
for table in table_templates:
    #find the title of the table
    title = table.findall('{http://graphpad.com/prism/Prism.htm}Title', ns)[0]
    title = title.text
    #add the table to the template_dict
    template_dict[title] = table

class PrismFile():
    def __init__(self) -> None:

        self.template_file = copy.deepcopy(template_file)
        #clear the contents of the template
        if self.template_file is not None:
            self.main_file =self.clear_template_contents()

        self._internal_table_map = {} #this is a map of table names to their ycolumns, used for quick access


    def clear_template_contents(self):
        #first clear the table sequence
        table_sequence = self.template_file.findall('{http://graphpad.com/prism/Prism.htm}TableSequence', ns)[0]
        #clear all its children
        for child in list(table_sequence.iter())[1:]:
            table_sequence.remove(child)
        #then clear the tables
        tables = self.template_file.findall('{http://graphpad.com/prism/Prism.htm}Table', ns)
        for table in tables:
            self.template_file.getroot().remove(table)
        
        #delete the Template field, this is a binary field that is not needed
        template_field = self.template_file.findall('{http://graphpad.com/prism/Prism.htm}Template', ns)[0]
        self.template_file.getroot().remove(template_field)

        return self.template_file

    def make_group_table(self, group_name, group_values, groupby=None, cols=None, subgroupcols=None,
                          subgroupby=None, rowgroupcols=None, rowgroupby=None, append=True):
        """ Create a prism "grouped" table from a pandas dataframe. This table has the following structure:
            | Groupby1     | Groupby2    | Groupby3    | Groupby4    | Groupby5    |
            | sub1 |  sub2 | sub1 | sub2 | sub1 | sub2 | sub1 | sub2 | sub1 | sub2 |
        row1|
        row2|
            Essentially we have three ways of grouping the data, by the groupby column, by the subgroupby column, and by the rowgroupby column.
            The groupby column is the main column that the data is grouped by, the subgroupby column is the subcolumn that the data is grouped by, and the rowgroupby column is the row that the data is grouped by.
            in the case of the rowgroupby column, if None, then the data is placed sequentially in the table.
            Takes:
                group_name: str, the name of the group
                group_values: pd.DataFrame, the data to be grouped
                groupby: str, the column to group by
                cols: list of strs, the columns to be used as 'data', if None, then all columns are used. Does not need to include subgroupby or rowgroupby columns
                subgroupcols: list of strs, If list of str, these are assumed to be the col names from the group_values dataframe to be used as subcolumns in the table
                subgroupby: str, the column to subgroup by.
                rowgroupcols: list of strs, If list of str, these are assumed to be the col names from the group_values dataframe to be used as rowgroups in the table
                rowgroupby: str, the column to rowgroup by.
        """

        #first drop the groupby column
        group_values_no_groupby = group_values.drop(groupby, axis=1) if groupby is not None else group_values
        group_values_no_groupby = group_values_no_groupby.drop(subgroupby, axis=1) if subgroupby is not None else group_values_no_groupby
        group_values_no_groupby = group_values_no_groupby.drop(rowgroupby, axis=1) if rowgroupby is not None else group_values_no_groupby
        
        #easiest way is to brute force ravel the data, the data is a 1d array of the data points, then seperate columns
        #grab the data columns 
        logging.info(f"Grouping data by {groupby}, {subgroupby}, {rowgroupby}")
        data_cols = [x for x in (cols, subgroupcols, rowgroupcols) if x is not None]
        if len(data_cols) == 0:
            #the datacolumns will be all the columns that are not groupby, subgroupby, or rowgroupby
            data_cols = group_values_no_groupby.columns
        else:
            data_cols = np.hstack(data_cols)
        raveled_data = []
        logging.info(f"Raveling data...")
        for i, row in group_values.iterrows():
            for col in data_cols:
                data_point = row[col]
                temp_dict = {'row': i, 'col': col, 'data_point': data_point}
                #add in the remaining columns
                for col in group_values.columns:
                    if col not in data_cols:
                        temp_dict[col] = row[col]
                raveled_data.append(temp_dict)
        raveled_data = pd.DataFrame(raveled_data)
        raveled_data_no_groupby = raveled_data.drop(groupby, axis=1) if groupby is not None else raveled_data
        raveled_data_no_groupby = raveled_data_no_groupby.drop(subgroupby, axis=1) if subgroupby is not None else raveled_data_no_groupby
        raveled_data_no_groupby = raveled_data_no_groupby.drop(rowgroupby, axis=1) if rowgroupby is not None else raveled_data_no_groupby
        logging.info(f"Data raveled, shape: {raveled_data.shape}")

        #get the number of ycolumns by the number of unique groups
        if groupby is None:
            num_ycolumns = 1
            name_ycolumns = group_name
        else:
            #groupby should be a string, if its a one element list, then just take the element
            if isinstance(groupby, list):
                if len(groupby) == 1:
                    groupby = groupby[0]
                else:
                    raise ValueError('groupby must be a string or a one element list')
            #make the groupby column a string
            raveled_data[groupby] = raveled_data[groupby].astype(str)
            num_ycolumns = raveled_data[groupby].unique().shape[0]
            name_ycolumns = raveled_data[groupby].unique().astype(str)
            idxs_per_group = raveled_data.groupby(groupby).groups
            logging.info(f"Grouping main columns by labels: {groupby}")
            logging.info(f'num_ycolumns: {num_ycolumns}, name_ycolumns: {name_ycolumns}')


        #get the num subcolumns by the number of non-groupby columns
        if subgroupby is not None and subgroupcols is not None:
            raise ValueError('subgroupby and subgroupcols cannot both be specified')
        elif subgroupby is None:
            #if there is no subgroupby, then we look at subgroupcols
            if subgroupcols is not None: 
                
                num_subcolumns = len(subgroupcols) #this is the number of subcolumns
                name_subcolumns = subgroupcols #this is the name of the subcolumns
                subgroupby_func = "col"
                logging.info(f"Grouping subcolumns by columns: {subgroupcols}")
                logging.info(f'num_subcolumns: {num_subcolumns}, name_subcolumns: {name_subcolumns}')
            elif rowgroupcols is not None:
                #if there is no subgroupby, then we look at rowgroupcols
                #if these are present, then we likely be assigning the subcolumns sequentially
                logging.info(f"Rowgroupcols present, subcolumns will be assigned sequentially")
                name_subcolumns = group_values_no_groupby.index #this is the name of the subcolumns
                num_subcolumns = len(group_values_no_groupby.index) #this is the number of subcolumns
                subgroupby_func = 'row'
            else: #if there is no subgroupby or subgroupcols, then we just passed in the data
                name_subcolumns = raveled_data_no_groupby['col'].unique() #this is the name of the subcolumns
                num_subcolumns = len(name_subcolumns) #this is the number of subcolumns
                subgroupby_func = "col"
                logging.info(f"Grouping subcolumns sequentially")
                logging.info(f'num_subcolumns: {num_subcolumns}, name_subcolumns: {name_subcolumns}')
        elif isinstance(subgroupby, str):
            #get the number of unique subgroups
            #make the subgroupby column a string
            raveled_data[subgroupby] = raveled_data[subgroupby].astype(str)
            num_subcolumns = raveled_data[subgroupby].unique().shape[0] #this is the number of subcolumns
            name_subcolumns = raveled_data[subgroupby].unique().astype(str) #this is the name of the subcolumns
            subgroupby_func = subgroupby #this is the function to group by
            logging.info(f"Grouping subcolumns by labels: {subgroupby}")
            logging.info(f'num_subcolumns: {num_subcolumns}, name_subcolumns: {name_subcolumns}')
        else:
            raise ValueError('subgroupby must be None, a string')
        

        #get the number of rows by the number of unique rowgroups
        if rowgroupby is not None and rowgroupcols is not None:
            raise ValueError('rowgroupby and rowgroupcols cannot both be specified')
        elif rowgroupby is None:
            if rowgroupcols is not None:
                num_rows = len(rowgroupcols)
                name_rows = rowgroupcols
                rowgroupby_func = 'col'
                logging.info(f"Grouping rows by columns: {rowgroupcols}")
            else:
                rowgroupby_func = 'row'
                name_rows = None #[f"Row {i}" for i in range(raveled_data_no_groupby.shape[0])]
                num_rows = raveled_data_no_groupby.shape[0] #this is the number of rows
                logging.info(f"Rows placed sequentially")
        elif isinstance(rowgroupby, str):
            raveled_data[rowgroupby] = raveled_data[rowgroupby].astype(str)
            num_rows = raveled_data[rowgroupby].unique().shape[0]
            name_rows = raveled_data[rowgroupby].unique().astype(str)
            rowgroupby_func = rowgroupby
            logging.info(f"Grouping rows by labels: {rowgroupby}")
            logging.info(f'num_rows: {num_rows}, name_rows: {name_rows}')
        else:  
            raise ValueError('rowgroupby must be None, a string')

        #get the number of groups
        idxs_per_group = raveled_data.groupby([groupby, subgroupby_func, rowgroupby_func]).groups #this is the index of the subcolumns
        suby_per_group = raveled_data.groupby([groupby, subgroupby_func]).groups #this is for counting the number of subcolumns per group
        
        #warn the user if the subgroupby_func is a label, and rowgroupby_func is a label, then there should be only one data point per group
        if (subgroupby_func != 'row' and subgroupby_func != 'col') and (rowgroupby_func != 'row' and rowgroupby_func != 'col'):
            if np.any([len(x) > 1 for x in idxs_per_group.values()]):
                logging.warning('If subgroupby_func and rowgroupby_func are both labels, then there should be only one data point per group')
            
        #make the table definition
        table_def = group_str_template['table_def'].replace('TABLE_NAME', group_name).replace('NUM_REPLICATES', str(num_subcolumns))
        #table_def += group_str_template['table_end']
        new_table = ET.fromstring(table_def, )
        #add the title to the table
        title = group_str_template['ycolumn_title'].replace('COLUMN_TITLE', group_name)
        title = ET.fromstring(title)
        new_table.append(title)

        if name_rows is not None:
            #declare the row titles
            row_title_list = ET.fromstring(group_str_template['row_title_decl'])
            #add the row titles
            [row_title_list[0].append(ET.fromstring(group_str_template['row_title'].replace('ROW_TITLE', row))) for row in name_rows]
            #add the row titles
            new_table.append(row_title_list)
        #otherwise rows are just sequential

        #declare the subcolumn titles
        if name_subcolumns is not None:
            subcol_title_list = ET.fromstring(group_str_template['subcol_title_decl'])
            #actually we will handle this later
        #otherwise subcolumns are just sequential
        
        
        #nestle the data points into nested dicts
        ycols_map = {} #this is a map of ycolumns to their subcolumns, and subcolumns to their data points
        for ycol, subycol, row in idxs_per_group:
            if ycol not in ycols_map: #if the ycol is not in the ycols_map
                #make the ycolumn
                ycolumn = group_str_template['ycolumn']
                ycolumn = ET.fromstring(ycolumn)
                #add the title as a child   
                title = group_str_template['ycolumn_title'].replace('COLUMN_TITLE', str(ycol))
                title = ET.fromstring(title)
                ycolumn.append(title)
                ycols_map[ycol] = {'object': ycolumn}
                subymap = {}
                ycols_map[ycol]['subymap'] = subymap
            else:
                ycolumn = ycols_map[ycol]['object'] #get the ycolumn
                subymap = ycols_map[ycol]['subymap'] #get the subymap
            
            if f"{subycol}" not in subymap: #if the subcolumn is not in the subcolumn map
                #make the subcolumn
                subcolumn = group_str_template['subcolumn'] #make the subcolumn
                subcolumn = ET.fromstring(subcolumn) 
                subymap[f"{subycol}"] = subcolumn
                subyrowmap = {}
                #fill the subcolumn with blank data points
                for i in range(num_rows):
                    if name_rows is not None:
                        row_name = name_rows[i]
                    else:
                        row_name = i
                    map_row = ET.fromstring(group_str_template['data_point'].replace('DATA_POINT', ''))
                    subcolumn.append(map_row)
                    subyrowmap[f"{subycol}_{row_name}"] = map_row #map the subcolumn to the row
                ycols_map[ycol]['subyrowmap'] = subyrowmap #update the subymap
            else:
                subcolumn = subymap[f"{subycol}"]

            
            #now replace the empty data points with the actual data points
            #get the data points for this ycol, subycol, row
            data_points = raveled_data.loc[idxs_per_group[(ycol, subycol, row)], 'data_point']
            for data_point in data_points:
                data_point = group_str_template['data_point'].replace('DATA_POINT', str(data_point))
                data_point = ET.fromstring(data_point)
                #track down the subcolumn and row
                subyrow_key = f"{subycol}_{row}" #this is the key for the subyrowmap
                if subyrow_key in ycols_map[ycol]['subyrowmap']:
                    #remove the empty data point
                    #ycols_map[ycol]['subyrowmap'][subyrow_key].clear()
                    #append the data point to the subcolumn
                    ycols_map[ycol]['subyrowmap'][subyrow_key].text = data_point.text
                else:
                    logging.warning(f"Subyrow key {subyrow_key} not found in subyrowmap for ycol {ycol}, subycol {subycol}")
                    #if not found, then just append it to the subcolumn
                    subcolumn.append(data_point)

        #now add all the subcolumns to their respective ycolumns
        for ycol in ycols_map.values():
            for subcolumn in ycol['subymap'].values():
                ycol['object'].append(subcolumn)

            #update the replicates in the ycolumn
            ycol['object'].set('Subcolumns', str(len(ycol['subymap'])))

        #add the subcolumn titles
        #to the subcol_title_list if it exists
        if subcol_title_list is not None:
            #spawn a subcolumn title for each subcolumn, up to a max of num_subcolumns
            [subcol_title_list.append(ET.fromstring(group_str_template['subcolumn'])) for i in range(num_subcolumns)]
            # we need to iter throught ycols_map and subymap to get the subcolumn titles
            for ycol in ycols_map.values():
                for i, (key, subcolumn) in enumerate(ycol['subymap'].items()):
                    #get the title
                    title = group_str_template['subcol_title'].replace('SUBCOL_TITLE', str(key))
                    title = ET.fromstring(title)
                    subcol_title_list[i].append(title)
                if i < num_subcolumns - 1:
                    #we need to add empty subcolumns with just </d>
                    for j in range(i+1, num_subcolumns):
                        subcol_title_list[j].append(ET.fromstring("<d></d>"))
            new_table.append(subcol_title_list)

        #once all the ycolumns are made, add them to the table
        for ycolumn in ycols_map.values():
            new_table.append(ycolumn['object'])
        


        #close the table
        #add it to the TableSequence
        if append:
            logging.info(f"Appending table {group_name} to the main file")
            self.append_xml_table(new_table, group_name)
            self._internal_table_map[group_name] = {'ycols': ycols_map, 'row_list': name_rows}
        return new_table
    
    def append_xml_table(self, new_table, name):
        table_sequence = self.main_file.findall('{http://graphpad.com/prism/Prism.htm}TableSequence', ns)[0]
        table_sequence.append(ET.fromstring(group_str_template['table_sequence'].replace('TABLE_NAME', name)))
        #add the new table to the template, add it before the <template> tag
        self.main_file.getroot().append(new_table)
        return self.main_file
    
    def delete_table(self, table_name):
        #find the table in the table sequence
        table_sequence = self.main_file.findall('{http://graphpad.com/prism/Prism.htm}TableSequence', ns)[0]
        table_refs = table_sequence.iter()
        for table_ref in table_refs:
            if 'ID' in table_ref.attrib:
                if table_ref.attrib['ID'] == table_name:
                    logging.info(f"Removing table {table_name} from table sequence")
                    table_sequence.remove(table_ref)
                    break
        #find the table in the main file
        tables = self.main_file.findall('Table', ns)
        for table in tables:
            if 'ID' in table.attrib:
                if table.attrib['ID'] == table_name:
                    logging.info(f"Removing table {table_name} from main file")
                    self.main_file.getroot().remove(table)
                    break

    def get_table(self, table_name):
        """ Get a table from the main file by its name """
        tables = self.main_file.findall('Table', ns)
        for table in tables:
            if 'ID' in table.attrib:
                if table.attrib['ID'] == table_name:
                    return table
        raise ValueError(f"Table {table_name} not found in main file")
    
    def to_dataframe(self, table_name):
        """ Convert a table to a pandas dataframe """

        table = self._internal_table_map.get(table_name, None)
        #get the ycolumns
        ycolumns = table['ycols'].keys()
        row_list = table['row_list'] if 'row_list' in table else None
        data = []
        for ycolumn in ycolumns:
            #get the subcolumns
            subcolumns = table['ycols'][ycolumn]['subymap'].keys()
            for subcolumn in subcolumns:
                #get the data points
                data_points = table['ycols'][ycolumn]['subymap'][subcolumn]
                for i, data_point in enumerate(data_points):
                    if row_list is None or i >= len(row_list):
                        logging.warning(f"Row index {i} out of bounds for row_list in table {table_name}")
                        row_name = f"Row {i}"  # Fallback if row_list is shorter than data points
                    else:
                        row_name = row_list[i]
                    data.append({
                        'ycol': ycolumn,
                        'subycol': subcolumn,
                        'row': row_name,
                        'data_point': data_point.text
                    })

        #convert to dataframe
        df = pd.DataFrame.from_dict(data)
        df = df.pivot_table(index='row', columns=['ycol', 'subycol'], values='data_point')
        return df

    def write(self, file_path, xml_declaration=True, encoding='utf-8', method="xml", default_namespace="", pretty_print=True):
        logging.info(f"Writing to {file_path}")
        if pretty_print:
            indent(self.main_file.getroot())
        self.main_file.write(file_path, xml_declaration=xml_declaration, encoding=encoding, method=method, default_namespace=default_namespace)
        return None

    def save(self, file_path, *args, **kwargs):
        self.write(file_path, *args, **kwargs)
        return None
    
def indent(elem, level=0): #from stackoverflow: https://stackoverflow.com/questions/15418509/python-and-elementtree-writing-one-long-line-of-output
    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j



if __name__=="__main__":
    #todo cli
    pass



