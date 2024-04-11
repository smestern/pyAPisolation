import numpy as np
import xml.etree.ElementTree as ET
import lxml
import os
import glob
import shutil
import pandas as pd
import time
import copy
#get the path of this script
path = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(path, 'prism_template2.pzfx')
def register_all_namespaces(filename):
    namespaces = dict([node for _, node in ET.iterparse(filename, events=['start-ns'])])
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])
register_all_namespaces(template_path)


group_str_template = {
"table_def": """<Table ID=\"TABLE_NAME\" XFormat=\"none\" YFormat=\"replicates\" 
Replicates=\"NUM_REPLICATES\" TableType=\"TwoWay\" EVFormat=\"AsteriskAfterNumber\"></Table>""",
"ycolumn": f"<YColumn Width=\"243\" Decimals=\"10\" Subcolumns=\"NUM_REPLICATES\"></YColumn>",
"ycolumn_title": "<Title>COLUMN_TITLE</Title>", 
"subcolumn": "<Subcolumn><Title>SUBCOLUMN_TITLE</Title></Subcolumn>",
"data_point": "<d>DATA_POINT</d>", 
"ycolumn_end": "</YColumn>",
"table_end": "</Table>",
"table_sequence": "<Ref ID=\"TABLE_NAME\"/>",
"row_title_decl":"""<RowTitlesColumn Width=\"125\"><Subcolumn></Subcolumn></RowTitlesColumn>""",
"row_title": "<d>ROW_TITLE</d>",
"subcol_title_decl" : """<SubColumnTitles OwnSet=\"0\"></SubColumnTitles>""",
"subcol_title": """<Subcolumn><d><TextAlign align="Center">SUBCOL_TITLE</TextAlign></d></Subcolumn>"""
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
        self.main_file =self.clear_template_contents()
          #copy.copy(self.template_file)

        pass

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

    def make_group_table(self, group_name, group_values, groupby=None, subgroupcols=None, subgroupby=None, rowgroupby=None):
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
                subgroupcols: list of strs, If list of str, these are assumed to be the col names from the group_values dataframe to be used as subcolumns in
                subgroupby: str, the column to subgroup by.
                rowgroupby: str, the column to rowgroup by.
        """

        group_values_no_groupby = group_values.drop(groupby, axis=1)
        group_values_no_groupby = group_values_no_groupby.drop(subgroupby, axis=1) if subgroupby is not None else group_values_no_groupby
        group_values_no_groupby = group_values_no_groupby.drop(rowgroupby, axis=1) if rowgroupby is not None else group_values_no_groupby
        
        #get the number of ycolumns by the number of unique groups
        if groupby is None:
            num_ycolumns = 1
            name_ycolumns = group_name
        else:
            #make the groupby column a string
            group_values[groupby] = group_values[groupby].astype(str)
            num_ycolumns = group_values[groupby].unique().shape[0]
            name_ycolumns = group_values[groupby].unique().astype(str)
            idxs_per_group = group_values.groupby(groupby).groups


        #get the num subcolumns by the number of non-groupby columns
        subgroup_method = None
        if subgroupby is not None and subgroupcols is not None:
            raise ValueError('subgroupby and subgroupcols cannot both be specified')
        elif subgroupby is None:
            #if there is no subgroupby, then we look at subgroupcols
            if subgroupcols is not None: 
                num_subcolumns = len(subgroupcols) #this is the number of subcolumns
                name_subcolumns = subgroupcols #this is the name of the subcolumns
                subgroup_method = 'subgroupcols' #this is the method of subgrouping
            else:
                num_subcolumns = group_values_no_groupby.columns.shape[0]
                name_subcolumns = group_values_no_groupby.columns.astype(str)
                subgroup_method = 'subgroupcols'
        elif isinstance(subgroupby, str):
            #get the number of unique subgroups
            #make the subgroupby column a string
            group_values[subgroupby] = group_values[subgroupby].astype(str)
            num_subcolumns = group_values[subgroupby].unique().shape[0] #this is the number of subcolumns
            name_subcolumns = group_values[subgroupby].unique().astype(str) #this is the name of the subcolumns
            idxs_per_group = group_values.groupby([groupby, subgroupby]).groups #this is the index of the subcolumns
            subgroup_method = 'subgroupby' #this is the method of subgrouping
        else:
            raise ValueError('subgroupby must be None, a string')
        
        #get the number of rows by the number of unique rowgroups
        if rowgroupby is None:
            num_rows = -1
        else:
            group_values[rowgroupby] = group_values[rowgroupby].astype(str)
            num_rows = group_values[rowgroupby].unique().shape[0]
            name_rows = group_values[rowgroupby].unique().astype(str)
            idxs_per_group = group_values.groupby([groupby, subgroupby, rowgroupby]).groups

            

           
        #make the table definition
        table_def = group_str_template['table_def'].replace('TABLE_NAME', group_name).replace('NUM_REPLICATES', str(num_subcolumns))
        #table_def += group_str_template['table_end']
        new_table = ET.fromstring(table_def, )
        #add the title to the table
        title = group_str_template['ycolumn_title'].replace('COLUMN_TITLE', group_name)
        title = ET.fromstring(title)
        new_table.append(title)

        if rowgroupby is not None:
            #declare the row titles
            row_title_list = ET.fromstring(group_str_template['row_title_decl'])
            #add the row titles
            [row_title_list.append(ET.fromstring(group_str_template['row_title'].replace('ROW_TITLE', row))) for row in name_rows]
            #add the row titles
            new_table.append(row_title_list)
        #otherwise rows are just sequential

        #declare the subcolumn titles
        if subgroup_method == 'subgroupcols':
            subcol_title_list = ET.fromstring(group_str_template['subcol_title_decl'])
            [subcol_title_list.append(ET.fromstring(group_str_template['subcol_title'].replace('SUBCOL_TITLE', subcol))) for subcol in name_subcolumns]
            new_table.append(subcol_title_list)
        elif subgroup_method == 'subgroupby':
            subcol_title_list = ET.fromstring(group_str_template['subcol_title_decl'])
            [subcol_title_list.append(ET.fromstring(group_str_template['subcol_title'].replace('SUBCOL_TITLE', subcol))) for subcol in name_subcolumns]
            new_table.append(subcol_title_list)
        else:
            pass

        #make the ycolumn definition
        for ycol in np.arange(num_ycolumns):
            #make the ycolumn
            ycolumn = group_str_template['ycolumn'].replace('NUM_REPLICATES', str(num_subcolumns))
            ycolumn = ET.fromstring(ycolumn)
            #add the title as a child
            title = group_str_template['ycolumn_title'].replace('COLUMN_TITLE', name_ycolumns[ycol].astype(str))
            title = ET.fromstring(title)
            ycolumn.append(title)
            
            #add the subcolumns as children
            for suby in name_subcolumns:
                subcolumn = group_str_template['subcolumn'].replace('SUBCOLUMN_TITLE', suby)
                subcolumn = ET.fromstring(subcolumn)
                #add the data points as children to the subcolumns
                subcolumn_data = group_values_no_groupby[suby] if subgroup_method == 'subgroupcols' else group_values_no_groupby
                #only where the groupby column matches the current ycolumn
                if subgroup_method == 'subgroupby':
                    subcolumn_data = subcolumn_data.iloc[idxs_per_group[(name_ycolumns[ycol], name_subcolumns[ycol])], 0].to_list()
                elif subgroup_method == 'rowsubgroupby':
                    subcolumn_data = subcolumn_data.iloc[idxs_per_group[(name_ycolumns[ycol], name_subcolumns[ycol], name_rows[ycol])], 0].to_list()
                else:
                    subcolumn_data = subcolumn_data[idxs_per_group[name_ycolumns[ycol]]].to_list()

                #add the data points to the subcolumn, this time as children
                for data_point in subcolumn_data:
                    data_point = group_str_template['data_point'].replace('DATA_POINT', str(data_point))
                    data_point = ET.fromstring(data_point)
                    subcolumn.append(data_point)
                ycolumn.append(subcolumn)
            #add the ycolumn to the table
            new_table.append(ycolumn)
            #close the ycolumn
        #close the table
        #add it to the TableSequence
        self.append_xml_table(new_table, group_name)

        return self.main_file
    
    def append_xml_table(self, new_table, name):
        table_sequence = self.main_file.findall('{http://graphpad.com/prism/Prism.htm}TableSequence', ns)[0]
        table_sequence.append(ET.fromstring(group_str_template['table_sequence'].replace('TABLE_NAME', name)))
        #add the new table to the template, add it before the <template> tag
        self.main_file.getroot().append(new_table)
        return self.main_file
    
    def write(self, file_path, xml_declaration=True, encoding='utf-8', method="xml", default_namespace=""):
        self.main_file.write(file_path, xml_declaration=xml_declaration, encoding=encoding, method=method, default_namespace=default_namespace)
        return None

    def save(self, file_path, *args, **kwargs):
        self.write(file_path, *args, **kwargs)
        return None
    

if __name__=="__main__":
    np.random.seed(42)
    #make some random data to test with
    x = np.random.randint(0, 9, size=(100,2))
    labels = np.random.choice(['Ycol_a', 'Ycol_b'], 100)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    #too double check the groupby, add 100 to all Ycol_a values in rnd1
    df.loc[df['labels'] == 'Ycol_a', 'rnd1'] += 100
    #make all Ycol_b values in rnd2 negative
    df.loc[df['labels'] == 'Ycol_b', 'rnd2'] *= -1
    #pass to make
    file = PrismFile()
    out = file.make_group_table('rnd21', df, groupby='labels')
    #makemore
    x = np.random.rand(100, 2)
    labels = np.random.choice(['Ycol_a', 'Ycol_b'], 100)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    labels2 = np.random.choice(['sub1', 'sub2'], 100)
    df['labels2'] = labels2
    #pass to make
    out = file.make_group_table('rnd22', df, groupby='labels', subgroupby='labels2') 

    #try a 3way group
    x = np.random.rand(50, 2)
    labels = np.random.choice(['Ycol_a', 'Ycol_b'], 50)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    labels2 = np.random.choice(['sub1', 'sub2'], 50)
    df['labels2'] = labels2
    labels3 = np.random.choice(['row1', 'row2'], 50)
    df['labels3'] = labels3
    #pass to make
    out = file.make_group_table('rnd23', df, groupby='labels', subgroupby='labels2', rowgroupby='labels3')

    #try just rowgroupby
    x = np.random.rand(50, 2)
    labels = np.random.choice(['Ycol_a', 'Ycol_b'], 50)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    labels3 = np.random.choice(['row1', 'row2'], 50)
    df['labels3'] = labels3
    #pass to make
    out = file.make_group_table('rnd24', df, rowgroupby='labels3')



    #try to write it
    file.write('test.pzfx')