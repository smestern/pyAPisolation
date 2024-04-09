import numpy as np
import xml.etree.ElementTree as ET
import lxml
import os
import glob
import shutil
import pandas as pd
import time
import copy

ET.register_namespace("", "pz:http://www.graphpad.com/prism/Prism.htm")
group_str_template = {
"table_def": """<Table ID=\"TABLE_NAME\" XFormat=\"none\" YFormat=\"replicates\" 
Replicates=\"NUM_REPLICATES\" TableType=\"TwoWay\" EVFormat=\"AsteriskAfterNumber\"></Table>""",
"ycolumn": f"<YColumn Width=\"243\" Decimals=\"10\" Subcolumns=\"NUM_REPLICATES\"></YColumn>",
"ycolumn_title": "<Title>COLUMN_TITLE</Title>", 
"subcolumn": "<Subcolumn>SUBCOLUMN_TITLE</Subcolumn>",
"data_point": "<d>DATA_POINT</d>", 
"ycolumn_end": "</YColumn>",
"table_end": "</Table>",
"table_sequence": "<Ref ID=\"TABLE_NAME\"/>"}


#get the path of this script
path = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(path, 'prism_template2.pzfx')

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
table_templates = template_file.findall('Table')
template_dict = {'mv_example': None, 'group_example': None, 'col_example': None, 'xy_example': None}
for table in table_templates:
    #find the title of the table
    title = table.findall('pz:Title', ns)[0]
    title = title.text
    #add the table to the template_dict
    template_dict[title] = table

class PrismFile():
    def __init__(self) -> None:
        pass

    def clear_table_contents(table):
        #clear the contents of the table
        for child in table.children:
            if child.name == 'ycolumn':
                child.decompose() 
        return table

    def make_group_table(group_name, group_values, groupby=None):
        #table = clear_table_contents(template)
        group_values_no_groupby = group_values.drop(groupby, axis=1)
        #print(table)
        #get the num subcolumns by the number of non-groupby columns
        if groupby is None:
            num_subcolumns = 1
        else:
            num_subcolumns = group_values_no_groupby.columns.shape[0]
            name_subcolumns = group_values_no_groupby.columns.astype(str)
        
        #get the number of ycolumns by the number of unique groups
        if groupby is None:
            num_ycolumns = 1
            name_ycolumns = group_name
        else:
            num_ycolumns = group_values[groupby].unique().shape[0]
            name_ycolumns = group_values[groupby].unique()
            idxs_per_group = group_values.groupby(groupby).groups
    
                

        #make the table definition
        table_def = group_str_template['table_def'].replace('TABLE_NAME', group_name).replace('NUM_REPLICATES', str(num_ycolumns))
        #table_def += group_str_template['table_end']
        new_table = ET.fromstring(table_def, )
        #make the ycolumn definition
        for ycol in np.arange(num_ycolumns):
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
                subcolumn_data = group_values_no_groupby[suby]
                #only where the groupby column matches the current ycolumn
                subcolumn_data = subcolumn_data[idxs_per_group[name_ycolumns[ycol]]].to_list()
                for data_point in subcolumn_data:
                    data_point = group_str_template['data_point'].replace('DATA_POINT', str(data_point))
                    data_point = ET.fromstring(data_point)
                    subcolumn.append(data_point)
                ycolumn.append(subcolumn)
            #add the ycolumn to the table
            new_table.append(ycolumn)
            #close the ycolumn
            #ycolumn.append(bs4.BeautifulSoup(group_str_template['ycolumn_end']))
        #close the table
        #add it to the TableSequence
        table_sequence = template_file.findall('{http://graphpad.com/prism/Prism.htm}TableSequence', ns)[0]
        table_sequence.append(ET.fromstring(group_str_template['table_sequence'].replace('TABLE_NAME', group_name)))
        #add the new table to the template, add it before the <template> tag
        template_file.getroot().insert(-2, new_table)

        


        return template_file

    

if __name__=="__main__":
    #make some random data to test with
    x = np.random.randint(0, 10, size=(100,2))
    labels = np.random.randint(0,2,100)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    #pass to make
    out = make_group_table('rnd1', df, groupby='labels')
    #makemore
    x = np.random.rand(100, 2)
    labels = np.random.randint(0,2,100)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    #pass to make
    

    #try to write it
    with open('test.pzfx', 'wb') as f:
        out.write("test.pzfx",  xml_declaration=True,encoding='utf-8',
            method="xml")