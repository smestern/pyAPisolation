import numpy as np
import bs4
import lxml
import os
import glob
import shutil
import pandas as pd
import time
import copy

group_str_template = {
"table_def": "<Table ID=\"TABLE_NAME\" XFormat=\"none\" YFormat=\"replicates\" Replicates=\"NUM_REPLICATES\" TableType=\"TwoWay\" EVFormat=\"AsteriskAfterNumber\"></Table>",
"ycolumn": f"<YColumn Width=\"243\" Decimals=\"10\" Subcolumns=\"NUM_REPLICATES\"></YColumn>",
"ycolumn_title": "<Title>COLUMN_TITLE</Title>", 
"subcolumn": "<Subcolumn>SUBCOLUMN_TITLE</Subcolumn>",
"data_point": "<d>DATA_POINT</d>", 
"ycolumn_end": "</YColumn>",
"table_end": "</Table>"}


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
        soup = bs4.BeautifulSoup(f, 'lxml')
    return soup

#load the template
template_file = load_prism_file(template_path, backup=False)
#get the table templates
table_templates = template_file.find_all('table')
mv_template = template_file.find_all('title', string='mv_example')[0].parent
group_template = template_file.find_all('title', string='group_example')[0].parent
col_template = template_file.find_all('title', string='col_example')[0].parent
xy_template = template_file.find_all('title', string='xy_example')[0].parent

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
    new_table = bs4.BeautifulSoup(table_def,  "html.parser")
    #make the ycolumn definition
    for ycol in np.arange(num_ycolumns):
        ycolumn = group_str_template['ycolumn'].replace('NUM_REPLICATES', str(num_subcolumns))
        ycolumn = bs4.BeautifulSoup(ycolumn,  "html.parser")
        #add the title as a child
        title = group_str_template['ycolumn_title'].replace('COLUMN_TITLE', name_ycolumns[ycol].astype(str))
        title = bs4.BeautifulSoup(title,  "html.parser")
        ycolumn.append(title)
        #add the subcolumns as children
        for suby in name_subcolumns:
            subcolumn = group_str_template['subcolumn'].replace('SUBCOLUMN_TITLE', suby)
            subcolumn = bs4.BeautifulSoup(subcolumn,  "html.parser")
            #add the data points as children to the subcolumns
            subcolumn_data = group_values_no_groupby[suby]
            #only where the groupby column matches the current ycolumn
            subcolumn_data = subcolumn_data[idxs_per_group[name_ycolumns[ycol]]].to_list()
            for data_point in subcolumn_data:
                data_point = group_str_template['data_point'].replace('DATA_POINT', str(data_point))
                data_point = bs4.BeautifulSoup(data_point,  "html.parser")
                subcolumn.append(data_point)
            ycolumn.append(subcolumn)
        #add the ycolumn to the table
        new_table.append(ycolumn)
        #close the ycolumn
        #ycolumn.append(bs4.BeautifulSoup(group_str_template['ycolumn_end']))
    #close the table
    #new_table.append(bs4.BeautifulSoup(group_str_template['table_end']))
    #add the new table to the template, add it before the <template> tag
    template_file.find('template').insert_before(new_table)

    return template_file

    

if __name__=="__main__":
    #make some random data to test with
    x = np.random.rand(100,2)
    labels = np.random.randint(0,2,100)
    #turn it into a dataframe
    df = pd.DataFrame(x, columns=['rnd1', 'rnd2'])
    df['labels'] = labels
    #pass to make
    out = make_group_table('rnd1', df, groupby='labels')
    #try to write it
    with open('test.pzfx', 'w') as f:
        f.write(str(out))