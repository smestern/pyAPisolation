import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import bs4
import json
import sys
import logging
from scipy.signal import resample, decimate
from pyAPisolation.patch_ml import *
from pyAPisolation.patch_utils import *
from pyAPisolation.featureExtractor import *
from pyAPisolation.loadFile import loadABF
import pyabf
from http.server import HTTPServer, CGIHTTPRequestHandler
import matplotlib.pyplot as plt
import anndata as ad


import shutil
from .flaskApp import tsServer
from .tsDatabaseViewer import tsDatabaseViewer
from .webVizConfig import webVizConfig
from ._scriptTemplates import generate_onload, generate_umap, generate_paracoords, colors, generate_colors

logger = logging.getLogger(__name__)

_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

def gen_table_head_str_(col, soup, dict_args=None):
    tag = soup.new_tag(f"th")
    tag['data-field'] = f"{col}"
    tag['data-sortable'] = f"false"
    tag['data-searchable'] = f"false"
    tag['data-formatter'] = f"valFormatter"
    if dict_args is not None:
        for key, value in dict_args.items():
            tag[key] = value
    tag.string = f"{col}"
    return tag 

def generate_plots(df, static, filename='filename', foldername='foldername.1'):
    ids = df['filename'].to_numpy()
    folders = df['foldername.1'].to_numpy()
    full_y = []   
    for f, fp in zip(ids, folders):
        x, y, z = loadABF(os.path.join(fp,f+'.abf')) 
        y = decimate(y, 4, axis=1)
        x = decimate(x, 4, axis=1)
        idx = np.argmin(np.abs(x-2.5))
        y = y[:, :idx]
        y = np.vstack((x[0, :idx], y))
        fp = create_dir(f"./data/")
        fp += f"{f}.csv"
        if not static:
            np.savetxt(fp, y, delimiter=',', fmt='%.8f')
        elif static:
            
            full_y.append(fp)
    return full_y



class ephysDatabaseViewer(tsDatabaseViewer):
    def __init__(self, database_file, config=None, **kwargs):
        super().__init__(database_file, config, **kwargs)



def main(database_file=None, config=None, static=False):
    """Main function to run the web visualization. This script can be run from the command line or imported as a module.
    The function will generate a web page with the data from the database file. The data will be displayed in a table. Page can be exported static for use with github pages etc.
    takes:
        database_file: str, path to the database file
        config: dict, configuration for the web visualization
        static: bool, if True, the plots will be generated and saved in a separate folder, and then loaded into the html file. If False, the plots will be served via flask
    returns:
        None
    """
    #if the user does not provide a database file, ask for one
    if database_file is None:
        files = filedialog.askopenfilenames(filetypes=(('All files', '*.*'), ('Csv Files', '*.csv'),),
                                            title='Select Input File')
    else: 
        files = [database_file]
    if config is None:
        config = webVizConfig()
    elif isinstance(config, str):
        config = webVizConfig(file=config)
    elif isinstance(config, dict):
        config = webVizConfig(**config)
    
    files = list(files)
    fileList = files

    TEMPLATE = os.path.join(_LOCAL_PATH, "index.html")# if not static else os.path.join(_LOCAL_PATH, "template_static.html")
    with open(TEMPLATE) as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt, 'html.parser')


    
    

    #load the data
    full_dataframe = pd.DataFrame()
    for x in fileList:
        if x.endswith('.csv'):
            temp_df = pd.read_csv(x)
        elif x.endswith('.xlsx'):
            temp_df = pd.read_excel(x)
        elif x.endswith('.h5ad'):
            temp = ad.read_h5ad(x)
            temp_df = temp.to_df()
            temp_df = temp_df.join(temp.obs)
        else:
            print("File type not supported")
            return
        full_dataframe = full_dataframe.append(temp_df)

    ### Preprocess the data, 
    #get the columns that are required by the config    
    full_dataframe = df_select_by_col(full_dataframe, 
                                      [*config.table_vars_rq, *config.table_vars, *config.umap_cols, *config.umap_labels, *config.para_vars, config.primary_label 
                                       if config.primary_label is not None else 'label', config.file_index, config.file_path])
    full_dataframe['ID'] = full_dataframe[config.file_index] if config.file_index in full_dataframe.columns else full_dataframe.index #add an ID column, if it does not exist
            
    #Very specific hardcode here #but we need to make sure dandiset is a string, #this will be removed in the future but for now it is necessary
    if 'dandiset label' in full_dataframe.columns:
        full_dataframe['dandiset label'] = full_dataframe['dandiset label'].astype(str)
        full_dataframe['dandiset label'] = [x.zfill(6) for x in full_dataframe['dandiset label']]
    
    #get the labels from the primary config
    if config.primary_label is not None:
        if config.primary_label not in full_dataframe.columns:
            logger.error(f"Primary label {config.primary_label} not in dataframe")
            labels = None
        else:    
            labels = full_dataframe[config.primary_label].to_numpy()
    else:
        labels = None

    pred_col, labels = extract_features(full_dataframe.select_dtypes(["float32", "float64", "int32", "int64"]), ret_labels=True, labels=labels)
    full_dataframe['label'] = labels
    #check if the umap_cols are present in the dataframe
    if not all([x in full_dataframe.columns for x in config.umap_cols]):
        print("Umap columns not present in the dataframe, generating...")
        #make the umap columns
        data, outlier_idx = preprocess_df(full_dataframe.select_dtypes(["float32", "float64", "int32", "int64"]))
        umap_data = dense_umap(data)
        #insert nan values for the outliers
        for idx in outlier_idx:
            umap_data = np.insert(umap_data, idx, np.nan, axis=0)
        full_dataframe['Umap X'] = umap_data[:, 0]
        full_dataframe['Umap Y'] = umap_data[:, 1]
        print("Umap columns generated")
    else:
        umap_data = full_dataframe[config.umap_cols].to_numpy()
        full_dataframe['Umap X'] = umap_data[:, 0]
        full_dataframe['Umap Y'] = umap_data[:, 1]

    #rename the columns if needed
    full_dataframe = config.process_rename(full_dataframe)

    #populate umap-drop-menu 
    for label in config.umap_labels:
        full_dataframe[label] = full_dataframe[label]#.astype(str)
        umap_drop = soup.find('div', {'id': 'umap-drop-menu'})
        temp_div = soup.new_tag('div', attrs={'class': 'form-check'})
        #set the class to form-check
        if label == config.primary_label:
            temp_opt = f"""<input id="{label}" type="radio" class="form-check-input" name="label-select" value="{label}" checked></input>"""
        else:
            temp_opt = f"""<input id="{label}" type="radio" class="form-check-input" name="label-select" value="{label}"></input>"""
        
        temp_div.append(bs4.BeautifulSoup(temp_opt, 'html.parser'))
        temp_opt = f"""<label class="form-check-label" for="{label}">{label}</label>"""
        temp_opt = bs4.BeautifulSoup(temp_opt, 'html.parser')
        temp_div.append(temp_opt)
        #temp_div = temp_div.prettify()
        umap_drop.append(temp_div)

    ## handle plots
    if config.plots_path: #if the user has already pregeneated the plots
        plot_data = [os.path.join(config.plots_path, x+".csv") for x in full_dataframe['ID'].to_numpy()]
    else:
        plot_data = generate_plots(full_dataframe, static=static, filename=config.file_index, foldername=config.file_path)
    
    #Fix foldernames by truncating
    new_names = []
    for name in full_dataframe[config.file_path].to_numpy():
        temp = name#os.path.split(name)[0]
        new_names.append(temp)
    full_dataframe['foldername'] = new_names #add the new foldername column


    def table_prettier(parsed):
        for i, dict_data in enumerate(parsed):
            dict_data['y'] = plot_data[i]
            for key, value in dict_data.items():
                if isinstance(value, np.int64):
                    dict_data[key] = int(value)
                elif isinstance(value, str):
                    dict_data[key] = value
                elif np.isscalar(value):
                    if value < 1 and value > -1:
                        dict_data[key] = round(value, 4)
                    else:
                        dict_data[key] = round(value, 2)

            parsed[i] = dict_data
        return parsed


    #prior to the conversion, we should check if the user is spliting the table
    if config.table_split is not None:
        
        #split the table on the column requested
        unique_splits = full_dataframe[config.table_split].unique()
        json_split_strs = f"var split_strs = \"{config.table_split}\" \n"
        subtables = {}
        for unique in unique_splits:
            subtables[unique] = full_dataframe.loc[full_dataframe[config.table_split]==unique]

    else:
        subtables = {'': full_dataframe}
        
        json_split_strs = f"var split_strs = \"\" \n"


    #convert the dataframe to json
    for title, val in subtables.items():
        temp_json = val.to_json(orient='records')
        parsed = json.loads(temp_json)
        
        if static:
            parsed = table_prettier(parsed)
        #temp_json_str = json.dumps(parsed)
        subtables[title] = parsed

    json_full = "var subtables = " + json.dumps(subtables) + " \n "

    json_df = full_dataframe.to_json(orient='records')
    parsed = json.loads(json_df)
    if static:
        parsed = table_prettier(parsed)
       
    json_str = json.dumps(parsed)

    #open our template, and insert the json data
    if config.table_split is not None:
        json_var = f" var data_tb = subtables['{config.split_default}']"
    else:
        json_var = '  var data_tb = ' + json_str + ' '


    #script
    json_script =  soup.new_tag('script')
    json_script.string = json_split_strs+ json_full + json_var
    #soup.head.append(json_script)

    #unhide our dataset selector if needed
    if config.table_split is not None:
        table_drop = soup.find('div', {'id': 'dataset-drop-menu'})
        table_drop_parent = soup.find('div', {'id': 'dataset-select'})
        table_drop_parent['class'] = "row"
        for label in subtables.keys():
            temp_div = soup.new_tag('div', attrs={'class': 'form-check'})
            #set the class to form-check
            if label == config.split_default:
                temp_opt = f"""<input id="{label}" type="checkbox" class="form-check-input" name="dataset-select" value="{label}" checked></input>"""
            else:
                temp_opt = f"""<input id="{label}" type="checkbox" class="form-check-input" name="dataset-select" value="{label}"></input>"""
            
            temp_div.append(bs4.BeautifulSoup(temp_opt, 'html.parser'))
            temp_opt = f"""<label class="form-check-label" for="{label}">{label}</label>"""
            temp_opt = bs4.BeautifulSoup(temp_opt, 'html.parser')
            temp_div.append(temp_opt)
            #temp_div = temp_div.prettify()
            table_drop.append(temp_div)

    
    #column tags
    table_head= soup.find('tr')
    visible_cols = np.hstack((*config.table_vars_rq, *[x for x in config.table_vars if x in full_dataframe.columns]))
    #make sure these are unique
    visible_cols = np.unique(visible_cols)
    visible_cols = np.setdiff1d(visible_cols, config.file_index)
    #remove file_index if its there
    if config.hidden_table is not None:
        if config.hidden_table is False:
            visible_cols = np.hstack((config.file_index, '_plot', visible_cols, '_plot_fi'))
            #we need to add the hidden columns
            hidden_cols = np.setdiff1d(full_dataframe.columns, visible_cols)
            for col in visible_cols:
                logger.info(f"Adding column {col}")\
                
                #if the column is _plot, add a special tag
                if '_plot' in col:
                    test = soup.new_tag(f"th")
                    test['data-field'] = f"{col}"
                    test['data-formatter'] = f"plotFormatterDummy"
                    test['data-sortable'] = f"false"
                    test['data-searchable'] = f"false"
                    test.string = f""
                elif col == config.file_index:
                    test = gen_table_head_str_(col, soup, dict_args={'data-class': 'file-ID'})
                else:
                    test = gen_table_head_str_(col, soup)
                table_head.append(test)

            #add the hidden columns
            for col in hidden_cols:
                logger.info(f"Adding hidden column {col}")
                test = gen_table_head_str_(col, soup, dict_args={'data-visible': 'false', 'data-searchable': 'false', 'data-sortable': 'true'})
                table_head.append(test)
        else:
            #our actual columns are just then
            _true_visible = np.hstack((config.file_index, '_plot', '_ephys', '_plot_fi'))
            for col in _true_visible:
                if '_plot' in col:
                    test = soup.new_tag(f"th")
                    test['data-field'] = f"{col}"
                    test['data-formatter'] = f"plotFormatterDummy"
                    test['data-sortable'] = f"false"
                    test['data-searchable'] = f"false"
                    test.string = f""
                elif '_ephys' in col:
                    test = gen_table_head_str_(col, soup, dict_args={'data-formatter': 'ephysFormatterDummy', 'data-searchable': 'false', 'data-sortable': 'false'})
                    test.string = f""
                elif col == config.file_index:
                    test = gen_table_head_str_(col, soup, dict_args={'data-class': 'file-ID'})
                else:
                    test = gen_table_head_str_(col, soup)
                table_head.append(test)

            #add the hidden columns
            hidden_cols = np.setdiff1d(full_dataframe.columns, _true_visible)
            for col in hidden_cols:
                logger.info(f"Adding hidden column {col}")
                test = gen_table_head_str_(col, soup, dict_args={'data-visible': 'false', 'data-searchable': 'false', 'data-sortable': 'true'})
                table_head.append(test)

    if not static:
        #replace the template.js import in the html with 
        #template_dyn.js import
        script_tag = soup.find_all('head')[0]
        #add a new script tag onto the end of the list
        new_tag = soup.new_tag('script')
        new_tag['src'] = 'assets/template_dyn.js'
        script_tag.append(new_tag)

    #generate the umap and paracoords scripts
    umap_script = generate_umap('data_tb', [*config.umap_cols, config.umap_labels[0]])
    paracoords_script = generate_paracoords('data_tb', config.para_vars, config.para_var_colors)
    color_script = generate_colors(config.color_schemes)
    #add the onload script to the end of the body
    
    if config.db_title is not None:
        #find the title tag and replace the text
        title_tag = soup.find('h4',{'id':'db_title'})
        title_tag.string = config.db_title

        #also the page title
        soup.title.string = config.db_title

    if config.db_subtitle is not None:
        #find the subtitle tag and replace the text
        subtitle_tag = soup.find('h6',{'id':'db_subtitle'})
        subtitle_tag.string = config.db_subtitle
    if config.db_description is not None:
        #find the description tag and replace the text
        description_tag = soup.find('p',{'id':'db_desc'})
        inner_soup = bs4.BeautifulSoup(config.db_description, 'html.parser')
        #clear the placeholder children
        description_tag.clear()
        #add the new children
        for child in inner_soup.children:
            description_tag.append(child)
    if config.db_links is not None:
        #get the card-links
        card_links = soup.find_all('a', {'class': 'card-link'})
        #clear the children
        [x.clear() for x in card_links]
        description_tag = soup.find('p',{'id':'db_desc'})
        for link in config.db_links:
            #create a new link tag
            new_link = soup.new_tag('a')
            new_link['href'] = config.db_links[link]
            new_link['class'] = 'card-link'
            new_link.string = link
            description_tag.insert_after(new_link)
            #also add a break
            description_tag.insert_after(soup.new_tag('div'))

    #update the embed and paracoords titles
    if config.db_embed_title is not None:
        embed = soup.find('div', {'id': 'embed_card'})
        #find the title tag in the children
        title_tag = embed.find('h4')
        title_tag.string = config.db_embed_title
    if config.db_para_title is not None:
        paracoords = soup.find('div', {'id': 'para_card'})
        #find the title tag in the children
        title_tag = paracoords.find('h4')
        title_tag.string = config.db_para_title




    #== Saving the output ==#
    #export everything to the out_path_folder
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    
    with open(os.path.join(config.output_path, "index.html"), "w") as outf:
        outf.write(str(soup))

    #copy the 'assets' folder
    shutil.copytree(os.path.join(_LOCAL_PATH, "assets"), os.path.join(config.output_path,"assets"), dirs_exist_ok=True)

    #load the template.js file as a string
    template_js = os.path.join(_LOCAL_PATH, "assets/template.js")
    with open(template_js) as inf:
        template_js = inf.read()
        #add the onload script to the template.js file
        template_js = template_js.replace("/* onload */", umap_script + "\n \t" + paracoords_script)
        template_js = template_js.replace("/* data_tb */", json_var)
        template_js = template_js.replace("/* colors */", colors+str(config.color_schemes))

        template_js = template_js.replace("/* ekeys */", "var ekeys = " + json.dumps(visible_cols.tolist()))

        template_js = template_js.replace("/* para_keys */", "var para_keys = "+ json.dumps(config.para_vars))

        template_js = template_js.replace("/* umap_labels */", "var umap_labels = " + json.dumps(config.umap_labels))


    #save the template.js file
    with open(os.path.join(config.output_path, "assets/template.js"), "w") as outf:
        outf.write(template_js)
    #this si instered into the assets/data.js file
    with open(os.path.join(config.output_path, "assets/data.js"), "w") as outf:
        outf.write(json_split_strs+ json_full + json_var)        
    if static:
        print("=== Running Server ===")
        #Create server object listening the port 80
        #change cwd to the output path
        os.chdir(config.output_path)
        server_object = HTTPServer(server_address=('', 8000), RequestHandlerClass=CGIHTTPRequestHandler)
        #spawn a new thread for the server to run on
        server_object.server_activate()
        
        
        #start the server
        server_object.serve_forever()
    else:
        print("=== Running Server ===")
        tsServer(config=config, static=False).run()



if __name__ == '__main__':
    main()