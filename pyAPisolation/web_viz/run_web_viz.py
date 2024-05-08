########
# This script will run the web vizualization of the output,
# essentially we will call the backend script and then server the data
# to the frontend.
# There are 3 options for the backend:
# 1. static: This will generate the plots and then package them into the html file
# 2. dynamic: This will generate the plots and then save them separately, and then load them into the html file via webserver
# 3. dash: This will generate the plots and then serve them via a dash app

import os
from pyAPisolation.utils import arg_wrap
from . import build_database, web_viz_config, dash_folder_app, run_output_to_web
from http.server import HTTPServer, CGIHTTPRequestHandler

def run_web_viz(dir_path=None, database_file=None, config=None, backend='static'):
    if dir_path is None:
        dir_path = os.getcwd()
    if database_file is None:
        build_database.main

        database_file = os.path.join(dir_path, 'output', 'database.csv')

    if config is None:
        config = web_viz_config.web_viz_config()
    elif isinstance(config, str):
        config = web_viz_config.web_viz_config(file=config)
    elif isinstance(config, dict):
        config = web_viz_config.web_viz_config(**config)

    if backend == 'static' or backend == 'dynamic':
        run_output_to_web.main(database_file=database_file, config=config, static=(backend=='static'))
        return
    elif backend == 'dash':
        app = dash_folder_app.run_app(database_file)
        return app


if __name__ == '__main__':
    # make an argparse to parse the command line arguments. command line args should be the path to the data folder, or
    # pregenerated dataframes
    parser = argparse.ArgumentParser(
        description='web app for visualizing data')
    parser.add_argument('--backend', type=str, default='dynamic',
                        help='backend to use for the web app. Options are static, dynamic, dash')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_folder', type=str,
                       help='path to the data folder containing the ABF files')
    group.add_argument('--data_df', type=str,
                       help='path to the pregenerated database')
    group.add_argument('--database_config', type=str,)
    
    #parser = arg_wrap(parser)

    args = parser.parse_args()
    data_folder = args.data_folder
    data_df = args.data_df

    app = run_web_viz(data_folder, database_file=data_df)

    app.app.run(host='0.0.0.0', debug=False)