########
# This script will run the web vizualization of the output,
# essentially we will call the backend script and then server the data
# to the frontend.
# There are 2 options for the backend:
# 1. static: This will generate the plots and then package them into the html file (recommended for GitHub Pages)
# 2. dynamic: This will generate the plots and then save them separately, and then load them into the html file via webserver (⚠️ EXPERIMENTAL)

import os
import argparse
from ..database import build_database
from pyAPisolation.utils import arg_wrap
from . import ephysDatabaseViewer, webVizConfig
from http.server import HTTPServer, CGIHTTPRequestHandler

def run_web_viz(dir_path=None, database_file=None, config=None, backend='static'):
    if dir_path is None:
        dir_path = os.getcwd()
    if database_file is None:
        # Auto-generate database if not provided
        output_dir = os.path.join(dir_path, 'output')
        os.makedirs(output_dir, exist_ok=True)
        database_file = os.path.join(output_dir, 'database.csv')
        build_database.main(['--folder', dir_path, '--outfile', database_file])

    if config is None:
        config = webVizConfig.webVizConfig()
    elif isinstance(config, str):
        config = webVizConfig.webVizConfig(file=config)
    elif isinstance(config, dict):
        config = webVizConfig.webVizConfig(**config)

    if backend == 'static' or backend == 'dynamic':
        ephysDatabaseViewer.main(database_file=database_file, config=config, static=(backend=='static'))
        return
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'static' or 'dynamic'.")


if __name__ == '__main__':
    # make an argparse to parse the command line arguments. command line args should be the path to the data folder, or
    # pregenerated dataframes
    parser = argparse.ArgumentParser(
        description='Web app for visualizing electrophysiology data')
    parser.add_argument('--backend', type=str, default='static',
                        help='Backend to use: "static" (recommended, for GitHub Pages) or "dynamic" (⚠️ experimental, requires server)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (.json or .yaml). See webviz_config.yaml for example.')
    parser.add_argument('--production', action='store_true',
                        help='Run dynamic mode with gunicorn for production deployment (implies --backend dynamic)')
    
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
    
    # Load config if provided
    config_obj = None
    if args.config:
        config_obj = webVizConfig(file=args.config)
    
    # Handle production mode
    if args.production:
        if args.backend != 'dynamic':
            print("⚠️ --production flag implies dynamic backend, switching to dynamic mode")
            args.backend = 'dynamic'
        print("\n" + "="*60)
        print("PRODUCTION DEPLOYMENT MODE")
        print("="*60)
        print("\nFor production use, deploy with gunicorn:")
        print("  gunicorn -w 4 -b 0.0.0.0:8000 pyAPisolation.webViz.flaskApp:app")
        print("\nOr with more workers and custom config:")
        print("  gunicorn -w 8 -b 0.0.0.0:8000 --timeout 120 pyAPisolation.webViz.flaskApp:app")
        print("\nFirst generate the HTML with:")
        app = run_web_viz(data_folder, database_file=data_df, config=config_obj, backend=args.backend)
        print("\nNow start the gunicorn server in the output directory.")
        

    app = run_web_viz(data_folder, database_file=data_df, config=config_obj, backend=args.backend)

    app.app.run(host='0.0.0.0', debug=False)