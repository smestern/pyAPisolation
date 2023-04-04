########
# This script will run the web vizualization of the output,
# essentially we will call the backend script and then server the data

import os
import sys
import dash_folder_app
import run_output_to_web
import subprocess
import time
import argparse

from http.server import HTTPServer, CGIHTTPRequestHandler

def run_web_viz(dir_path=None, database_file=None, backend='static'):
    if dir_path is None:
        dir_path = os.getcwd()
    if database_file is None:
        database_file = os.path.join(dir_path, 'output', 'database.csv')
    if backend == 'static':
        run_output_to_web.main(database_file=database_file)
        # Create server object listening the port 80
        server_object = HTTPServer(server_address=('', 80), RequestHandlerClass=CGIHTTPRequestHandler)
        # Start the web server
        server_object.serve_forever()
    elif backend == 'dynamic':
        run_output_to_web.run_output_to_web_dynamic(dir_path)
        # Create server object listening the port 80
        server_object = HTTPServer(server_address=('', 80), RequestHandlerClass=CGIHTTPRequestHandler)
        # Start the web server
        server_object.serve_forever()
    elif backend == 'dash':
        run_output_to_web.run_output_to_web_dynamic(dir_path)
        os.chdir("./pyAPisolation/")
        os.chdir("./web_viz")
        sys.path.append('..')
        sys.path.append('')
        app = dash_folder_app.run_app(database_file)
        return app





if __name__ == '__main__':
    # make an argparse to parse the command line arguments. command line args should be the path to the data folder, or
    # pregenerated dataframes
    parser = argparse.ArgumentParser(
        description='web app for visualizing data')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_folder', type=str,
                       help='path to the data folder containing the ABF files')
    group.add_argument('--data_df', type=str,
                       help='path to the pregenerated database')
    args = parser.parse_args()
    data_folder = args.data_folder
    data_df = args.data_df

    app = run_web_viz(data_folder, database_file=data_df)

    app.app.run(host='0.0.0.0', debug=False)