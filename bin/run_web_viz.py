from pyAPisolation.web_viz import run_web_viz
import argparse
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
    group.add_argument('--data_dir', type=str,
                       help='path to the directory containing the pregenerated database')
    
    
    #parser = arg_wrap(parser)

    args = parser.parse_args()
    data_folder = args.data_folder
    data_df = args.data_df
    backend = args.backend

    app = run_web_viz.run_web_viz(data_folder, database_file=data_df, backend=backend)
    if backend == 'dash':
        app.app.run(host='0.0.0.0', debug=False)                 
    