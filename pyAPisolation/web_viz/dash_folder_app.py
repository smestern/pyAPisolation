# os sys imports
from pyAPisolation.loadNWB import loadFile
from pyAPisolation.abf_featureextractor import *
from pyAPisolation.patch_ml import *
import os
import sys
import glob
import argparse

# dash / plotly imports
import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px

# data science imports
import pandas as pd
import numpy as np
import pyabf
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.append('..')
sys.path.append('')
print(os.getcwd())
# pyAPisolation imports

class analysis_fields():
    def __init__(self, **kwargs):
        self.file_index = 'filename'
        self.file_path = "foldername"
        self.table_vars_rq = ['filename', 'foldername']
        self.table_vars = ["rheo", "QC", 'label_c']
        self.para_vars = ["rheo", 'label_c']
        self.umap_labels = ['label']
        self.para_vars_limit = 10
        self.table_vars_limit = 5
        self.__dict__.update(kwargs)

GLOBAL_VARS = analysis_fields()

def _df_select_by_col(df, string_to_find):
    columns = df.columns.values
    out = []
    for col in columns:
        string_found = [x in col for x in string_to_find]
        if np.any(string_found):
            out.append(col)
    return df[out]


class live_data_viz():
    def __init__(self, dir_path=None, database_file=None):
        self.df_raw = None
        self.df = None
        self.para_df = None
        self._run_analysis(dir_path, database_file)

        app = dash.Dash("abf", external_stylesheets=[dbc.themes.ZEPHYR])

        # find pregenerated labels
        self.labels = self._find_label_cols(self.df_raw)
        self.dropdown_options = [{'label': x, 'value': x} for x in self.labels]
        self.dropdown = dcc.Dropdown(
            self.dropdown_options, id='dropdown', value=self.dropdown_options[0]['value'])
        # Umap
        umap_fig = self.gen_umap_plots()
        self.umap_selected = None
        para_fig = self.gen_para_plots()
        self.para_selected = None

        # Make grid ui
        # make the header descibing the app
        header = dbc.Row([
            dbc.Col(html.H1("pyAPisolation", className="text-center"), width=12),
            dbc.Col(html.H3("Live Data Visualization",
                    className="text-center"), width=12),
            dbc.Col(html.H5("Select a file to view",
                    className="text-center"), width=12),
        ])

        col_long = dbc.Col(dbc.Container([dbc.Card([
            dbc.CardHeader("Longitudinal Plot"),
            dbc.CardBody([dcc.Loading(
            id="loading-2", fullscreen=False, type="default",
            children=[html.Div(id='datatable-plot-cell', style={
                "flex-wrap": "nowrap"})])])])
                ]), width=6)
        col_umap = dbc.Col(dbc.Container([
            dbc.Card([dbc.CardHeader("UMAP Plot"),
                      dbc.CardBody([umap_fig], id='umap-cardbody',
                                   style={"min-height": "500px"}),
                      dbc.CardFooter(["Select Color:", self.dropdown]),
                      ])]),
            width=6)
        col_para = dbc.Col([dbc.Card(
            [dbc.CardHeader(dbc.Button(
                            "Paracoords Plot",
                            id="para-collapse-button",
                            className="",
                            color="info",
                            n_clicks=0,
                            )),
             dbc.CardBody([dbc.Collapse(
                 para_fig,
                 id="para-collapse",
                 is_open=True,
             )])])], width=12)
        col_datatable = dbc.Col([dash_table.DataTable(
            id='datatable-row-ids',
            columns=[
                {'name': i, 'id': i, 'deletable': True} for i in self.df.columns
                # omit the id column
                if i != 'id'
            ],
            data=self.df.to_dict('records'),
            filter_action="native",
            sort_action="native",
            sort_mode='multi',
            row_selectable='multi',
            selected_rows=[],
            page_action='native',
            page_current=0,
            page_size=10,
            style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0
            }

        )], id='data-table-col')

        app.layout = html.Div([dbc.Container([
            dbc.Row([header]),
            dbc.Row([col_para]),
            dbc.Row([col_umap, col_long]),
            dbc.Row([col_datatable])
        ]),
            dcc.Interval(
            id='interval-component',
            interval=240*1000,  # in milliseconds
            n_intervals=0
        )
        ])
        self.app = app
        # Define Callbacks
        app.callback(
            Output('loading-2', 'children'),
            Input('datatable-row-ids', 'derived_virtual_row_ids'),
            Input('datatable-row-ids', 'selected_row_ids'),
            Input('datatable-row-ids', 'active_cell'),
            Input('datatable-row-ids', 'data'))(self.update_cell_plot)

        app.callback(Output('datatable-row-ids', 'data'),
                     Input('UMAP-graph', 'selectedData'),
                     Input('UMAP-graph', 'figure'),
                     Input('para-graph', 'restyleData'),
                     Input('para-graph', 'figure')
                     )(self._filter_datatable)

        app.callback(Output('umap-cardbody', 'children'),
                     State('umap-cardbody', 'children'),
                     Input('dropdown', 'value'))(self.alter_umap_color)

        app.callback(
            Output("para-collapse", "is_open"),
            [Input("para-collapse-button", "n_clicks")],
            [State("para-collapse", "is_open")],
        )(self.toggle_collapse)

    def _gen_abf_list(self, dir):
        # os.path.abspath(dir)
        pass

    def _run_analysis(self, dir=None, df=None):
        if df is None:
            if dir is not None:
                _, df, _ = folder_feature_extract(
                    os.path.abspath(dir), default_dict, protocol_name='')
            else:
                _, df, _ = folder_feature_extract(os.path.abspath(
                    '../data/'), default_dict, protocol_name='')
        else:
            df = pd.read_csv(df)
        self.df_raw = copy.deepcopy(df)
        df = _df_select_by_col(self.df_raw, GLOBAL_VARS.table_vars_rq)
        df_optional = _df_select_by_col(
            self.df_raw, GLOBAL_VARS.table_vars).iloc[:, :GLOBAL_VARS.table_vars_limit]
        df = pd.concat([df, df_optional], axis=1)
        df['id'] = df[GLOBAL_VARS.file_index] if GLOBAL_VARS.file_index in df.columns else df.index
        df.set_index('id', inplace=True, drop=False)
        self.df = df
        return [dash_table.DataTable(
                id='datatable-row-ids',
                columns=[
                    {'name': i, 'id': i, 'deletable': True} for i in self.df.columns
                    # omit the id column
                    if i != 'id'
                ],
                data=self.df.to_dict('records'),
                filter_action="native",
                sort_action="native",
                sort_mode='multi',
                row_selectable='multi',
                selected_rows=[],
                page_action='native',
                page_current=0,
                page_size=10,
                style_cell={
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'maxWidth': 0
                }

                )]

    def gen_umap_plots(self, labels=None, label_legend=None, data=None):
        umap_labels_df, labels_df = _df_select_by_col(
            self.df_raw, ['umap']), _df_select_by_col(self.df_raw, GLOBAL_VARS.umap_labels)
        if umap_labels_df.empty is False:

            data = umap_labels_df[['umap X', 'umap Y']].to_numpy()
            if labels is None:
                labels = labels_df[labels_df.columns.values[0]].to_numpy()

        else:
            pre_df, outliers = preprocess_df(self.df)
            data = dense_umap(pre_df)
            labels = cluster_df(pre_df)
        fig = go.Figure(data=go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers',
                                        marker=dict(color=labels), ids=self.df['id'].to_numpy()), )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        fig.layout.autosize = True
        return dcc.Graph(
            id='UMAP-graph',
            figure=fig,
            style={
                "width": "100%",
                "height": "100%"
            },
            config=dict(
                autosizable=True,
                frameMargins=0,
            ),
            responsive=True,

        )

    def alter_umap_color(self, fig, color_col):
        # index into the umap
        # get the color column from our pregenerated df
        labels = self.labels[color_col]['data']
        if self.labels[color_col]['encoder'] is not None:
            label_legend = self.labels[color_col]['encoder'].classes_
        else:
            label_legend = None
        fig = self.gen_umap_plots(labels=labels, label_legend=label_legend)
        return [fig]

    def gen_para_plots(self, *args):

        df = _df_select_by_col(self.df, GLOBAL_VARS.para_vars).iloc[:, :GLOBAL_VARS.para_vars_limit]
        #fig = go.Figure(data=go.Scatter(x=data[:,0], y=data[:,1], mode='markers', marker=dict(color=labels), ids=self.df['id'].to_numpy()))
        fig = px.parallel_coordinates(df,
                                      color_continuous_scale=px.colors.diverging.Tealrose)
        self.para_df = df
        fig.layout.autosize = True
        return dcc.Graph(
            id='para-graph',
            figure=fig,
            style={
                "width": "100%",
                "height": "100%"
            },
            config=dict(
                autosizable=True,
            ),
            responsive=True

        )

    def _filter_datatable_para(self, selectedData, fig):
        def bool_multi_filter(df, kwargs):
            return ' & '.join([f'{key} >= {i[0]} & {key} <= {i[1]}' for key, i in kwargs.items()])

        if selectedData is None:
            out_data = self.df.to_dict('records')
        else:
            #selected_ids = [x['id'] for x in selectedData['points']]
            constraints = {}
            for row in fig['data'][0]['dimensions']:
                try:
                    constraints[row['label']] = row['constraintrange']
                except:
                    constraints[row['label']] = [-9999, 9999]
            out = bool_multi_filter(self.df, constraints)
            filtered_df = self.df.query(out)
            out_data = filtered_df.to_dict('records')
        return out_data

    def _filter_datatable(self, umap_selectedData, umap_fig, para_selectedData, para_fig):
        # this nightmare of a function is to filter the datatable based on the selected points in the umap and parallel coordinates plots
        if umap_selectedData != self.umap_selected:
            self.umap_selected = umap_selectedData
            umap_out_data = self._filter_datatable_umap(
                umap_selectedData, umap_fig)
        else:
            umap_out_data = None
        if para_selectedData != self.para_selected:
            self.para_selected = para_selectedData
            para_out_data = self._filter_datatable_para(
                para_selectedData, para_fig)
        else:
            para_out_data = None
        if umap_selectedData is None and para_selectedData is None:
            combined = self.df.to_dict('records')
        if umap_selectedData is None and para_selectedData is not None:
            combined = para_out_data
        if umap_selectedData is not None and para_selectedData is None:
            combined = umap_out_data
        if umap_selectedData is not None and para_selectedData is not None:
            combined =  para_out_data.update(umap_out_data)
        #combined = None
        return combined

    def _filter_datatable_umap(self, selectedData, fig):
        def bool_multi_filter(df, kwargs):
            return ' & '.join([f'{key} >= {i[0]} & {key} <= {i[1]}' for key, i in kwargs.items()])

        if selectedData is None:
            out_data = self.df.to_dict('records')
        else:
            selected_ids = [x['id'] for x in selectedData['points']]

            filtered_df = self.df.loc[selected_ids]
            out_data = filtered_df.to_dict('records')
        return out_data

    def update_cell_plot(self, row_ids, selected_row_ids, active_cell, data):
        selected_id_set = set(selected_row_ids or [])

        if row_ids is None:
            dff = self.df
            # pandas Series works enough like a list for this to be OK
            row_ids = self.df['id']
        else:
            dff = self.df.loc[row_ids]

        active_row_id = active_cell['row_id'] if active_cell else None
        if active_row_id is None:
            active_row_id = self.df.iloc[0]['id']
        if active_row_id is not None:
            fold = self.df.loc[active_row_id]["foldername"]
            if isinstance(fold, (list, tuple, np.ndarray, pd.Series)):
                fold = fold.to_numpy()[0]
            file_path = os.path.join(fold, active_row_id + ".abf")
            x, y, c = loadABF(file_path)

            cutoff = np.argmin(np.abs(x-2.50))
            x, y = x[:, :cutoff], y[:, :cutoff]
            traces = []
            for sweep_x, sweep_y in zip(x, y):
                traces.append(go.Scattergl(x=sweep_x, y=sweep_y, mode='lines'))
            fig = go.Figure(data=traces, )

            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            fig.update_yaxes(automargin=True)
            fig.update_xaxes(automargin=True)
            fig.layout.autosize = True

            return dcc.Graph(
                id="file_plot",
                figure=fig,
                style={
                    "width": "100%",
                    "height": "100%"
                },
                config=dict(
                    autosizable=True,
                    
                frameMargins=0,
                ),
                responsive=True
            )

    def _find_label_cols(self, df):
        label_cols = {}
        df_labels = _df_select_by_col(df, GLOBAL_VARS.umap_labels)
        if df_labels.empty:
            return label_cols
        for col in df_labels.columns:
            temp_dict = {}
            temp_dict['data'] = df_labels[col].to_numpy()
            # if its a str, encode it
            if isinstance(temp_dict['data'][0], str):
                temp_dict['encoder'] = LabelEncoder()
                temp_dict['data'] = temp_dict['encoder'].fit_transform(
                    temp_dict['data'])
            else:
                temp_dict['encoder'] = None
            label_cols[col] = temp_dict
        return label_cols

    def toggle_collapse(self, n, is_open):
        if n:
            return not is_open
        return is_open


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

    app = live_data_viz(data_folder, database_file=data_df)

    app.app.run(host='0.0.0.0', debug=False)
