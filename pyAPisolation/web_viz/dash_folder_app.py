# os sys imports
from pyAPisolation.loadFile import loadFile
from pyAPisolation.feature_extractor import *
from pyAPisolation.patch_ml import *
import os
import sys
import argparse
from . import web_viz_config
# dash / plotly imports
import dash
from dash.dependencies import Input, Output, State
from dash import dash_table, dcc
import dash_ag_grid as dag

import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
# data science imports
import pandas as pd
import numpy as np
import json

sys.path.append('..')
sys.path.append('')
sys.path.append("pyAPisolation/web_viz/")
print(os.getcwd())
#get current file path
file_path = os.path.dirname(os.path.realpath(__file__))
#change working directory to file path
# pyAPisolation imports


GLOBAL_VARS = web_viz_config.web_viz_config()


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

        #either load the database or generate it
        self.df_raw = None
        self.df = None
        self.para_df = None
        datatable = self._run_analysis(dir_path, database_file)

        # make the app
        app = dash.Dash(__name__, external_scripts=[{'src':"https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"}])

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
        header = self._generate_header()
        col_umap = dbc.Col([
            dbc.Card([dbc.CardHeader("UMAP Plot"),
                      dbc.CardBody([umap_fig], id='umap-cardbody',
                                   style={"min-height": "300px"}),
                      dbc.CardFooter(["Select Color:", self.dropdown]),
                      ])],
            width=4)
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
             )])])
             ])  #col for the paracoords plot
        col_datatable = dbc.Col(datatable, id='data-table-col')

        app.layout = dbc.Container([
            dbc.Row([header, dbc.Col([
                dbc.Row([col_umap, col_para]), 
                dbc.Row([col_datatable]),
                html.Div(id='datatable-row-ids-store', style={'display': 'none'}),
            ])]),
            
        ], id='primcont', className="container-lg", style={"padding-right": "0px", "padding-left": "0px", 'max-width': '95%'},
        )
        
        self.app = app
        # Define Callbacks
        # app.callback(
        #     Output('datatable-row-ids-store', 'children'),
        #     State('datatable-row-ids', 'derived_virtual_row_ids'),
        #     State('data-table-col', 'children'),
        #     State('datatable-row-ids', 'selected_row_ids'),
        #     Input('datatable-row-ids', 'active_cell'),
        #     State('datatable-row-ids', 'data'), prevent_intial_call=True)(self.update_cell_plot)

        app.callback(Output('datatable-row-ids', "rowData"),
                     Input('UMAP-graph', 'selectedData'),
                     Input('UMAP-graph', 'figure'),
                     Input('para-graph', 'restyleData'),
                     Input('para-graph', 'figure')
                     )(self._filter_datatable)

        app.callback(Output('umap-cardbody', 'children'),
                     State('umap-cardbody', 'children'),
                     Input('dropdown', 'value'))(self.alter_umap_color)

        #app.callback(
        #     Output("para-collapse", "is_open"),
        #     [Input("para-collapse-button", "n_clicks")],
        #     [State("para-collapse", "is_open")],
        # )(self.toggle_collapse)

        app.callback(
            Output('datatable-row-ids-store', 'children'),
            Input("datatable-row-ids", "cellRendererData")
        )(self.graphClickData)        

    def _gen_abf_list(self, dir):
        # os.path.abspath(dir)
        pass

    ## DATATABLE FUNCTIONS
    def _run_analysis(self, dir=None, df=None):
        if df is None:
            if dir is not None:
                _, df, _ = folder_feature_extract(
                    os.path.abspath(dir), default_dict, protocol_name='')
            else:
                _, df, _ = folder_feature_extract(os.path.abspath(
                    '../data/'), default_dict, protocol_name='')
        else:
            if isinstance(df, pd.DataFrame):
                pass
            else:
                df = pd.read_csv(df) if df.endswith('.csv') else pd.read_excel(df)
        self.df_raw = copy.deepcopy(df)
        df = _df_select_by_col(self.df_raw, GLOBAL_VARS.table_vars_rq)
         #add in a column for dropdown:
        

        df_optional = _df_select_by_col(
            self.df_raw, GLOBAL_VARS.table_vars).iloc[:, :GLOBAL_VARS.table_vars_limit]
        df = pd.concat([df, df_optional], axis=1)

        df['id'] = df[GLOBAL_VARS.file_index] if GLOBAL_VARS.file_index in df.columns else df.index
        df.set_index('id', inplace=True, drop=False)
        self.df = df
        return self.gen_datatable(df)

    def add_show_more(self, df):
        #df will already be in records format
        #$for row in df:
        #    row['show_more'] = '+'
        return df

    def gen_datatable(self, df):
        col_defs = []
        for col in ['show_more', *df.columns.values]:
            if col != 'id':
                if col =='show_more':
                    col_defs.append({"field": col, "cellRenderer": "expandRow"})
                else:
                    col_defs.append({"field": col,})
        return [dag.AgGrid(
            id='datatable-row-ids',
            columnDefs=col_defs,
            rowData=self.add_show_more(self.df.to_dict('records')),
            dashGridOptions={'pagination':True, 'onRowSelected': 'expandRow',"isFullWidthRow": {"function": "params.rowNode.data.id.includes('graph')"},  "getRowNodeId": "function(data) { return data.id; }"},
        )]#"isFullWidthRow": {"function": "params.rowNode.data.id"},

    def gen_umap_plots(self, labels=None, label_legend=None, data=None):
        umap_labels_df, labels_df = _df_select_by_col(
            self.df_raw, ['umap']), _df_select_by_col(self.df_raw, GLOBAL_VARS.umap_labels)
        if umap_labels_df.empty is False:

            data = umap_labels_df[['umap X', 'umap Y']].to_numpy()
            if labels is None:
                labels = labels_df[labels_df.columns.values[0]].to_numpy()
            hover_names = self.df.iloc[:, :]['id'].to_numpy()

        else:
            #if the data is not embedded, embed it
            #first preprocess the data
            pre_df, outliers = preprocess_df(self.df)
            data = dense_umap(pre_df)
            labels = cluster_df(pre_df)
            hover_names = self.df.iloc[[x not in outliers for x in np.arange(len(self.df))], :]['id'].to_numpy()
        fig = px.scatter(x=data[:, 0], y=data[:, 1], color=labels.astype(str), hover_name=hover_names)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(scaleanchor="y", scaleratio=1))
        fig.update_yaxes(automargin=True, autorange=True) #force square aspect ratio
        fig.update_xaxes(automargin=True, autorange=True, )
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
        return fig

    def gen_para_plots(self, *args):

        df = _df_select_by_col(self.df_raw, GLOBAL_VARS.para_vars).iloc[:, :GLOBAL_VARS.para_vars_limit]
        #fig = go.Figure(data=go.Scatter(x=data[:,0], y=data[:,1], mode='markers', marker=dict(color=labels), ids=self.df['id'].to_numpy()))
        fig = px.parallel_coordinates(df, color=GLOBAL_VARS.para_var_colors,
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

    ##INTERNAL FILTER FUNCT
    def _filter_datatable_para(self, selectedData, fig):
        def bool_multi_filter(df, kwargs):
            
            return ' & '.join([f'{key} >= {i[0]} & {key} <= {i[1]}' for key, i in kwargs.items() if key in df.columns])

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

    def _filter_datatable_umap(self, selectedData, fig):
        if selectedData is None:
            out_data = self.df.to_dict('records')
        else:
            selected_ids = [x['hovertext'] for x in selectedData['points']]

            filtered_df = self.df.loc[selected_ids]
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
            combined =  para_out_data.append(umap_out_data)
        #combined = None

        return combined


    def graphClickData(self, d):
        if d is None:
            return dash.no_update
        fig = self.update_cell_plot(d['value']['filename'], None, None, None, self.df)

        return dash.html.A(id=f"graph{d['rowId']}", children=fig.to_json(), style={'display': 'none'})

    def update_cell_plot(self, row_ids, selected_row_ids, data, selected_cell, df):
        
        selected_id_set = set(selected_row_ids or [])

        if row_ids is None:
            dff = self.df
            # pandas Series works enough like a list for this to be OK
            row_ids = self.df['id']
        else:
            dff = self.df.loc[row_ids]

        active_row_ids = selected_row_ids
        if active_row_ids is None or len(active_row_ids) == 0:
            active_row_ids = [self.df.iloc[0]['id']]
        if active_row_ids is not None:
            #determine the amount of different cells to plot, then make up to 4 subplots
            len_active_row_ids = len(active_row_ids)
            if len_active_row_ids == 1:
                fig = make_subplots(rows=1, cols=1, subplot_titles=selected_row_ids)
                plot_coords = [(1,1)]
            elif len_active_row_ids == 2:
                fig = make_subplots(rows=1, cols=2, subplot_titles=selected_row_ids)
                plot_coords = [(1,1), (1,2)]
            elif len_active_row_ids >= 3:
                fig = make_subplots(rows=2, cols=2,subplot_titles=selected_row_ids[:4])
                plot_coords = [(1,1), (1,2), (2,1), (2,2)]
            #now iter through the active row ids and plot them
            for active_row_id in active_row_ids[:4]:
                fold = self.df.loc[active_row_id]["foldername"]
                if isinstance(fold, (list, tuple, np.ndarray, pd.Series)):
                    fold = fold.to_numpy()[0]
                file_path = os.path.join(fold, active_row_id + ".abf")
                x, y, c = loadABF(file_path)

                cutoff = np.argmin(np.abs(x-2.50))
                x, y = x[:, :cutoff], y[:, :cutoff]
                traces = []
                for sweep_x, sweep_y in zip(x, y):
                    traces.append(go.Scatter(x=sweep_x, y=sweep_y, mode='lines', ))
                #fig = go.Figure(data=traces, )
                fig.add_traces(traces, rows=plot_coords[active_row_ids.index(active_row_id)][0], cols=plot_coords[active_row_ids.index(active_row_id)][1])
                fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            fig.update_yaxes(automargin=True)
            fig.layout.autosize = True

            return fig

    def _find_label_cols(self, df):
        label_cols = {}
        df_labels = _df_select_by_col(df, GLOBAL_VARS.umap_labels)
        if df_labels.empty:
            return label_cols
        for col in df_labels.columns:
            temp_dict = {}
            temp_dict['data'] = df_labels[col].to_numpy()
            # if its a str, encode it
            #if isinstance(temp_dict['data'][0], str):
            #    temp_dict['encoder'] = LabelEncoder()
            #    temp_dict['data'] = temp_dict['encoder'].fit_transform(
            #        temp_dict['data'])
            #else:
            temp_dict['encoder'] = None
            label_cols[col] = temp_dict
        return label_cols

    def toggle_collapse(self, n, is_open):
        if n:
            return not is_open
        return is_open

    def _generate_header(self):
        return dbc.Col([html.H1("pyAPisolation", className="text-center"), html.H3("Live Data Visualization",
                    className="text-center"),html.H5("Select a file to view",
                    className="text-center")
        ], className="col-xl-4", style={"max-width": "10%"})

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

# [dash_table.DataTable(
#                 id='datatable-row-ids',
#                 columns=[
#                     {'name': i, 'id': i, 'deletable': True} for i in self.df.columns
#                     # omit the id column
#                     if i != 'id'
#                 ],
#                 data=self.df.to_dict('records'),
#                 filter_action="native",
#                 sort_action="native",
#                 sort_mode='multi',
#                 #row_selectable='multi',
#                 selected_rows=[],
#                 page_action='native',
#                 page_current=0,
#                 page_size=10,
#                 style_cell={
#                     'overflow': 'hidden',
#                     'textOverflow': 'ellipsis',
#                     'maxWidth': 0
#                 },
#                 style_as_list_view=True
#                 )]