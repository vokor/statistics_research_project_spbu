import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from info import df, categorical_features, continuous_features, app


def get_eda_layout():
    return html.Div([
        html.Div([
            dbc.Label("Select categorical features:"),
            html.Div([
                dbc.Button("Select All", id="select-all-categorical", n_clicks=0, color="primary", className="me-1"),
                dbc.Button("Clear All", id="clear-all-categorical", n_clicks=0, color="secondary"),
            ], style={'marginBottom': '10px', 'marginTop': '10px'}),
            dcc.Dropdown(
                id='categorical-select',
                options=[{'label': col, 'value': col} for col in categorical_features],
                multi=True
            )
        ], style={'width': '100%', 'padding': '20px'}),

        html.Div([
            dbc.Label("Select continuous features for general information:"),
            html.Div([
                dbc.Button("Select All", id="select-all-continuous-info", n_clicks=0, color="primary",
                           className="me-1"),
                dbc.Button("Clear All", id="clear-all-continuous-info", n_clicks=0, color="secondary"),
            ]),
            dcc.Dropdown(
                id='continuous-select-info',
                options=[{'label': col, 'value': col} for col in continuous_features],
                multi=True,
                style={'marginBottom': '10px'}
            )
        ], style={'width': '100%', 'padding': '20px'}),

        html.Div(id='output-container'),
    ])


@app.callback(
    Output('output-container', 'children'),
    [Input('categorical-select', 'value'),
     Input('continuous-select-info', 'value')]
)
def update_output(selected_categoricals, selected_continuous):
    children = []

    for feature in selected_categoricals:
        nan_count = df[feature].isna().sum()
        counts = df[feature].value_counts()

        fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.3)])
        fig.update_layout(title_text=f'Distribution of {feature}', title_x=0.5)

        stats_card = dbc.Card([
            dbc.CardBody([
                html.H5(f"Feature: {feature} (Categorical)", className="card-title"),
                html.P(f"Number of NaNs: {nan_count}"),
                html.P(f"Value Counts:"),
                html.Ul([html.Li(f"{k}: {v}") for k, v in counts.items()])
            ])
        ])

        feature_container = html.Div([
            dcc.Graph(figure=fig, style={'flex': '50%'}),
            html.Div(stats_card, style={'flex': '50%', 'padding': '20px'})
        ], style={'display': 'flex', 'flex-wrap': 'nowrap', 'justify-content': 'space-between',
                  'align-items': 'center'})

        children.append(html.Div(feature_container, style={'marginTop': '20px', 'marginBottom': '40px'}))

    for feature in selected_continuous:
        nan_count = df[feature].isna().sum()
        mean = df[feature].mean()
        median = df[feature].median()
        percentiles = df[feature].quantile([0.25, 0.5, 0.75, 0.95]).to_dict()

        fig = px.histogram(df, x=feature, title=f'Distribution of {feature}')

        stats_card = dbc.Card([
            dbc.CardBody([
                html.H5(f"Feature: {feature} (Numerical)", className="card-title"),
                html.P(f"Number of NaNs: {nan_count}"),
                html.P(f"Mean: {mean:.2f}"),
                html.P(f"Median: {median:.2f}"),
                html.P(f"25th Percentile: {percentiles[0.25]:.2f}"),
                html.P(f"50th Percentile: {percentiles[0.5]:.2f}"),
                html.P(f"75th Percentile: {percentiles[0.75]:.2f}"),
                html.P(f"95th Percentile: {percentiles[0.95]:.2f}")
            ])
        ])

        feature_container = html.Div([
            dcc.Graph(figure=fig, style={'flex': '50%', 'display': 'inline-block'}),
            html.Div(stats_card, style={'flex': '50%', 'padding': '20px', 'display': 'inline-block'})
        ], style={'display': 'flex', 'flex-wrap': 'nowrap', 'justify-content': 'space-between',
                  'align-items': 'center'})

        children.append(html.Div(feature_container, style={'marginTop': '20px', 'marginBottom': '40px'}))

    return children


@app.callback(
    Output('categorical-select', 'value'),
    [Input('select-all-categorical', 'n_clicks'),
     Input('clear-all-categorical', 'n_clicks')],
    [State('categorical-select', 'options')]
)
def update_categorical_select(select_all_clicks, clear_all_clicks, options):
    ctx = dash.callback_context
    if not ctx.triggered:
        return []
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'select-all-categorical':
        return [option['value'] for option in options]
    elif button_id == 'clear-all-categorical':
        return []


@app.callback(
    Output('continuous-select-info', 'value'),
    [Input('select-all-continuous-info', 'n_clicks'),
     Input('clear-all-continuous-info', 'n_clicks')],
    [State('continuous-select-info', 'options')]
)
def update_continuous_select(select_all_clicks, clear_all_clicks, options):
    ctx = dash.callback_context
    if not ctx.triggered:
        return []
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'select-all-continuous-info':
        return [option['value'] for option in options]
    elif button_id == 'clear-all-continuous-info':
        return []

