import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from scipy import stats
import numpy as np

from sklearn.cluster import KMeans
from info import df, categorical_features, continuous_features, app
from models import SimpleClusterModel, models
from scipy.stats import f_oneway
from dash.exceptions import PreventUpdate
from itertools import combinations


def get_corr_cluster():
    return html.Div([
        html.Div([
            dbc.Label("Select continuous features for correlation matrix:"),
            html.Div([
                dbc.Button("Select All", id="select-all-continuous-corr", n_clicks=0, color="primary",
                           className="me-1"),
                dbc.Button("Clear All", id="clear-all-continuous-corr", n_clicks=0, color="secondary"),
            ]),
            dcc.Dropdown(
                id='continuous-select-corr',
                options=[{'label': col, 'value': col} for col in continuous_features],
                multi=True,
                style={'marginBottom': '10px'}
            ),
            dbc.Button("Show Correlation Matrix", id="show-corr-matrix", n_clicks=0, style={'marginTop': '10px'})
        ], style={'width': '100%', 'padding': '20px'}),

        dcc.Graph(id='correlation-matrix-graph'),

        html.Button('Show High Correlations', id='show-high-corr', n_clicks=0),
        html.Div(id='high-corr-features'),

        html.Div([
            dbc.Label("Select two variables for correlation t-test:"),
            dcc.Dropdown(
                id='t-test-select-variable-1',
                options=[{'label': col, 'value': col} for col in continuous_features],
                style={'marginBottom': '10px'}
            ),
            dcc.Dropdown(
                id='t-test-select-variable-2',
                options=[{'label': col, 'value': col} for col in continuous_features],
                style={'marginBottom': '10px'}
            ),
            dbc.Input(id='significance-level', type='number', placeholder='Enter significance level (e.g., 0.05)', step=0.01,
                      style={'marginBottom': '10px'}),
            dbc.Button("Perform t-test", id="perform-t-test", n_clicks=0, color="success", style={'marginTop': '10px'})
        ], style={'width': '100%', 'padding': '20px'}),

        html.Div(id='t-test-result-display'),

        html.Div([
            dbc.Label("Set SalePrice Thresholds:"),
            dcc.RangeSlider(
                id='saleprice-range-slider',
                min=df['SalePrice'].min(),
                max=df['SalePrice'].max(),
                step=1000,
                value=[df['SalePrice'].min(), df['SalePrice'].max()],
                marks={int(df['SalePrice'].min()): str(int(df['SalePrice'].min())),
                       int(df['SalePrice'].max()): str(int(df['SalePrice'].max()))},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '50%', 'padding': '20px', 'margin': 'auto'}),

        html.Div([
            dbc.Label("Select Feature for Analysis:"),
            dcc.Dropdown(
                id='feature-select-1',
                options=[{'label': col, 'value': col} for col in df.columns if col != 'SalePrice'],
            ),
            dbc.Button("Perform Tests", id="perform-tests", n_clicks=0, color="primary", style={'marginTop': '20px'})
        ], style={'width': '50%', 'padding': '20px', 'margin': 'auto'}),

        html.Br(),
        html.Div(id='dataset-x-info', style={'width': '50%', 'padding': '20px', 'margin': 'auto'}),
        html.Div(id='dataset-y-info', style={'width': '50%', 'padding': '20px', 'margin': 'auto'}),
        html.Div(id='test-results', style={'width': '50%', 'padding': '20px', 'margin': 'auto'}),

        dbc.Label("Select continuous features for cluster analysis:"),
        html.Div([
            dbc.Button("Select All", id="select-all-features", n_clicks=0, color="primary", className="me-1"),
            dbc.Button("Clear All", id="clear-all-features", n_clicks=0, color="secondary"),
        ]),
        dcc.Dropdown(
            id='features-dropdown',
            options=[{'label': col, 'value': col} for col in continuous_features],
            multi=True,
            style={'marginBottom': '10px'}
        ),
        dbc.Input(id='num-clusters', type='number', placeholder='Enter number of clusters'),
        dbc.Button("Learn", id="learn-clusters", n_clicks=0, style={'marginTop': '10px'}),
        dbc.Button("Save Model", id="save-model", n_clicks=0, style={'marginTop': '10px', 'marginLeft': '10px'}),
        dcc.Graph(id='elbow-plot'),
        dcc.Graph(id='cluster-plot'),
        html.Div(id="anova-results-container"),
    ])


@app.callback(
    Output('continuous-select-corr', 'value'),
    [Input('select-all-continuous-corr', 'n_clicks'),
     Input('clear-all-continuous-corr', 'n_clicks')],
    [State('continuous-select-corr', 'options')]
)
def update_continuous_corr_select(select_all_clicks, clear_all_clicks, options):
    ctx = dash.callback_context
    if not ctx.triggered:
        return []
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'select-all-continuous-corr':
        return [option['value'] for option in options]
    elif button_id == 'clear-all-continuous-corr':
        return []


@app.callback(
    Output('correlation-matrix-graph', 'figure'),
    [Input('show-corr-matrix', 'n_clicks')],
    [State('continuous-select-corr', 'value')]
)
def update_correlation_matrix(n_clicks, selected_features):
    if n_clicks > 0 and selected_features:
        df_filtered = df[selected_features]
        corr_matrix = df_filtered.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", labels=dict(color="Correlation"))
        fig.update_xaxes(side="bottom")
        fig.update_layout(title="Correlation Matrix", xaxis_title="", yaxis_title="")
        return fig
    else:
        return go.Figure()


@app.callback(
    Output('high-corr-features', 'children'),
    [Input('show-high-corr', 'n_clicks')],
    [State('continuous-select-corr', 'value')]
)
def update_high_correlation_list(n_clicks, selected_features):
    if n_clicks > 0 and selected_features:
        df_filtered = df[selected_features]
        corr_matrix = df_filtered.corr()
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs < 1]
        corr_pairs = corr_pairs.drop_duplicates().sort_values(key=abs, ascending=False)

        high_corr_features = html.Ul([
            html.Li(f"{pair[0]} - {pair[1]}: {value:.2f}") for pair, value in corr_pairs.items()
        ])
        return high_corr_features
    else:
        return html.P("No data selected or insufficient clicks.")


@app.callback(
    Output('t-test-result-display', 'children'),
    [Input('perform-t-test', 'n_clicks')],
    [State('t-test-select-variable-1', 'value'),
     State('t-test-select-variable-2', 'value'),
     State('significance-level', 'value')]
)
def perform_t_test(n_clicks, var1, var2, alpha):
    if n_clicks > 0 and var1 and var2 and alpha:
        data1 = df[var1]
        data2 = df[var2]

        corr, p_value = stats.pearsonr(data1, data2)

        result_text = f"Correlation coefficient: {corr:.3f}, P-value: {p_value:.4f}\n"
        if p_value < alpha:
            result_text += "Result is statistically significant."
        else:
            result_text += "Result is not statistically significant."

        return [html.P(result_text)]
    return []


@app.callback(
    [Output('dataset-x-info', 'children'),
     Output('dataset-y-info', 'children'),
     Output('test-results', 'children')],
    [Input('saleprice-range-slider', 'value'),
     Input('feature-select-1', 'value'),
     Input('perform-tests', 'n_clicks')]
)
def update_datasets_and_perform_tests(slider_range, feature1, n_clicks):
    if n_clicks > 0 and feature1:
        threshold_x, threshold_y = slider_range
        dataset_x = df[df['SalePrice'] < threshold_x][feature1]
        dataset_y = df[df['SalePrice'] > threshold_y][feature1]

        def descriptive_stats(data):
            return {
                'Mean': np.mean(data),
                'Median': np.median(data),
                'Standard Deviation': np.std(data, ddof=1),
                'Variance': np.var(data, ddof=1),
                'Minimum': data.min(),
                'Maximum': data.max()
            }

        stats_x = descriptive_stats(dataset_x)
        stats_y = descriptive_stats(dataset_y)

        def format_stats(stats):
            return html.Ul(
                [html.Li(f"{key}: {value:.4f}") if isinstance(value, float) else html.Li(f"{key}: {value}")
                 for key, value in stats.items()]
            )

        info_x = [
            html.H4('Dataset Head Statistics:'),
            format_stats(stats_x)
        ]
        info_y = [
            html.H4('Dataset Tail Statistics:'),
            format_stats(stats_y)
        ]

        def check_normality(data):
            _, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            return p_value

        normal_x = check_normality(dataset_x)
        normal_y = check_normality(dataset_y)
        alpha = 0.05

        if normal_x > alpha and normal_y > alpha:
            stat, p_val = stats.ttest_ind(dataset_x, dataset_y, equal_var=False)
            test_result = f"Perform t-test as distributions are normal: statistic={stat}, p-value={p_val}"
        else:
            stat, p_val = stats.mannwhitneyu(dataset_x, dataset_y, alternative='two-sided')
            test_result = f"Perform Mann-Whitney U Test as distributions are not normal: statistic={stat}, p-value={p_val}"

        return info_x, info_y, test_result

    return "Dataset X details will be shown here.", "Dataset Y details will be shown here.", "No tests performed yet."


@app.callback(
    Output('features-dropdown', 'value'),
    [Input('select-all-features', 'n_clicks'),
     Input('clear-all-features', 'n_clicks')],
    [State('features-dropdown', 'options')]
)
def update_feature_selection(select_all_clicks, clear_all_clicks, options):
    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'select-all-features':
        return [option['value'] for option in options]
    elif triggered_id == 'clear-all-features':
        return []
    else:
        return dash.no_update


@app.callback(
    [
        Output('elbow-plot', 'figure'),
        Output('cluster-plot', 'figure'),
        Output("anova-results-container", "children")
    ],
    [Input('learn-clusters', 'n_clicks')],
    [State('features-dropdown', 'value'),
     State('num-clusters', 'value')]
)
def update_plots(n_clicks, selected_features, n_clusters):
    if n_clicks == 0 or not selected_features or not n_clusters:
        raise PreventUpdate

    data = df[selected_features]

    inertia = []
    k_range = range(1, 15)
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42).fit(data)
        inertia.append(model.inertia_)
    elbow_fig = px.line(
        x=k_range,
        y=inertia,
        title='Elbow Method for Optimal K',
        labels={'x': 'Number of Clusters', 'y': 'Inertia'}
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
    df['Cluster'] = kmeans.labels_
    data['Cluster'] = kmeans.labels_

    cluster_fig = px.scatter_matrix(
        data,
        dimensions=selected_features,
        color='Cluster',
        title="Cluster Scatter Matrix"
    )

    anova_results = []
    cluster_pairs = list(combinations(range(n_clusters), 2))
    for feature in selected_features:
        for pair in cluster_pairs:
            cluster_data_1 = df[df['Cluster'] == pair[0]][feature]
            cluster_data_2 = df[df['Cluster'] == pair[1]][feature]
            f_stat, p_val = f_oneway(cluster_data_1, cluster_data_2)
            result_text = f"Feature {feature}, Cluster {pair[0]} vs Cluster {pair[1]}: F-stat = {f_stat:.3f}, P-value = {p_val:.4f}"
            anova_results.append(html.P(result_text))

    return elbow_fig, cluster_fig, anova_results


@app.callback(
    Output('save-model', 'children'),
    [Input('save-model', 'n_clicks')],
    [State('features-dropdown', 'value'),
     State('num-clusters', 'value')]
)
def save_model(n_clicks, selected_features, n_clusters):
    if n_clicks > 0 and selected_features and n_clusters:
        cluster_model = SimpleClusterModel(selected_features)
        cluster_model.fit(n_clusters)
        models.append(cluster_model)
        return f"Model with {n_clusters} clusters saved!"
    else:
        return "Save Model"



