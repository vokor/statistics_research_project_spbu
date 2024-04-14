from dash import dcc, html, Input, Output

from corr_cluster import get_corr_cluster
from eda import get_eda_layout
from info import app, markdown_content
from model_building import get_model_building

tab_style = {
    'padding': '6px',
    'fontWeight': 'bold'
}

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-eda', children=[
        dcc.Tab(label='Exploratory Data Analysis (EDA)', value='tab-eda', style=tab_style),
        dcc.Tab(label='Correlation and Cluster Analysis', value='tab-correlation', style=tab_style),
        dcc.Tab(label='Model Building', value='tab-model', style=tab_style),
        dcc.Tab(label='Results Presentation and Reporting', value='tab-results', style=tab_style)
    ]),
    html.Div(id='tabs-content')
])


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-eda':
        return get_eda_layout()
    elif tab == 'tab-correlation':
        return get_corr_cluster()
    elif tab == 'tab-model':
        return get_model_building()
    elif tab == 'tab-results':
        return html.Div([
            dcc.Markdown(children=markdown_content)
        ])


if __name__ == '__main__':
    app.run_server(debug=True)
