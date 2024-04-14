import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State
import statsmodels.api as sm
from scipy import stats

from info import df, app
from scipy.stats import zscore
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

from models import models

df['log_SalePrice'] = np.log(df['SalePrice'])
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop(['SalePrice', 'log_SalePrice'], axis=1)
y = df_encoded['log_SalePrice']
X = sm.add_constant(X)


def get_model_building():
    return html.Div([
        html.Button('Run Regression', id='run-regression', n_clicks=0),
        dcc.Graph(id='regression-qq-plot'),
        dcc.Graph(id='regression-residual-plot'),
        dcc.Graph(id='residual-leverage-plot'),
        html.Pre(id='model-summary'),
    ])


@app.callback(
    [Output('regression-qq-plot', 'figure'),
     Output('regression-residual-plot', 'figure'),
     Output('residual-leverage-plot', 'figure'),
     Output('model-summary', 'children')],
    [Input('run-regression', 'n_clicks')]
)
def update_analysis(n_clicks):
    if n_clicks == 0:
        raise PreventUpdate

    X_copy = X.select_dtypes(include=[np.number])
    X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
    X_copy = X_copy.dropna(axis=0)
    y_copy = y.copy()
    y_copy = y_copy[X_copy.index]
    for i, model in enumerate(models):
        X_copy[f'cluster_predictions_{i}'] = model.predict(X_copy[model.features])

    model = sm.OLS(y_copy, X_copy)
    results = model.fit()
    summary = results.summary().as_text()

    fitted_values = results.predict()
    residuals = results.resid

    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, fit=True)
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers'))
    fig_qq.add_trace(go.Scatter(x=osm, y=intercept + slope * osm, mode='lines'))
    fig_qq.update_layout(
        title='Q-Q with normal distribution',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Observed Quantiles'
    )

    fig_residuals_fitted = go.Figure()
    fig_residuals_fitted.add_trace(go.Scatter(x=fitted_values, y=residuals, mode='markers'))
    fig_residuals_fitted.update_layout(title="Residuals vs Fitted", xaxis_title="Fitted Values",
                                       yaxis_title="Residuals")

    influence = results.get_influence()
    leverage = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]
    standardized_residuals = zscore(residuals)

    df_leverage = pd.DataFrame({
        'Leverage': leverage,
        'Standardized Residuals': standardized_residuals,
        "Cook's Distance": cooks
    })

    fig_leverage = px.scatter(df_leverage, x='Leverage', y='Standardized Residuals', size="Cook's Distance",
                              hover_data=['Cook\'s Distance'],
                              title="Leverage vs Standardized Residuals",
                              labels={"index": "Index Rule"})

    return fig_qq, fig_residuals_fitted, fig_leverage, summary
