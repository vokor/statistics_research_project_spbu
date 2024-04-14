import pandas as pd
import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

df = pd.read_csv('train.csv')

categorical_features = [col for col in df.columns if
                        pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object]
continuous_features = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]


def load_markdown_file():
    with open('README.md', 'r') as file:
        return file.read()


markdown_content = load_markdown_file()
