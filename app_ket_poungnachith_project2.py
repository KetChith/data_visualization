# COMP 4433: Data Visualization
# Project 2: Interactive Dash Application
# Name:      Ket Poungnachith
# Due:       13 Nov 2024
# -------------------------------------

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np
import dash_bootstrap_components as dbc
import yfinance as yf
import datetime

# Initialize the start and end dates for stock historical data
years = 10
end = datetime.datetime.today()
start = end - datetime.timedelta(days=years*365) # 10 years from today

# Pull data from yfinance
VOOG = yf.download('VOOG', start=start, end=end)
VOO = yf.download('VOO', start=start, end=end)
VNQ = yf.download('VNQ', start=start, end=end)
VGT = yf.download('VGT', start=start, end=end)

# Save data to csv files
VOOG.to_csv('voog_historical_data.csv')
VOO.to_csv('voo_historical_data.csv')
VNQ.to_csv('vnq_historical_data.csv')
VGT.to_csv('vgt_historical_data.csv')

# Load CSV files into pandas DataFrames
voog = pd.read_csv('voog_historical_data.csv')
voo = pd.read_csv('voo_historical_data.csv')
vnq = pd.read_csv('vnq_historical_data.csv')
vgt = pd.read_csv('vgt_historical_data.csv')

# Ensure 'Date' column is in datetime format
for df in [voog, voo, vnq, vgt]:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Estimated Value Traded'] = df['Close'] * df['Volume']
    df['30D Moving Avg'] = df['Close'].rolling(window=30).mean()
    df['Daily Percent Change'] = df['Close'].pct_change() * 100

# Map ETF names to DataFrames
etf_data = {
    'VOOG': voog,
    'VOO': voo,
    'VNQ': vnq,
    'VGT': vgt
}

# Determine the earliest and latest dates across all datasets
min_date = min(df['Date'].min() for df in etf_data.values())
max_date = max(df['Date'].max() for df in etf_data.values())

# Create a Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ETF Analysis Dashboard"

# App layout
app.layout = dbc.Container(
    [
        html.H1(
            "ETF Analysis Dashboard",
            style={
                'textAlign': 'center',
                'marginBottom': '30px',
                'color': 'white',
                'fontWeight': 'bold',
            },
        ),
        html.P(
            "Compare ETFs, view KDE plots, analyze cumulative returns, and evaluate risk metrics.",
            style={'textAlign': 'center', 'fontSize': '16px', 'color': 'lightgray'},
        ),
        dbc.Row(
            [
                # Filters and KDE Plot Column
                dbc.Col(
                    [
                        # Filters Card
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Filters", className="card-title", style={'color': 'white'}),
                                    html.Label("Select Date Range:", style={'color': 'white'}),
                                    dcc.DatePickerRange(
                                        id='date-picker-range',
                                        start_date=min_date,
                                        end_date=max_date,
                                        display_format='YYYY-MM-DD',
                                        className="form-control"
                                    ),
                                    html.Br(),
                                    html.Label("Select ETFs:", style={'color': 'white'}),
                                    dcc.Dropdown(
                                        id='etf-dropdown',
                                        options=[{'label': name, 'value': name} for name in etf_data.keys()],
                                        value=['VOOG', 'VOO'],
                                        multi=True,
                                        placeholder="Select one or more ETFs",
                                        className="form-control",
                                    ),
                                    html.Br(),
                                    html.Label("Select Price Type:", style={'color': 'white'}),
                                    dcc.Dropdown(
                                        id='price-type-dropdown',
                                        options=[
                                            {'label': 'Close', 'value': 'Close'},
                                            {'label': '30D Moving Avg.', 'value': '30D Moving Avg'},
                                            {'label': 'Adj. Close', 'value': 'Adj Close'},
                                            {'label': 'Est. Value Traded', 'value': 'Estimated Value Traded'},
                                            {'label': 'Daily Percent Change', 'value': 'Daily Percent Change'},
                                        ],
                                        value='Close',
                                        multi=False,
                                        className="form-control",
                                    ),
                                ]
                            ),
                            className="mb-4",
                            style={'backgroundColor': '#2c3e50', 'border': 'none'},
                        ),
                        # KDE Plot Card
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("KDE Plot", className="card-title", style={'color': 'white'}),
                                    html.P(
                                        "Kernel Density Estimation (KDE) to analyze the distribution and visualize volatility of selected ETFs.",
                                        style={'color': 'lightgray'},
                                    ),
                                    dcc.Loading(dcc.Graph(id='kde-plot', style={'height': '550px'}), type="circle"),
                                ]
                            ),
                            className="mb-4",
                            style={'backgroundColor': '#2c3e50', 'border': 'none'},
                        ),
                    ],
                    width=3,
                ),
                # Line Chart and Cumulative Return Column
                dbc.Col(
                    [
                        # Line Chart Card
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Line Chart", className="card-title", style={'color': 'white'}),
                                    html.P(
                                        "View the trends for selected ETFs over the specified date range.",
                                        style={'color': 'lightgray'},
                                    ),
                                    dcc.Loading(dcc.Graph(id='comparison-graph', style={'height': '390px'}), type="circle"),
                                ]
                            ),
                            className="mb-4",
                            style={'backgroundColor': '#2c3e50', 'border': 'none'},
                        ),
                        # Cumulative Return Chart Card
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(
                                        "Cumulative Return Bar Chart",
                                        className="card-title",
                                        style={'color': 'white'},
                                    ),
                                    html.P(
                                        "Visualize cumulative returns for investing $1 on the selected start date.",
                                        style={'color': 'lightgray'},
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(id='cumulative-return-chart', style={'height': '390px'}), type="circle"
                                    ),
                                ]
                            ),
                            className="mb-4",
                            style={'backgroundColor': '#2c3e50', 'border': 'none'},
                        ),
                    ],
                    width=6,
                ),
                # Correlation Heatmap and Risk Metrics Column
                dbc.Col(
                    [
                        # Correlation Heatmap Card
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Correlation Heatmap", className="card-title", style={'color': 'white'}),
                                    html.P(
                                        "Visualize correlations between selected ETFs.", style={'color': 'lightgray'}
                                    ),
                                    dcc.Loading(
                                        dcc.Graph(id='correlation-heatmap', style={'height': '500px'}), type="circle"
                                    ),
                                ]
                            ),
                            className="mb-4",
                            style={'backgroundColor': '#2c3e50', 'border': 'none'},
                        ),
                        # Risk Metrics Card
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Risk Metrics", className="card-title", style={'color': 'white'}),
                                    html.P("Annualized Volatility:", className="card-text", style={'color': 'lightgray'}),
                                    html.Ul(id='volatility-list', style={'color': 'white'}),
                                    html.P("Sharpe Ratio:", className="card-text", style={'color': 'lightgray'}),
                                    html.Ul(id='sharpe-list', style={'color': 'white'}),
                                ]
                            ),
                            className="mb-4",
                            style={'backgroundColor': '#2c3e50', 'border': 'none'},
                        ),
                    ],
                    width=3,
                ),
            ]
        ),
    ],
    fluid=True,
    style={'backgroundColor': '#1c2833'},#'#1c2833'
)

@app.callback(
    [
        Output('comparison-graph', 'figure'),
        Output('kde-plot', 'figure'),
        Output('cumulative-return-chart', 'figure'),
        Output('correlation-heatmap', 'figure'),
        Output('volatility-list', 'children'),
        Output('sharpe-list', 'children'),
    ],
    [
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('etf-dropdown', 'value'),
        Input('price-type-dropdown', 'value'),
    ]
)
def update_graphs(start_date, end_date, selected_etfs, price_type):
    if not selected_etfs:  # No ETFs selected
        empty_fig = go.Figure().update_layout(
            title="No Data Selected",
            template="plotly_white"
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, [], []

    # Filter data based on the selected date range
    filtered_data = {
        etf: df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        for etf, df in etf_data.items()
    }

    # Define the custom color palette
    color_palette = ['royalblue', 'burlywood', 'slategray', 'gold']

    # Create the line chart
    line_fig = go.Figure()
    for idx, etf in enumerate(selected_etfs):
        df = filtered_data[etf]
        if not df.empty:
            line_fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df[price_type],
                    mode='lines',
                    name=f"{etf} - {price_type}",
                    line=dict(color=color_palette[idx % len(color_palette)], width=1.5),
                    opacity=0.9,
                )
            )

    line_fig.update_layout(
        title=f"Comparison of {', '.join(selected_etfs)} - {price_type}",
        xaxis_title="Date",
        yaxis_title="Value" if price_type != 'Daily Percent Change' else "Percent Change (%)",
        template="plotly_white",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
        legend_title="ETFs",
    )

    # Create the KDE plot
    kde_fig = go.Figure()
    for idx, etf in enumerate(selected_etfs):
        data = filtered_data[etf][price_type].dropna()
        if len(data) > 1:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 500)
            y_kde = kde(x_range)
            kde_fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    fill='tozeroy',
                    mode='lines',
                    name=f"{etf} - KDE",
                    line=dict(color=color_palette[idx % len(color_palette)]),
                )
            )

    kde_fig.update_layout(
        title="KDE Plot of Selected ETFs",
        xaxis_title=price_type,
        yaxis_title="Density",
        template="plotly_white",
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.1),
        legend_title="ETFs",
    )

    # Create the cumulative return bar chart
    cumulative_fig = go.Figure()
    if price_type in ['Daily Percent Change', 'Estimated Value Traded']:
        # Add "NA" annotation
        cumulative_fig.add_annotation(
            text="NA",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=30, color="gray"),
        )
        cumulative_fig.update_layout(
            title="Cumulative Returns Unavailable",
            template="plotly_white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
    else:
        for idx, etf in enumerate(selected_etfs):
            df = filtered_data[etf]
            if not df.empty:
                # Calculate cumulative return
                df['Cumulative Return'] = (1 + df['Close'].pct_change()).cumprod()

                cumulative_fig.add_trace(
                    go.Bar(
                        x=df['Date'],
                        y=df['Cumulative Return'],
                        name=f"{etf} - Cumulative Return",
                        marker=dict(
                            color=color_palette[idx % len(color_palette)],
                            line=dict(color=color_palette[idx % len(color_palette)], width=1),
                        ),
                        opacity=0.5,
                    )
                )

        cumulative_fig.update_layout(
            title="Cumulative Returns of Selected ETFs",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1),
            legend_title="ETFs",
        )

    # Create Correlation Heatmap
    correlation_fig = go.Figure()
    returns_data = pd.DataFrame({
        etf: filtered_data[etf]['Close'].pct_change().dropna()
        for etf in selected_etfs
    })
    if len(selected_etfs) > 1:
        correlation_matrix = returns_data.corr()
        correlation_fig.add_trace(
            go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='Viridis',
                colorbar=dict(title="Correlation"),
            )
        )
        correlation_fig.update_layout(
            title="ETF Correlation Heatmap",
            xaxis_title="ETF",
            yaxis_title="ETF",
            template="plotly_white",
        )
    else:
        correlation_fig.add_annotation(
            text="Select at least 2 ETFs",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
        correlation_fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_white",
        )

    # Calculate Risk Metrics
    volatility_metrics = []
    sharpe_ratio_metrics = []
    for etf in selected_etfs:
        df = filtered_data[etf]
        if not df.empty:
            df['Daily Return'] = df['Close'].pct_change()
            annualized_volatility = df['Daily Return'].std() * np.sqrt(252) #252 is est number of trading days per year
            volatility_metrics.append(html.Li(f"{etf}: {annualized_volatility:.2%}"))
            sharpe_ratio = (df['Daily Return'].mean() / df['Daily Return'].std()) * np.sqrt(252)
            sharpe_ratio_metrics.append(html.Li(f"{etf}: {sharpe_ratio:.2f}"))

    return line_fig, kde_fig, cumulative_fig, correlation_fig, volatility_metrics, sharpe_ratio_metrics

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
