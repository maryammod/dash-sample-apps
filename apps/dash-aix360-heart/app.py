import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score

from utils import compute_plot_gam
from modeling import lrr, df, fb, col_map
from modeling import dfTrain, dfTrainStd, dfTest, dfTestStd, yTrain, yTest, y, lime_rf, lrr2
import interpret

def Header(name, app):
    title = html.H2(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 50}
    )

    return dbc.Row([dbc.Col(title, md=9), dbc.Col(logo, md=3)])


def LabeledSelect(label, **kwargs):
    return dbc.FormGroup([dbc.Label(label), dbc.Select(**kwargs)])

# Add bootstrap components if need be https://dash-bootstrap-components.opensource.faculty.ai/examples/iris/

# Compute the explanation dataframe, GAM, and scores
xdf = lrr.explain().rename(columns={"rule/numerical feature": "rule"})
xPlot, yPlot, plotLine = compute_plot_gam(lrr, df, fb, df.columns)
train_acc = accuracy_score(yTrain, lrr.predict(dfTrain, dfTrainStd))
test_acc = accuracy_score(yTest, lrr.predict(dfTest, dfTestStd))


# Start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Card components
cards = [
    dbc.Card(
        [
            html.H2(f"{train_acc*100:.2f}%", className="card-title"),
            html.P("Model Training Accuracy", className="card-text"),
        ],
        body=True,
        color="light",
    ),
    dbc.Card(
        [
            html.H2(f"{test_acc*100:.2f}%", className="card-title"),
            html.P("Model Test Accuracy", className="card-text"),
        ],
        body=True,
        color="dark",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"{dfTrain.shape[0]} / {dfTest.shape[0]}", className="card-title"),
            html.P("Train / Test Split", className="card-text"),
        ],
        body=True,
        color="primary",
        inverse=True,
    ),
]

# Graph components
graphs = [

    [
        LabeledSelect(
            id="select-gam",
            options=[{"label": col_map[k], "value": k} for k in xPlot.keys()],
            value=list(xPlot.keys())[0],
            label="Visualize GAM",
        ),
        dcc.Graph("graph-hole"),
        dcc.Graph("graph-lime"),
        dcc.Graph("graph-gam"),
    ],
    [
        LabeledSelect(
            id="select-coef",
            options=[{"label": v, "value": k} for k, v in col_map.items()],
            value=list(xPlot.keys())[0],
            label="Filter Features",
        ),
        dcc.Graph(id="graph-output"),
        dcc.Graph("graph-exp-AIX"),


    ],

]

app.layout = dbc.Container(
    [
        Header("Dash Heart Disease Prediction with AIX360", app),
        html.Hr(),
        dbc.Row([dbc.Col(card) for card in cards]),
        html.Br(),
        dbc.Row([dbc.Col(graph) for graph in graphs]),
    ],
    fluid=False,
)


@app.callback(
    [ Output("graph-exp-AIX", "figure"), Output("graph-lime", "figure"), Output("graph-gam", "figure"), Output("graph-hole", "figure"), Output("graph-output", "figure")], #Output("graph-exp", "figure"),
    [ Input("select-gam", "value"), Input("select-gam", "value"), Input("select-coef", "value")], #not all figs affected by change in data
)
# coef_fig, lime_fig, gam_fig, hole_fig, output_fig

def update_figures(gam_col2, gam_col, coef_col):

    # Filter based on chosen column
    xdf_filt = xdf[xdf.rule.str.contains(coef_col)].copy()
    xdf_filt["desc"] = "<br>" + xdf_filt.rule.str.replace("AND ", "AND<br>")
    xdf_filt["condition"] = [
        [r for r in r.split(" AND ") if coef_col in r][0] for r in xdf_filt.rule
    ]

    coef_fig = px.bar(
        xdf_filt,
        x="desc",
        y="coefficient",
        color="condition",
        title="Rules Explanations",
    )
    coef_fig.update_xaxes(showticklabels=False)

    if plotLine[gam_col]:
        plot_fn = px.line
    else:
        plot_fn = px.bar

    output_fig = px.scatter(
        x=df[gam_col],
        y=y,
        title="Real Y vales",
        labels={"x": gam_col, "y": "Output Y"},
    )

    gam_fig = plot_fn(
        x=xPlot[gam_col2],
        y=yPlot[gam_col2],
        title="Generalized additive model component",
        labels={"x": gam_col2, "y": "contribution to log-odds of Y=1"},
    )

    # Rola testing doughnut visualizations, Use `hole` to create a donut-like pie chart
    import numpy as np
    predictions = np.squeeze(lrr2.predict_proba(dfTest[1:2]))
    labels = ['Negative', 'Positive']
    print(predictions)
    values = [predictions[0], predictions[1]]
    hole_fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

    lime_local = lime_rf.explain_local(dfTest[:5], yTest[:5], name='LIME') # this takes a long time,may be due to the number of features.

    lime_fig = lime_local.visualize(1)  #go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)]) #lime_local.visualize(1) # no args is the summary, but that is not working in the lime gui either.
    #interpret.show(lime_local)
    return coef_fig, lime_fig, gam_fig, hole_fig, output_fig
    #Output("graph-exp-AIX", "figure"), Output("graph-gam3", "figure"), Output("graph-gam2", "figure"), Output("graph-gam", "figure"), Output("graph-coef", "figure")

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')