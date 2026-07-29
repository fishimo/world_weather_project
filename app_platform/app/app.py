from dash import Dash, dcc, html

from app_platform.data.stations import load_stations
from app_platform.figures.japan_map import create_japan_map

# Dashインスタンスを生成する
app = Dash(__name__)

# コンポーネントをlayout属性に渡す
app.layout = html.Div(
    [
        html.H1("日本のAMeDAS観測点"),
        dcc.Graph(
            figure=create_japan_map(load_stations()),
            style={"height": "80vh"},
        ),
    ]
)

if __name__ == "__main__":
    # アプリケーションを起動する
    app.run(debug=True)
