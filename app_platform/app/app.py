import dash
from dash import html
# Dashインスタンスを生成する
app = dash.Dash(__name__)
# コンポーネントをlayout属性に渡す
app.layout = html.P(
    "こんにちは。Dashが起動できました。",
    # スタイルの設定
    style={
        "fontSize": 50,  # 文字サイズ
        "color": "white",  # 文字色
        "backgroundColor": "#000000",  # 背景色
    },
)

if __name__ == "__main__":
    # アプリケーションを起動する
    app.run(debug=True)