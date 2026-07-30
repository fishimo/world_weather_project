import dash
from dash import Dash, dcc, html

# Dashインスタンスを生成する。
# pages_folder="" で自動探索を切り、page は下で明示的に import する。
# app_platform は editable install 済みのパッケージなので、自動探索より
# import のほうがパス解決に依存せず確実（page 追加時は import を1行足す）
app = Dash(__name__, use_pages=True, pages_folder="")

# import した時点で各 page の dash.register_page() が実行され、登録される
from app_platform.pages import home, station_detail  # noqa: E402,F401

# 全ページ共通の外枠。page_container の中身が URL に応じて差し替わる
app.layout = html.Div(
    [
        # コールバックから pathname を書き換えることでページ遷移させる
        dcc.Location(id="url"),
        html.H1("日本のAMeDAS観測点"),
        dash.page_container,
    ]
)

if __name__ == "__main__":
    # アプリケーションを起動する
    app.run(debug=True)
