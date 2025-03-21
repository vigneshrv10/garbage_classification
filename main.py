from kivy.app import App
from kivy.uix.webview import WebView

class MyStreamlitApp(App):
    def build(self):
        webview = WebView(url="http://localhost:8501")
        return webview

if __name__ == "__main__":
    MyStreamlitApp().run()
