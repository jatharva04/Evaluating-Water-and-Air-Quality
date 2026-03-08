import os
from flask import Flask

from config import *
from services.ml_service import *
from services.chatbot_service import *

from routes.pages import pages_bp
from routes.api import api_bp

app = Flask(__name__)
 
# Register Blueprints
app.register_blueprint(pages_bp)
app.register_blueprint(api_bp)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
