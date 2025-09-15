from flask import Flask, jsonify
from .core.config import Config
from .utils.logging import configure_logging

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Logging
    configure_logging(level=app.config.get("LOG_LEVEL", "INFO"))

    # Basic JSON settings (UTF-8, etc.)
    app.config.update(JSON_AS_ASCII=False)

    # Blueprints
    from .routes.health import bp as health_bp
    from .routes.predict import bp as predict_bp      # already added in backend step 7
    from .routes.analysis import bp as analysis_bp    # already added in backend step 7
    from .routes.web import bp as web_bp              # NEW

    app.register_blueprint(health_bp,  url_prefix="/api")
    app.register_blueprint(predict_bp, url_prefix="/api")
    app.register_blueprint(analysis_bp, url_prefix="/api")
    app.register_blueprint(web_bp)  # NEW  (serves /app)

    # Root ping (keep JSON for quick health checks)
    @app.get("/")
    def index():
        return jsonify({"app": "stockapp", "status": "ok"})

    return app
