# app/routes/web.py
from __future__ import annotations
from flask import Blueprint, render_template

bp = Blueprint("web", __name__)

@bp.get("/app")
def ui_index():
    return render_template("index.html")
