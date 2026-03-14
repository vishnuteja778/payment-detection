"""
Vercel serverless entry point.
Vercel looks for a WSGI-compatible `app` object in this file.
"""
import sys
import os

# Add the project root to the Python path so we can import app.py
# On Vercel, the function runs from the api/ directory, so we need
# to go one level up to find app.py and its dependencies.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Change working directory to project root so relative paths resolve
os.chdir(project_root)

from app import app

# Vercel looks for `app` — the WSGI application object
