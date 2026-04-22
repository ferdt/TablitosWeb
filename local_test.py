import sys
import os

# Append current directory to path so we can import functions.main
sys.path.append(os.path.join(os.path.dirname(__file__), 'functions'))

try:
    from functions.main import app
    from flask import send_from_directory
except Exception as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

@app.route('/')
def root():
    return send_from_directory('../public', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # This will serve the static files from the 'public' directory
    return send_from_directory('../public', path)

if __name__ == '__main__':
    print("Serving locally at http://localhost:5000/tablas.html")
    app.run(port=5000, debug=True)
