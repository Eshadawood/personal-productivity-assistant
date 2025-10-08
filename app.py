import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from backend.model import get_response
import subprocess

# Load environment variables
load_dotenv()

# ⚠️ Instead of running another Python process every time app starts,
# only call ingest_data.py once when needed
if not os.path.exists("db"):  # only run if the vector DB doesn’t exist
    print("Running data ingestion...")
    subprocess.run(["python", "backend/ingest_data.py"], check=True)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query_assistant():
    try:
        data = request.get_json(force=True)
        user_query = data.get("query", "").strip()

        if not user_query:
            return jsonify({"error": "Query cannot be empty"}), 400

        response = get_response(user_query)

        # If response is an object (LangChain output), make it serializable
        if isinstance(response, dict) and "output_text" in response:
            response = response["output_text"]
        elif not isinstance(response, str):
            response = str(response)

        return jsonify({"response": response})

    except Exception as e:
        # Catch all exceptions to prevent Werkzeug HTML render crash
        print("Error while processing query:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
