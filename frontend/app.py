from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)
BACKEND_URL = "http://localhost:8000"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    response = requests.post(
        f"{BACKEND_URL}/upload",
        files={"file": (file.filename, file.stream, "application/pdf")}
    )
    return jsonify(response.json())

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    response = requests.post(
        f"{BACKEND_URL}/ask",
        data={"paper_id": data["paper_id"], "question": data["question"]}
    )
    return jsonify(response.json())

@app.route("/history/<paper_id>")
def history(paper_id):
    response = requests.get(f"{BACKEND_URL}/history/{paper_id}")
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(port=5000, debug=True)