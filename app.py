from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import subprocess
import time

app = Flask(__name__)

UPLOAD_FOLDER = "uploads/"
RESULTS_FOLDER = "results/"
SERVER_URL = "http://172.20.193.3:5000"  # Change if needed

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Run Federated Learning Client
        subprocess.Popen(["python", "fl_client.py", SERVER_URL, file_path])

        return render_template("index.html", loading=True, completed=False)

    # Check if results exist to mark training as completed
    results_exist = os.path.exists(os.path.join(RESULTS_FOLDER, "accuracy_graph.png")) and os.path.exists(os.path.join(RESULTS_FOLDER, "client1_predictions.csv"))

    return render_template("index.html", loading=False, completed=results_exist)

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(RESULTS_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
