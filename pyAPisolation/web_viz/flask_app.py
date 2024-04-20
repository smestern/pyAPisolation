from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def get_data():
    data = [1, 2, 3, 4, 5]  # Replace with your actual data
    return jsonify(data)

if __name__ == '__main__':
    app.run()