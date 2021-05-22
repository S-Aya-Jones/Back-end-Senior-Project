from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS


app = Flask(__name__)

app.config['SECRET_KEY'] = 'password'
cors = CORS(app)


@app.route('/api', methods=['GET'])
def index():
    return jsonify({'name': "Hello From The Backend!"})


if __name__ == "__main__":
    app.run(debug=True)
