import os
from flask import Flask, request, jsonify, render_template
from ragweb import get_answer  

app = Flask(__name__)

# Add a route for the home page
@app.route('/')
def home():
    return render_template('index.html')
     #return "Hello, World!" 

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message')
    answer = get_answer(question)
    return jsonify({"response": answer})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    app.run(debug=True)
