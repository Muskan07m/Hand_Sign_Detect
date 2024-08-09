from flask import Flask, render_template, jsonify, Response
import subprocess
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('a.html')

@app.route('/data_coll', methods=['POST'])
def data_coll():
    try:
        # Execute the data.py script
        subprocess.run(["python", "C:/Users/muska/Downloads/Sign_Detection/data.py"])
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Execute the train.py script using subprocess
        result = subprocess.run([sys.executable, 'train.py'], capture_output=True, text=True)
        
        # Extract accuracy from the output
        lines = result.stdout.split('\n')
        accuracy_line = [line for line in lines if 'Validation Accuracy:' in line]
        accuracy = float(accuracy_line[0].split(':')[1].strip()) if accuracy_line else None
        
        return jsonify({'status': 'success', 'accuracy': accuracy})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/try_demo', methods=['POST'])
def try_demo():
    try:
        # Execute the test.py script
        subprocess.run(["python", "C:/Users/muska/Downloads/Sign_Detection/test.py"])
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)