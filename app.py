from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Tải mô hình đã lưu
model = joblib.load("C:/Users/khanh/Desktop/CD_CNKHDL/Model/svm93.sav")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Invalid input'}), 400
    
    text = request.json['text']
  
    prediction = model.predict([text])
    label = 'Thật' if prediction[0] == 0 else 'Giả'
        
    return jsonify({'prediction': label})

@app.route('/reset', methods=['POST'])
def reset():
    # Xử lý reset (có thể là xóa dữ liệu lưu trữ, khôi phục trạng thái, vv.)
    return jsonify({'message': 'Reset successful'})

if __name__ == '__main__':
    app.run(debug=True)
