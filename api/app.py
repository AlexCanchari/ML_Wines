from flask import Flask,jsonify,request
from utilities import predict_pipeline
import pandas as pd

app = Flask(__name__)

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    data = request.json
    try:
        df = pd.DataFrame(data['datos'])
        sample = df
    except KeyError:
        return jsonify({'error':'No data sent'})
    
    
    predictions = predict_pipeline(sample)
    result = jsonify(predictions)
    return result

    

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000,debug=True)