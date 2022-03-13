from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import sys
import json
import re


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

target_names = [
    'AE', 'BH', 'DZ', 
    'EG', 'IQ', 'JO', 
    'KW', 'LB', 'LY', 
    'MA', 'OM','PL', 
    'QA', 'SA', 'SD', 
    'SY', 'TN', 'YE'
]

arabic_dialects = {
    'AE': 'لهجة اماراتية', 'BH': 'لهجة بحرينية', 'DZ': 'لهجة جزائرية', 'EG': 'لهجة مصرية', 'IQ': 'لهجة عراقية',
    'JO': 'لهجة أردنية', 'KW': 'لهجة كويتية', 'LB': 'لهجة لبنانية', 'LY': 'لهجة ليبية', 'MA': 'لهجة مغربية', 
    'OM': 'لهجة عمانية', 'PL': 'لهجة فلسطينية', 'QA': 'لهجة قطرية', 'SA': 'لهجة سعودية', 'SD': 'لهجة سودانية',
    'SY': 'لهجة سورية', 'TN': 'لهجة تونسية', 'YE': 'لهجة يمنية'
}

def model(text):
    print(text, file=sys.stderr)
    
    filename = 'model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    pred = loaded_model.predict( [text] )
    pred_p = ( loaded_model.predict_proba( [text] )[0] * 100 ).round(2)  # %
    
    return arabic_dialects[target_names[pred[0]]], pred_p

@app.route('/')
def home():
    for i in range(3): print(i)
    return render_template('home.html')

@app.route('/api')
def predict():
    text_input = request.args.get('text')#.decode("utf-8")
    
    text_input = re.sub(r'[0-9a-zA-Z?]', '', text_input) #remove english words and numbers
    if text_input == "": return "null" 
    
    predict, predict_p = model(text_input)
    
    return jsonify(
        {
            'predict': json.dumps(predict, ensure_ascii = False ),
            'predict_p': predict_p.tolist(),
        }
    )



if __name__ == '__main__':
	app.run(debug=True)