from flask import Flask, request
from flask_cors import CORS

from use_ready import bn_4, bn_11

app = Flask(__name__)
CORS(app)

@app.route("/", methods = ['POST', 'GET'])
def predict():
    if(request.method != 'POST'):
        return "Bad Request Method"
    
    text = request.json.get("text")
    m_type = request.json.get("type")
    
    if(m_type == "Experimental"):
        bn = bn_11
    else:
        bn = bn_4
        
    return bn.pred_textual(text)
    
    
app.run(
    port = 8080
)