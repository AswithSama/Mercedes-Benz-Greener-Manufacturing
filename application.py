from flask import Flask, session, request, render_template, redirect, send_file
import pandas as pd
import tempfile
from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)
app=application
app.secret_key = '1928'

# Store the temporary file path in session instead of the CSV data
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        df = pd.read_csv(file)
        pipeline = PredictPipeline() 
        predictions = pipeline.predict(df)
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])       

        #Saving CSV to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        predictions_df.to_csv(temp_file.name, index=False)
        temp_file.close()

        session['predictions_file'] = temp_file.name

        return render_template('results.html', predictions=predictions_df.to_dict(orient='records'))

@app.route('/download_file', methods=['GET', 'POST'])
def download_file():
    if 'predictions_file' in session:
        return send_file(session['predictions_file'], download_name='predictions.csv', mimetype='text/csv', as_attachment=True)
    
    return redirect('/')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
