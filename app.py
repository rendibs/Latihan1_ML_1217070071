from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = r"C:\Users\LENOVO\tutorialFlask\tutorialFlask\model\hasil_pelatihan_model.pkl"
with open(model_path, "rb") as model_file:
    ml_model = joblib.load(model_file)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("Prediction started")

    if request.method == 'POST':
        try:
            # Extract values from the form
            RnD_Spend = float(request.form['RnD_Spend'])
            Admin_Spend = float(request.form['Admin_Spend'])
            Market_Spend = float(request.form['Market_Spend'])

            # Prepare the data for the model
            pred_args = [RnD_Spend, Admin_Spend, Market_Spend]
            pred_args_arr = np.array(pred_args).reshape(1, -1)
          
            # Perform prediction
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        
        except ValueError:
            return "Please check if the values are entered correctly"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
        
        # Render the result in predict.html
        return render_template('predict.html', prediction=model_prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
