# import the libraries
from flask import Flask, request, render_template, redirect, url_for 
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from werkzeug.utils import secure_filename
import os
import pickle

# Running the flask app
app = Flask(__name__)


# Define Neural Network Architecture
class ChurnPredictor(nn.Module):
    def __init__(self, input_size):
        super(ChurnPredictor, self).__init__()
        # Define the layers
        self.layer1 = nn.Linear(input_size, 128)  # First hidden layer
        self.layer2 = nn.Linear(128, 64)          # Second hidden layer
        self.output_layer = nn.Linear(64, 1)      # Output layer

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


# Define the input size for the network
input_features = 45

# Create an instance of the network
model = ChurnPredictor(input_features)

# Load the saved state dictionary into the model instance
model.load_state_dict(torch.load('models/churn_predictor_model.pth'))

# Load preprocessor into preprocessor instance 
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)


# Define preprocessing function
def preprocess_data(data):
    # List columns of interest
    val_col = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'tenure', 'phoneservice', 'multiplelines', 'internetservice',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
        'paymentmethod', 'monthlycharges', 'totalcharges']

    # convert cols to lower case
    data.columns = data.columns.str.lower()
    
    # Slice out columns of interest
    data = data[val_col]

    # drop space values under totalcharges if applicable
    try:
        data["totalcharges"] = data["totalcharges"].replace(" ", pd.NA)
        data.dropna(subset=['totalcharges'], inplace=True)
    except:
        pass


    # change totalcharges data type to float
    data['totalcharges'] = data['totalcharges'].astype(float)
    
    # Apply standardscaler on numeric data and OneHotEncoding on
    # categorical columns in the data using trained preprocessor
    processed_data = preprocessor.transform(data)        
    
    # Convert processed_data to PyTorch tensors (numpy)
    input_tensor = torch.tensor(processed_data.astype(np.float32))
  
    return input_tensor


# Set upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Define file to allow
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the app
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Check if the post request has the file part
            if 'file' not in request.files:
                return "No file part"
            
            file = request.files['file']
            
            # If user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return "No selected file"
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)


                # Inside the save file block
                try:
                    
                    # Check if the directory exists
                    if not os.path.exists(app.config['UPLOAD_FOLDER']):
                        # Create the directory if it doesn't exist
                        os.makedirs(app.config['UPLOAD_FOLDER'])
                        print("Directory created successfully")

                    file.save(file_path)

                except Exception as save_error:                    
                    return "An error occurred while saving the file. Please try again later."

                
                # Read uploaded CSV file into a DataFrame                
                uploaded_data = pd.read_csv(file_path)    

                # preprocess the data
                input_tensor = preprocess_data(uploaded_data)            
                
                # Make prediction
                with torch.no_grad():                    
                    model.eval()
                    predictions = model(input_tensor)

                    # check the length of entries
                    if len(predictions) == 1:
                        # if prediction is required on a single observation
                        # convert labels from numeric to actual label
                        predicted_labels = (predictions > 0.5).int().squeeze().numpy()

                        # convert 1 and 0 to yes and no respectively
                        predicted_labels = "".join(["Yes" if predicted_labels == 1 else "No"])
                    
                    else:
                        # if prediction is required on more than one observation simultaneously 
                        # convert labels from numeric to actual label                        
                        predicted_labels = (predictions > 0.5).int().squeeze()

                        # Initialize an empty list to store NumPy arrays
                        prediction_labels_list = []

                        # Loop through each element of the tensor and convert to NumPy array
                        for element in predicted_labels:
                            each_label = element.numpy().tolist()
                            prediction_labels_list.append(each_label) 

                        # convert 1 and 0 to yes and no respectively
                        prediction_labels_list = list(map(lambda x: "Yes" if x == 1 else "No", prediction_labels_list))
                        
                        # Align the prediction with the corresponding entries index. The first one as index of 1                        
                        prediction_labels_list = ", ".join(f"{i+1}. {value}" for i, value in enumerate(prediction_labels_list))

                        predicted_labels = prediction_labels_list                 

                    print(predicted_labels)
                    

                # Redirect to result page with prediction
                return redirect(url_for('result', prediction=predicted_labels))
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
