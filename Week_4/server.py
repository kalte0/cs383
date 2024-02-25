from flask import Flask, request, jsonify
import pickle
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__) # Initialize the Flask application 

# import and train the Iris dataset. 
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset: 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define a pipeline for preprocessing and training
pipeline = Pipeline([
  ('scaler', StandardScaler()), #Standardize features
  ('classifier', LogisticRegression()) # Logistic regression
])

# Train the model w/ pipeline: 
pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = pipeline.score(X_test, y_test)
print(f'Model accuracy: {accuracy:.2f}')

# Pickle the trained model
with open('logistic_regression_model.pkl', 'wb') as f: 
  pickle.dump(pipeline, f) 

# Load the pre-trained model
model = pickle.load(open('logistic_regression_model.pkl', 'rb')) 

@app.route('/')
def home(): 
  return 'Hello, World! This is the ML model API.'

@app.route('/predict', methods=['POST'])
def predict():
  #Get the data from POST request
  data = request.get_json(force=True)

  # Ensure that we received the expected array of features
  try: 
    features = data['features']
  except KeyError: 
    return jsonify(error="The 'features' key is missing from the request payload."), 400

  # Convert features into the right format and make a prediction 
  prediction = model.predict([features])

  # Return the prediction 
  return jsonify(prediction=int(prediction[0]))


@app.route('/test/<nickname>')
def hello_name(nickname):
    return 'Hello, {}!'.format(nickname)

if __name__ == '__main__':
  app.run(debug=True)

