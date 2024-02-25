from flask import Flask, request, jsonify
import pickle
import pandas as pd # used to read in the dataset 
from sklearn.datasets import load_diabetes # where we will get our model. 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier 

app = Flask(__name__) #Create a Flask application instance. 

#Load in a model from online: 
X, y = load_diabetes(return_X_y=True, as_frame=True)
X.head() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#after this line, y_train will be in the form of doubles. However, the model will expect them as integers. To fix this: 

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train) 

# a sample model downloaded which uses XGBoost (Extreme Gradient Boosting), a popular and powerful machine learning. 
model = XGBClassifier(base_score=0.5, # The initial prediction score of all instances 
             booster= 'gbtree', # the type of boosting algorithm used-- here: uses decision trees. 
              colsample_bylevel=1, #subsample ratio of columns for each level
              colsample_bynode=1,
              colsample_bytree=1, 
              gamma=0, 
              gpu_id=-1, 
              importance_type='gain', 
              interaction_constraints='',
              learning_rate=0.300000012, 
              max_delta_step=0, 
              max_depth=6,
              min_child_weight=1, 
              # missing=nan, # should be this by default, threw an error when I included.
              monotone_constraints='()',
              n_estimators=100, 
              n_jobs=0, 
              num_parallel_tree=1,
              objective='multi:softprob', 
              random_state=0, reg_alpha=0,
              reg_lambda=1, 
              scale_pos_weight=None, 
              subsample=1,
              tree_method='exact', 
              validate_parameters=1, 
              verbosity=None)

model.fit(X_train, y_train_encoded)
pickle.dump(model, open('model.pkl', 'wb')); # dump the model int model.pkl

pickled_model = pickle.load(open('model.pkl', 'rb')) # loads a pre-trained model saved in pickle.
pickled_model.predict(X_test)

@app.route('/') # home page: 
def hello_world():
  return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict(): 
    data = request.get_json(force=True) # Get the JSON data from the request
    prediction = pickled_model.predict([data['features']]) # make a prediction using the loaded model
    return jsonify(prediction=prediction[0]) # return prediction as JSON response

if __name__ == '__main__':
  app.run(debug=True)

