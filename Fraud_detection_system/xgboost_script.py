# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import os
# import pickle

# def model_fn(model_dir):
#     """
#     Load the XGBoost model from the model_dir.
#     """
#     model_file = os.path.join(model_dir, "xgboost-model")
#     model = pickle.load(open(model_file, 'rb'))
#     return model

# if __name__ == '__main__':
#     # 1. Load the training data
#     train_data_path = os.path.join('/opt/ml/input/data/train', 'train.csv') # SageMaker's training data path
#     train_df = pd.read_csv(train_data_path)

#     # Separate features and target variable
#     X_train = train_df.drop('isFraud', axis=1)  # Replace 'isFraud' if different
#     y_train = train_df['isFraud']

#     # 2. Train the XGBoost model
#     # Hyperparameters (you can pass these as script arguments)
#     # Example:
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--num_round', type=int, default=100)
#     # args, _ = parser.parse_known_args()
#     # num_round = args.num_round

#     num_round = 100
#     objective = 'binary:logistic'

#     # XGBoost training parameters
#     params = {
#         'objective': objective,
#         'eval_metric': 'auc'  # Area Under the Curve is good for fraud
#     }

#     dtrain = xgb.DMatrix(X_train, label=y_train) # Create XGBoost DMatrix
#     model = xgb.train(params, dtrain, num_boost_round=num_round)

#     # 3. Save the model
#     model_dir = "/opt/ml/model" #SageMaker's model directory
#     model_location = os.path.join(model_dir, 'xgboost-model')
#     pickle.dump(model, open(model_location, 'wb'))

#     print("Model saved to {}".format(model_location))
############################################################################
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import os
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import json

# def model_fn(model_dir):
#     """
#     Load the XGBoost model from the model_dir.
#     """
#     model_file = os.path.join(model_dir, "xgboost-model")
#     model = pickle.load(open(model_file, 'rb'))
#     return model

# if __name__ == '__main__':
#     # 1. Load the training data
#     train_data_path = os.path.join('/opt/ml/input/data/train', 'train.csv')
#     train_df = pd.read_csv(train_data_path)

#     # Separate features and target variable
#     X = train_df.drop('isFraud', axis=1)
#     y = train_df['isFraud']

#     # Split into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Convert to DMatrix
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dval = xgb.DMatrix(X_val, label=y_val)

#     # 2. Train the XGBoost model
#     num_round = 100
#     objective = 'binary:logistic'

#     # XGBoost training parameters
#     params = {
#         'objective': objective,
#         'eval_metric': 'auc',
#         'max_depth': 5,       # Reduced max_depth
#         'lambda': 1,       # L2 regularization
#         'alpha': 0        # L1 regularization
#     }

#     evallist = [(dtrain, 'train'), (dval, 'eval')]

#     # Train the model with early stopping
#     model = xgb.train(params, dtrain, num_boost_round=num_round, \
#         evals=evallist, early_stopping_rounds=10)

#     # 3. Log the validation:auc metric (using last)
#     val_preds = model.predict(dval)
#     val_auc = roc_auc_score(y_val, val_preds)
#     print(json.dumps({'validation:auc': val_auc}))

#     # 4. Save the model
#     model_dir = "/opt/ml/model"
#     model_location = os.path.join(model_dir, 'xgboost-model')
#     pickle.dump(model, open(model_location, 'wb'))

#     print("Model saved to {}".format(model_location))
##############################################################################

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def model_fn(model_dir):
    """
    Load the XGBoost model from the `model_dir` directory.
    """
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    return model

if __name__ == '__main__':
    # Read the training data
    train_data_path = os.path.join('/opt/ml/input/data/train', 'train.csv')
    df = pd.read_csv(train_data_path)

    # Separate features and target variable
    X = df.drop('isFraud', axis=1)  # Replace 'isFraud' with your target column
    y = df['isFraud']

    # Train-test split (you might not need this here if you pre-split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['error', 'logloss'],
        'eta': 0.2,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train the XGBoost model
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=100,
                      evals=watchlist, early_stopping_rounds=10, verbose_eval=True)

    # Evaluate the model (Optional - but good practice)
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save the trained model
    model_dir = '/opt/ml/model'  # Standard SageMaker model directory
    model.save_model(os.path.join(model_dir, "xgboost-model"))
