import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

# Load the datasets
train_data = pd.read_csv("C:\\Users\\17789\\LHL\\work\\XGBOOST\\Multi-class Classification with XGBoost\\train.csv")
test_data = pd.read_csv("C:\\Users\\17789\\LHL\\work\\XGBOOST\\Multi-class Classification with XGBoost\\test.csv")

# Data preprocessing
# Convert the 'Dates' column to datetime format
train_data['Dates'] = pd.to_datetime(train_data['Dates'])
test_data['Dates'] = pd.to_datetime(test_data['Dates'])

# Feature engineering
# For example, extracting the hour of the crime
train_data['Hour'] = train_data['Dates'].dt.hour
test_data['Hour'] = test_data['Dates'].dt.hour

# Encode categorical variables
le = LabelEncoder()
train_data['Category_encoded'] = le.fit_transform(train_data['Category'])

# Define your feature columns and target variable
X = train_data[['Hour', 'X', 'Y']] # Add other features you've engineered
y = train_data['Category_encoded']

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict_proba(X_val)

# Evaluate the model
val_log_loss = log_loss(y_val, val_predictions)
print(f"Validation Log Loss: {val_log_loss}")

# Make predictions on the test set for submission
test_predictions = model.predict_proba(test_data[['Hour', 'X', 'Y']])

# Format predictions for submission
submission_columns = [
    'Id', 'ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
    'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT',
    'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING',
    'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',
    'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION',
    'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
    'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC',
    'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS'
]

# match the encoded categories to the columns
category_mapping = dict(zip(le.classes_, submission_columns[1:]))
sorted_labels = [category_mapping[label] for label in le.classes_]

# Add the 'Id' column from the test dataset
submission_df = pd.DataFrame(test_predictions, columns=sorted_labels)
submission_df.insert(0, 'Id', test_data['Id'])

# Save the submission file
submission_file_path = "C:\\Users\\17789\\LHL\\work\\XGBOOST\\submission.csv"
submission_df.to_csv(submission_file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

# Display the first few rows of the submission file
print(submission_df.head())