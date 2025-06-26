from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from src.Voice_Based_Disease_Screening_using_ML.feature_extraction import generate_feature_dataframe
import joblib
import os
import pandas as pd
#defining variables
BASE_DIR = "/Users/vaibhavkavdia/Desktop/Projects_for_Resume/Medical/Voice-Based-Disease-Screening-using-ML/Coswara-Data"

#df=generate_feature_dataframe(base_dir=BASE_DIR)
#os.makedirs(df.to_csv("src/Voice_Based_Disease_Screening_using_ML/coswara_features.csv", index=False),exist_ok=True)
df=pd.read_csv("/Users/vaibhavkavdia/Desktop/Projects_for_Resume/Medical/Voice-Based-Disease-Screening-using-ML/features.csv")
X=df.drop(columns=['path','label'])
y=df['label']

def preprocessing_data(base_dir,output_path='data_splits.joblib',encoder_path='label_encoder.joblib'):
    #base_dir (str): Path to Coswara dataset.
    #output_path (str): Path to save train-test splits.
    # encoder_path (str): Path to save LabelEncoder.
    #define target variable and rest features
    X=df.drop(columns=['path','label'])
    y=df['label']
    
    #encode the string labels into integers
    le=LabelEncoder()
    y_encoded=le.fit_transform(y)
    
    #saving the encoder
    joblib.dump(le,encoder_path)
    
    
    #spliting the data 
    X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2,stratify=y_encoded,random_state=42)
    #Use stratify so that both training and testing sets will have roughly the same ratio of each class i.e. covid and non-covid.
    
    joblib.dump((X_train,X_test,y_train,y_test),output_path)
    return X_train,X_test,y_train,y_test
def scaling_data(X_train,X_test):
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    # Save the scaler for inference later

    joblib.dump(sc, "scaler.joblib")
    return X_train_scaled,X_test_scaled

X_train, X_test, y_train, y_test = preprocessing_data(BASE_DIR)
X_train_scaled,X_test_scaled=scaling_data(X_train,X_test)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
