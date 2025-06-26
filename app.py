#from src.Voice_Based_Disease_Screening_using_ML.feature_extraction import audio_files
#for path in audio_files:
#    print(path)
import os
import shutil
#from src.Voice_Based_Disease_Screening_using_ML.feature_extraction import BASE_DIR
import numpy as np
import joblib
def deleted_duplicate_folders(base_dir):
# Loop over all dated folders like 20200413, 20200414, ...

    for date_folder in os.listdir(base_dir):
        date_path = os.path.join(base_dir, date_folder)

        if os.path.isdir(date_path):
            # Check if it has a nested subfolder with the same name
            nested_path = os.path.join(date_path, date_folder)
            if os.path.exists(nested_path):
                print(f"Fixing: {nested_path}")
                
                # Move all subfolders (uids like abc123/) up
                for uid_folder in os.listdir(nested_path):
                    src = os.path.join(nested_path, uid_folder)
                    dest = os.path.join(date_path, uid_folder)
                    shutil.move(src, dest)
                
                # Delete the duplicate nested folder
                os.rmdir(nested_path)

from src.Voice_Based_Disease_Screening_using_ML.model_trainer import ModelTrainer


X_train, X_test, y_train, y_test = joblib.load("data_splits.joblib")
train_array = np.c_[X_train, y_train]
test_array = np.c_[X_test, y_test]

trainer=ModelTrainer()
trainer.initiate_model_trainer(train_array, test_array)

