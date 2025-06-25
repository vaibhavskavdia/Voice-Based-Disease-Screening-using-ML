import logging.handlers
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name="Voice-Based-Disease-Screening-using-ML"
list_of_files=[
    #".github/workflows/.gitkeep",
    f"src/{project_name}/feature_extraction.py",
    f"src/{project_name}/train_model.py"
    f"notebooks/{project_name}/visualisation.ipynb"
    "app.py",
    "Dockerfile",
    "requirement.txt",
    "setup.py",
    
    
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory:{filedir} for file {filename}")
        
    if (not os.path.exists(filepath) or (os.path.getsize(filepath)==0)):
        with open(filepath,"w") as f:
            pass
            logging.info(f"creating empty file:{filepath}")
    
    else:
        logging.info(f"{filename}: already exists")