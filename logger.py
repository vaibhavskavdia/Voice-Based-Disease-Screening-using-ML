import logging 
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_path=os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_path,
    format="[%(asctime)s]%(lineno)d%(name)s-%(levelname)s-%(message)s",
    level=logging.INFO,
    
    
)