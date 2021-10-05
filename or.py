"""
author: Abhishek
email: abhishekvarmad@gmail.com
"""


from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import os
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logs_dir, "running_logs.log"),level=logging.INFO, format=logging_str
,filemode='a')

def main(data, eta, epochs, filename, plotFileName):
    df = pd.DataFrame(data)
    logging.info(f"This is the actual DataFrame \n{df}")
    X,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)
    _ = model.total_loss() # dummy variable

    save_model(model,filename=filename)
    save_plot(df, plotFileName, model)

if __name__  == '__main__': # << entry point 
    OR = {
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,1]
    }
    ETA = 0.3 # between 0 and 1
    EPOCHS = 10
    try:
        logging.info(">>>> Starting training >>>>")
        main(data=OR, eta=ETA, epochs=EPOCHS, filename="or.model", plotFileName="or.png")
        logging.info("<<<< training done successfully <<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e