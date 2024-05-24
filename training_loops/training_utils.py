import torch 
import json
import hashlib
import pandas as pd
import os
from datetime import datetime
import logging

def compute_acc(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x = x.to(device=device)
            y = y.to(device=device)

            predictions = model(x)
            predictions[predictions > 0] = 1
            predictions[predictions < 0] = 0
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        return num_correct/num_samples
    

class DumpJSON():
    
    def __init__(self,
                 obj=None, name=None,
                 read_path='results.json', write_path='results.json'):
        if obj is not None:
            path = obj.anals_results_path+'/'+name+'.json'
            read_path=path
            write_path=path
        
        self.read_path=read_path
        self.write_path=write_path
        
        try:
            with open(self.read_path, 'r') as fp:
                self.results = json.load(fp)
        except:
                self.results = {}

    def count(self):
        return(len(self.results))

    def append(self, x):
        
        json_x = json.dumps(x)
        hash = hashlib.sha1(json_x.encode("UTF-8")).hexdigest()
        hash = hash[:10]
        tmp  = {hash:x}
        self.results.update(**tmp)
    
    def save_to_csv(self):
        self.save()
        self.to_csv()
        
    def save(self):
        with open(self.write_path, 'w') as fp:
            json.dump(self.results, fp)

    def to_csv(self):
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        filename = self.write_path
        filename = filename.split('.')
        if len(filename)>1:
            filename[-1] = 'csv'

        filename = '.'.join(filename)
        df.to_csv(filename)

def save_initialization(model, checkpoint_path):
    temp_path = os.path.join(os.path.dirname(checkpoint_path), "temp.pt")

    training_state = {
        'model' : model.state_dict(),
    }

    torch.save(training_state, temp_path)
    os.replace(temp_path, checkpoint_path)
    msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + checkpoint_path
    logging.info(msg)

def rewind_model(base_model, checkpoint_path):
    training_state = torch.load(checkpoint_path)
    base_model.load_state_dict(training_state['model'])

    logging.info("rewinded model!")

    return base_model