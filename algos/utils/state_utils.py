import torch
from os import listdir
from os.path import isfile, join
import logging

class ModelUtils:
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.file_name = 'checkpoint'
        self.latest = None
        self.count = 0
        self.logger = logging.getLogger(__name__)

    def save_state(self, overwrite=False):
        path = join(self.path, self.file_name)
        if not overwrite:
            path = f'{path}-{self.count}'
            self.count += 1
        
        self.latest = path
        torch.save(self.model.state_dict(), path)

    def load_state(self, checkpoint_index):
        onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        checkpoints = [ckp for ckp in onlyfiles if ckp.startswith(self.file_name)]
        checkpoints.sort()
        
        if checkpoint_index >= self.count:
            pass