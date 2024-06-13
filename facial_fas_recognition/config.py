# --------------------------
# config
# @author: Shi Junjie
# Tue 11 June 2024
# --------------------------

import os
import yaml

class Config:
    
    def __init__(self):
        
        self.CONFIG_FILE = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'config.yaml'
        )
        
        self.pretriained_weights_dir = self.get_directory()
        
        # set weights file path
        self.DETECTOR_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'detector.pt')
        self.SPOOF_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'spoof.pt')
        self.RECOGNIZER_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'recognizer.pt')

    def _get_config(self):
        if not os.path.exists(self.CONFIG_FILE):
            return {"pretrained_weights_dir": ""}
        
        with open(self.CONFIG_FILE, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _set_config(self, key, value):
        config = self._get_config()
        config[key] = value
        
        with open(self.CONFIG_FILE, 'w') as file:
            yaml.safe_dump(config, file)

    def get_directory(self):
        config = self._get_config()
        return config.get("pretrained_weights_dir", "")

    def set_directory(self, directory):
        self._set_config("pretrained_weights_dir", directory)

        self.pretriained_weights_dir = directory
        
        self.DETECTOR_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'detector.pt')
        self.SPOOF_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'spoof.pt')
        self.RECOGNIZER_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'recognizer.pt')


    