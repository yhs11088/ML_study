import glob
import numpy as np
from tqdm import tqdm
from torch.utils import data
import soundfile as sf

class RawDataset(data.Dataset):
    def __init__(self, common_path, audio_window = 20480):
        '''
        common_path : common path for LibriSpeech audio files
        - audio file path = f"{common_path}/{speaker id}/{chapter id}/{number}.flac"
        '''
        self.common_path = common_path
        self.audio_window = audio_window
        
        self.audio_paths = glob.glob(f"{common_path}/*/*/*flac")   
        self.audio = []
        for path in tqdm(self.audio_paths):
            audio, sr = sf.read(path)
            if len(audio) > audio_window:
                self.audio.append(audio)

    def __len__(self):
        return len(self.audio)
    
    def __getitem__(self, index):
        audio = self.audio[index]
        starttime = np.random.randint(len(audio) - self.audio_window + 1)
        return audio[starttime:starttime + self.audio_window]


