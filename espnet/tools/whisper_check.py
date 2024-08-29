import torch
import yaml
from espnet2.bin.asr_inference import Speech2Text
import soundfile
import librosa
import librosa.display
from datetime import datetime

# change the path based on the model and config that you want to use
base = 
model_path = base + "/weight.pth"
config_path = base + "/config.yaml"
speech2text = Speech2Text(  asr_model_file=model_path,
                            asr_train_config=config_path,
                            device='cuda',
                            minlenratio=0.0,
                            maxlenratio=0.0,
                            ctc_weight=0.0,
                            beam_size=1,  
                            batch_size=0                   
                          )

def asr(wav_file):
    audio3, rate = librosa.load(wav_file, sr=16000)
    nbests = speech2text(audio3)
    text, *_ = nbests[0]
    return text

if __name__ == '__main__':
    # define speech file here
    b = '/home/audio_files/'
    wav_files = ['model.wav']
    # wav_files = ['aaa.wav', 'model.wav']
    for file in wav_files:
        wav_file = b+file
        text = asr(wav_file)
        print(text)