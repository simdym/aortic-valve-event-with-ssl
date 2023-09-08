import os
import torch
from torch.utils.data import Dataset

import json

class AccRecording():
    def __init__(self, file_path: str, start_index: int, end_index: int, avo: bool, avc: bool):
        self.file_path = file_path
        self.start_index = start_index
        self.end_index = end_index
        self.avo = avo
        self.avc = avc
    
    def get_data(self):
        with open(self.file_path) as f:
            file_dict = json.load(f)

            return torch.Tensor([file_dict["acc_x"][self.start_index:self.end_index],
                                  file_dict["acc_y"][self.start_index:self.end_index],
                                  file_dict["acc_z"][self.start_index:self.end_index]])
    
    def is_avo(self):
        return self.avo
    
    def is_avc(self):
        return self.avc
    
    def get_label(self):
        return torch.Tensor([int(self.avo), int(self.avc)])

class ValveDataset(Dataset):
    def __init__(self, data_path, sample_len, overlap_len):
        """"""
        self.data_path = data_path
        self.data = []

        for experiment in os.listdir(self.data_path):
            for animal in os.listdir(os.path.join(self.data_path, experiment)):
                for intervention in os.listdir(os.path.join(self.data_path, experiment, animal)):
                    for file in os.listdir(os.path.join(self.data_path, experiment, animal, intervention)):
                        file_path = os.path.join(data_path, experiment, animal, intervention, file)
                        with open(file_path) as f:
                            file_dict = json.load(f)

                            rec_len = len(file_dict["acc_x"])
                            sample_start = 0
                            sample_end = sample_len

                            avo = file_dict["avo"]
                            avc = file_dict["avc"]

                            while sample_end < rec_len:
                                avo_in_sample = any(i >= sample_start and i < sample_end for i in avo)
                                avc_in_sample = any(i >= sample_start and i < sample_end for i in avc)
                                acc_rec = AccRecording(file_path, sample_start, sample_end, avo_in_sample, avc_in_sample)

                                self.data.append(acc_rec)

                                sample_start += sample_len - overlap_len
                                sample_end += sample_len - overlap_len
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].get_data(), self.data[idx].get_label()
                                        


if __name__ == "__main__":
    data_path = "C:\\Users\\simen\\datasets\\epicardially-attached-cardiac-accelerometer-data-from-canines-and-porcines-1.0.0\\accelerometer_data"
    dataset = ValveDataset(data_path, 68, 16)
    print(len(dataset))

    for i in range(10):
        import matplotlib.pyplot as plt
        
        print(dataset[i][0][0].shape)
        print(dataset[i][0][1].shape)
        print(dataset[i][0][2].shape)
        print("Avo", dataset[i][1])
        print("Avc", dataset[i][2])
        print()
        plt.plot(dataset[i][0][0])
        plt.plot(dataset[i][0][1])
        plt.plot(dataset[i][0][2])
        plt.title("Avo: " + str(dataset[i][1]) + " Avc: " + str(dataset[i][2]))
        plt.show()