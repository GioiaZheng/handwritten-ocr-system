# import libraries and modules
from sklearn.model_selection import train_test_split
import os
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Grayscale
from torch.utils.data import DataLoader
from tqdm import tqdm

# import custom modules
from preprocessor import Preprocessor
from data_loader import HandwritingDataset

# import the CRNN + CTC model
from crnn_ctc import Model as Model1
# from model2 import Network as Model2
# from model420 import Model as Model420
# from model_net import Network as ModelNet
# from crnn import CRNN
# from model3 import ImageToWordModel as Model3

# Import modules and fucntions
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.nn import CTCLoss
from utils import label_to_strings
from torchmetrics.text import CharErrorRate
from torchmetrics.text import WordErrorRate

# Function to set the device (CUDA if available, CPU otherwise)
def set_device():
    """
    Set the device. CUDA if available, CPU otherwise

    Args:
        None

    Returns:
        Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled in this device/notebook. \n"
              "If you want to enable it in colab, in the menu under `Runtime` -> \n"
              "`Hardware accelerator.` and select `GPU` from the dropdown menu")
    else:
        print("GPU is enabled in this device/notebook. \n"
              "If you want to disable it, in the menu under `Runtime` -> \n"
              "`Hardware accelerator.` and select `None` from the dropdown menu")

    return device

# Create a function to handle key press events
def on_key(event):
    if event.key == 'r':
        plt.close(event.canvas.get_window_title())# Close the window when 'r' key is pressed

def test():
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)

    metadata_folder = os.path.join('data', 'iam_sentences', 'metadata', 'sentences.txt')
    images_folder = os.path.join('data', 'iam_sentences', 'dataset')
    all_metadata = open(metadata_folder, "r").readlines()

    device =set_device()
    # device = "cpu"
    max_len = 0
    vocab = set()
    dataset = []

    # Read and preprocess metadata
    for line in tqdm(all_metadata):
        if line.startswith("#"):
            continue
        line_split = line.split(" ")
        if line_split[2] == "err":
            continue
        file_name = line_split[0] + ".png"
        label = line_split[-1].rstrip('\n')
        label = label.replace('|', ' ')
        rel_path = os.path.join(images_folder, file_name)
        if not os.path.exists(rel_path):
            continue
        dataset.append([rel_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

    # build vocabulary
    vocab = ''.join(sorted(vocab))  # vocabulary: a string containing all possible characters

    # Split the dataset into training, validation, and test sets
    # since we used the same random_state, we will get the same split as in train.py
    # so the model never seen any image in the test_data
    # we could also save the test split in a csv file and load it here
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    # create the dataset object for testing (the other 2s are not needed)
    test_dataset = HandwritingDataset(test_data, vocab, max_len)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("number of batches in the test loader:", test_loader.__len__())

    # Define model, loss function, and evaluation matrics
    blank = len(vocab)
    print(f'blank token: {blank}')
    criterion = CTCLoss(blank=len(vocab), reduction='mean', zero_infinity=True)
    cer_metric = CharErrorRate()
    wer_metric = WordErrorRate()
    model_path = os.path.join("Results", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M"))
    tensorboard = SummaryWriter(f"{model_path}/logs")



    input_dimension = 1
    # input_dimension = (1, 32, 256)
    output_dimension = len(vocab)

    model = Model1(input_dimension, output_dimension)
    model = model.to(device)
    path = os.path.join('Model', 'model.pt')
    state_dict = torch.load(path, torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()

    # state_dict = torch.load('C:/Users/gioia.zheng/Desktop/ACSAI/AI_lab/Handwritten-Text-Recognizer/Results/2024-04-23-20-52/model.pt', map_location=torch.device(device))
    # model.load_state_dict(state_dict)

    # model = ModelNet(output_dimension)
    # state_dict = torch.load('model01.pt', map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    # model.eval()

    tensorboard = SummaryWriter(f"{model_path}/logs")

    with torch.no_grad():
        display_images= True
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.unsqueeze(1)
            # inputs = inputs.permute(0,3,1,2)
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)

            target_lengths = torch.sum(targets != blank, dim=1)
            device = outputs.device
            target_unpadded = targets[targets != blank].view(-1).to(torch.int64)
            outputs_temp = outputs.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
            output_lengths = torch.full(size=(outputs_temp.size(1),), fill_value=outputs_temp.size(0), dtype=torch.int64).to(device)
            loss = criterion(outputs_temp, target_unpadded, output_lengths, target_lengths.to(torch.int64))
            # if device == "cuda":
            #     torch.cuda.empty_cache()
            preds, answers = label_to_strings(outputs.cpu().numpy(), targets.cpu().numpy(), vocab)

            # input --> numpy
            inputs_numpy = inputs.detach().cpu().numpy()


            # show the image and the result got
            if display_images == True:
                for y in range(64):
                    cv2.imshow(preds[y], inputs_numpy[y].squeeze())
                    print(f'our prediction: {preds[y]}')
                    print(f'our answer: {answers[y]}')
                    print()
                    if cv2.waitKey(2000) & 0xFF == ord('q'):
                        display_images = False
                        cv2.destroyAllWindows()
                        break

            cer_metric.update(preds, answers)
            wer_metric.update(preds, answers)
            # if display_images == False:
            #      break

            for pred, answer in zip(preds, answers):
                print(f'our prediction: {pred}')
                print(f'our answer: {answer}')
                print()

            tensorboard.add_scalar('Loss/test', loss.item(), i)
            tensorboard.add_scalar('CER/test', cer_metric.compute(), i)
            tensorboard.add_scalar('WER/test', wer_metric.compute(), i)

        cer_result, wer_result = cer_metric.compute(), wer_metric.compute()
        print(f'our final character error rate: {cer_result}')
        print(f'our final word error rate: {wer_result}')

        tensorboard.close()

    return

test()
