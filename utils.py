# Python
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np
# from torchmetrics.text import CharErrorRate, WordErrorRate
from torchmetrics.functional.text import char_error_rate as cer
from torchmetrics.functional.text import word_error_rate as wer
from itertools import groupby

def label_to_strings(predictions, targets, chars):
    """Convert model logits and target labels to decoded strings.

    Args:
        predictions: numpy array with shape (seq_len, batch_size, num_classes).
        targets: numpy array with shape (seq_len, batch_size).
        chars: list of vocabulary symbols.

    Returns:
        Tuple of (predicted_strings, target_strings).
    """
    # use argmax to find the index of the highest probability
    argmax_preds = np.argmax(predictions, axis=-1)  # now the shape is (seq_len, batch_size)

    # transpose to get the shape (batch_size, seq_len) in order to get batch_size number of sequences
    grouped_preds = [[k for k, _ in groupby(preds)] for preds in argmax_preds.T]
    grouped_targets = [[k for k, _ in groupby(tgt)] for tgt in np.asarray(targets).T]

    # convert indexes to chars
    texts = ["".join(chars[k] for k in group if k < len(chars)) for group in grouped_preds]
    target_texts = ["".join(chars[k] for k in group if k < len(chars)) for group in grouped_targets]

    return texts, target_texts

# Metrics ------------------------------------------------------------------------------------------
class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = []

    def update(self, y_pred, y_true):
        raise NotImplementedError

    def result(self):
        return np.mean(self.values)

class CERMetric(Metric):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.count = 0
        self.total = 0

    def update(self, y_pred, y_true):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        y_pred, y_true = label_to_strings(y_pred, y_true, self.vocab)
        value = cer(preds=y_pred, target=y_true)
        self.total += value.item()
        self.count += 1
        # self.values.append(value.item())
    
    def result(self):
        return self.total / self.count if self.count > 0 else 1
    
    def reset(self):
        self.values = []
        self.total = 0
        value = 0
        self.count = 0


class WERMetric(Metric):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.count = 0
        self.total = 0

    def update(self, y_pred, y_true):
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        y_pred, y_true = label_to_strings(y_pred, y_true, self.vocab)
        value = wer(preds=y_pred, target=y_true)
        self.total += value.item()
        self.count += 1
    
    def result(self):
        return self.total / self.count if self.count > 0 else 1
    
    def reset(self):
        self.values = []
        self.total = 0
        value = 0
        self.count = 0

# Callbacks ------------------------------------------------------------------------------------------
class Callback:
    def step(self, metric_result):
        raise NotImplementedError

class EarlyStopping(Callback):
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.at_least = 0.5
        self.best_score = None
        self.stop = False

    def step(self, metric_result):
        score = metric_result
        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score:
            self.counter += 1
            # if self.counter >= self.patience and metric_result < self.at_least:
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0

class ModelCheckpoint(Callback):
    def __init__(self, model, filepath):
        self.model = model
        self.filepath = filepath
        self.best_score = None

    def step(self, metric_result):
        if self.best_score is None or metric_result < self.best_score:
            self.best_score = metric_result
            torch.save(self.model.state_dict(), self.filepath)

class ReduceLROnPlateauPyTorch(Callback):
    def __init__(self, optimizer, factor, patience):
        self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

    def step(self, metric_result):
        self.scheduler.step(metric_result)


class TrainLogger(Callback):
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w')

    def step(self, loss, metrics):
        self.file.write(f"Loss: {loss}, Metrics: {metrics}\n")

    def close(self):
        self.file.close()
    # trainLogger = TrainLogger(f"{model_path}/train_log.txt")
    # trainLogger.step(loss.item(), {'CER': cer_metric.result(), 'WER': wer_metric.result()})
    # trainLogger.close()


def decoder(outputs, vocab):
    # Example outputs tensor
    # num_classes = 80
        #outputs = torch.randn(32, 80,num_classes )  # Assuming num_classes is the number of output classes

    # Convert logits to probabilities using softmax
    probs = torch.softmax(outputs, dim=1)

    # Apply CTC decoding
    decoded_strings = []
    for prob in probs:
        # Take the argmax for each time step
        predicted_labels = torch.argmax(prob, dim=1)
        
        # Convert labels to string using some mapping (e.g., index to character mapping)
        predicted_string = ''.join([chr(label + ord('a')) for label in predicted_labels])
        
        # Remove duplicate characters and blank symbols
        final_string = ''
        prev_char = ''
        for char in predicted_string:
            if char != prev_char and char != ' ':
                final_string += char
            prev_char = char
        
        decoded_strings.append(final_string)
    return decoded_strings





class CTCLoss(nn.Module):
    def __init__(self, blank: int, reduction: str="mean", zero_infinity: bool=False):
        """
        Args:
            blank: Index of the blank label
        """
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.blank = blank

    def forward(self, output, target):
        """
        Args:
            output: Tensor of shape (batch_size, num_classes, sequence_length)
            target: Tensor of shape (batch_size, sequence_length)
            
        Returns:
            loss: Scalar
        """
        # Remove padding and blank tokens from target
        target_lengths = torch.sum(target != self.blank, dim=1)
        using_dtype = torch.int32 if max(target_lengths) <= 256 else torch.int64
        device = output.device

        target_unpadded = target[target != self.blank].view(-1).to(using_dtype)

        output = output.permute(1, 0, 2)  # (sequence_length, batch_size, num_classes)
        output_lengths = torch.full(size=(output.size(1),), fill_value=output.size(0), dtype=using_dtype).to(device)

        loss = self.ctc_loss(output, target_unpadded, output_lengths, target_lengths.to(using_dtype))

        return loss
    


def decode_ctc(outputs, chars, blank_index):
    # Take the argmax over the channel dimension to get the predicted character indexes
    predicted_indexes = outputs.argmax(dim=-1)

    predicted_indexes = predicted_indexes.tolist()

    decoded_strings = []

    # for sequence in predicted_indexes:
    #     # Remove repeated characters and blank tokens
    #     sequence = [index for index, _ in groupby(sequence) if index != blank_index]

    #     # Decode the sequence into a string
    #     decoded_string = ''.join([chars[index] for index in sequence])

    #     decoded_strings.append(decoded_string)

    # return decoded_strings
    # For each batch in the sequence
    for batch in zip(*predicted_indexes):
        # Remove repeated characters and blank tokens
        batch = [index for index, _ in groupby(batch) if index != blank_index]

        # Decode the sequence into a string
        decoded_string = ''.join([chars[index] for index in batch])

        # Add the decoded string to the list of decoded strings
        decoded_strings.append(decoded_string)

    return decoded_strings
