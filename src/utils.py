import torch


def save_model(model, path='./models/fine_tuned_clip_svhn'):
    model.save_pretrained(path)


def load_model(path='./models/fine_tuned_clip_svhn'):
    return torch.load(path)
