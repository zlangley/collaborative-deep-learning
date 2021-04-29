import torch.nn as nn


class CollaborativeDeepLearning(nn.Module):
    def __init__(self, auto_encoders):
        self.auto_encoders = nn.ModuleList(auto_encoders)
