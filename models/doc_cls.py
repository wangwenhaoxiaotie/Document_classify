import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torchvision.models as torch_models
from .layoutlm import LayoutlmForFeatureExtraction


class DocumentClassifier(nn.Module):
    def __init__(self, layoutlm_config, img_model, feat_channels, num_classes):
        super().__init__()
        self.layoutlm = LayoutlmForFeatureExtraction(layoutlm_config)
        if img_model == "resnet18":
            self.img_model = nn.Sequential(*list(torch_models.resnet18().children())[:-2])
        elif img_model == "resnet34":
            self.img_model = nn.Sequential(*list(torch_models.resnet34().children())[:-2])
        elif img_model == "resnet50":
            self.img_model = nn.Sequential(*list(torch_models.resnet50().children())[:-2])
        elif img_model == "mobilenet_v2":
            self.img_model = torch_models.mobilenet_v2().features
        elif img_model == "mnasnet":
            self.img_model = torch_models.mnasnet1_0().layers
        elif img_model == "shufflenet_v2":
            self.img_model = nn.Sequential(*list(torch_models.shufflenet_v2_x1_0().children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.BatchNorm1d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(feat_channels, num_classes)
        )

    def forward(self, input_ids, bbox, attention_mask, image, label=None):
        lm_feat = self.layoutlm(input_ids, bbox, attention_mask)
        im_feat = self.img_model(image).mean([2, 3])
        feat = torch.cat([lm_feat, im_feat], 1)
        logits = self.classifier(feat)
        
        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, label)
            return logits, loss
        return logits
