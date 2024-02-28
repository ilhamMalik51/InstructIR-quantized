import sys
import os

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
import torch

from utils import *
from models import instructir

from text.models import LanguageModel, LMHead

# this function is considered my_function
SEED=42
seed_everything(SEED=SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG     = "./configs/eval5d.yml"
LM_MODEL   = "./models/lm_instructir-7d.pt"
MODEL_NAME = "./models/im_instructir-7d.pt"

# parse config file
with open(os.path.join(CONFIG), "r") as f:
    config = yaml.safe_load(f)

cfg = dict2namespace(config)

print("Creating InstructIR")
model = instructir.create_model(input_channels=cfg.model.in_ch,
                                width=cfg.model.width,
                                enc_blks=cfg.model.enc_blks,
                                middle_blk_num=cfg.model.middle_blk_num,
                                dec_blks=cfg.model.dec_blks,
                                txtdim=cfg.model.textdim)

print(device)
model = model.to(device)

################### LOAD IMAGE MODEL

assert MODEL_NAME, "Model weights required for evaluation"

print("IMAGE MODEL CKPT:", MODEL_NAME)
model.load_state_dict(torch.load(MODEL_NAME, map_location=torch.device(device)), strict=True)

nparams = count_params(model)
print("Loaded weights!", nparams / 1e6)

onnx_model = torch.onnx.dynamo_export(model)
print("Finished translating into onnx model")