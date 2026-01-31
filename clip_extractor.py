import torch
import open_clip
from PIL import Image
import numpy as np

from attribute_vocab import (
    COLOR_VOCAB, NECKLINE_VOCAB, SLEEVE_VOCAB,
    PATTERN_VOCAB, FABRIC_VOCAB, FIT_VOCAB
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device)

def clip_similarity(image_emb, texts):
    with torch.no_grad():
        text_tokens = open_clip.tokenize(texts).to(device)
        text_emb = model.encode_text(text_tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        sims = (image_emb @ text_emb.T).cpu().numpy()[0]
    return sims

def extract_clip_attributes(image_path):
    img = Image.open(image_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        img_emb = model.encode_image(img)
        img_emb /= img_emb.norm(dim=-1, keepdim=True)

    return {
        "color": COLOR_VOCAB[np.argmax(clip_similarity(img_emb, COLOR_VOCAB))],
        "neckline": NECKLINE_VOCAB[np.argmax(clip_similarity(img_emb, NECKLINE_VOCAB))],
        "sleeve": SLEEVE_VOCAB[np.argmax(clip_similarity(img_emb, SLEEVE_VOCAB))],
        "pattern": PATTERN_VOCAB[np.argmax(clip_similarity(img_emb, PATTERN_VOCAB))],
        "fabric": FABRIC_VOCAB[np.argmax(clip_similarity(img_emb, FABRIC_VOCAB))],
        "fit": FIT_VOCAB[np.argmax(clip_similarity(img_emb, FIT_VOCAB))],
    }
