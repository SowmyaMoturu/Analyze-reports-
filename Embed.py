# embeddings_extractor.py

import base64
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def decode_and_embed(base64_str):
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.cpu().numpy().flatten()

def process_embeddings(df, output_dir):
    embeddings = {}
    for idx, row in df[df['screenshot'].notnull()].iterrows():
        try:
            emb = decode_and_embed(row['screenshot'])
            embeddings[f"{row['scenario']}_{idx}"] = emb
        except Exception as e:
            print(f"Embedding error for {row['scenario']}: {e}")
    with open(os.path.join(output_dir, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

def compare_embeddings(output_dir, threshold=0.1):
    with open(os.path.join(output_dir, "embeddings.pkl"), "rb") as f:
        embeddings = pickle.load(f)

    keys = list(embeddings.keys())
    sims = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sim = cosine_similarity(embeddings[keys[i]], embeddings[keys[j]])
            if sim > (1 - threshold):
                sims.append((keys[i], keys[j], round(sim, 3)))

    if sims:
        print("Similar screenshots found:")
        for a, b, score in sims:
            print(f"  {a} vs {b} => Similarity: {score}")
    else:
        print("No similar screenshots found.")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
