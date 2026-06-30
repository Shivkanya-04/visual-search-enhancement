import os
import json
import numpy as np
import torch
import faiss
from tqdm import tqdm
from PIL import Image
import open_clip

# --- CONFIG ---
IMAGE_FOLDER = r"C:\Users\ThinkPad\Desktop\Visual Search Enhancer\dataset\images"
EMB_PATH = "embeddings.npy"
FILENAMES_PATH = "filenames.json"
FAISS_INDEX_PATH = "faiss_index.bin"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(device)

def encode_image(image_path):
    """Return CLIP embedding for a single image."""
    img = Image.open(image_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def build_embeddings():
    """Generate embeddings + filenames list, save as .npy + .json."""
    images = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(("jpg", "jpeg", "png"))
    ]

    all_embs = []
    for img_name in tqdm(images, desc="Encoding images"):
        path = os.path.join(IMAGE_FOLDER, img_name)
        emb = encode_image(path)
        all_embs.append(emb)

    # Save embeddings + filenames
    np.save(EMB_PATH, np.vstack(all_embs))
    with open(FILENAMES_PATH, "w") as f:
        json.dump(images, f)

    print("\nSaved embeddings to:", EMB_PATH)
    print("Saved filenames to:", FILENAMES_PATH)

def build_faiss_index():
    """Build FAISS index and save it."""
    embeddings = np.load(EMB_PATH).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity = dot-product on normalized vectors
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index saved to:", FAISS_INDEX_PATH)

def load_index():
    """Load FAISS + mapping files."""
    index = faiss.read_index(FAISS_INDEX_PATH)
    embeddings = np.load(EMB_PATH)
    with open(FILENAMES_PATH, "r") as f:
        filenames = json.load(f)
    return index, filenames, embeddings

def search_similar(image_path, top_k=5):
    """Encode query → return top-k similar images from dataset."""
    index, filenames, embeddings = load_index()

    query_emb = encode_image(image_path).astype("float32").reshape(1, -1)
    scores, indices = index.search(query_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "filename": filenames[idx],
            "score": float(score)
        })
    return results

if __name__ == "__main__":
    print("Step 1: Generating embeddings...")
    build_embeddings()

    print("\nStep 2: Building FAISS index...")
    build_faiss_index()

    print("\nDone! Use search_similar(image_path) to find similar products.")
