import argparse
import os
import json
import numpy as np

from clip_extractor import extract_clip_attributes
from local_metadata import generate_local_metadata
from visual_search import search_similar

def print_header(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def run_pipeline(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

  
    print_header("STEP 1: Extracting Visual Attributes (CLIP)")
    attributes = extract_clip_attributes(image_path)
    print(json.dumps(attributes, indent=4))

   
    print_header("STEP 2: Generating Product Metadata (Local LLM)")
    metadata = generate_local_metadata(attributes)
    print(json.dumps(metadata, indent=4))

    
    print_header("STEP 3: Finding Visually Similar Products (FAISS)")
    similar_items = search_similar(image_path, top_k=5)

    for i, item in enumerate(similar_items, start=1):
        print(f"{i}. {item['filename']}   (score: {item['score']:.4f})")

    
    print_header("PIPELINE COMPLETE")
    print("All steps finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full CLIP + LLM + FAISS pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    run_pipeline(args.image)
