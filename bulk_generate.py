# bulk_generate.py

import os
import time
import pandas as pd
from tqdm import tqdm

from clip_extractor import extract_clip_attributes
from local_metadata import generate_local_metadata

IMAGE_FOLDER = r"C:\Users\ThinkPad\Desktop\project\dataset\images"
OUTPUT_CSV = "final_metadata.csv"
CHECKPOINT_CSV = "checkpoint_partial.csv"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_CSV):
        print("Resuming from checkpoint...")
        return pd.read_csv(CHECKPOINT_CSV)
    return pd.DataFrame()

def save_checkpoint(df):
    df.to_csv(CHECKPOINT_CSV, index=False)

def main():
    existing = load_checkpoint()
    processed_images = set(existing["image"]) if not existing.empty else set()

    all_images = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(("jpg", "jpeg", "png"))
    ]

    rows = []

    for img_name in tqdm(all_images, desc="Processing images"):
        if img_name in processed_images:
            continue

        img_path = os.path.join(IMAGE_FOLDER, img_name)

        try:
            attributes = extract_clip_attributes(img_path)
            metadata = generate_local_metadata(attributes)

            row = {
                "image": img_name,
                **attributes,
                "title": metadata["title"],
                "bullet_points": "; ".join(metadata["bullet_points"]),
                "description": metadata["description"],
                "style_summary": metadata["style_summary"],
                "seo_tags": "; ".join(metadata["seo_tags"])
            }

            rows.append(row)

            if len(rows) % 10 == 0:
                temp_df = pd.DataFrame(rows)
                full_df = pd.concat([existing, temp_df], ignore_index=True)
                save_checkpoint(full_df)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    final_df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
