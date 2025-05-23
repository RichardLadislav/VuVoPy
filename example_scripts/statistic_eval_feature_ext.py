#!/usr/bin/env python3
"""
This script extracts speech features from the PARCZ database using VuVoPy,
reading WAV files organized in subdirectories by speaker code, in parallel,
with a real-time progress bar using concurrent.futures and per-task timeouts.

Directory structure:
  PARCZ_complet/recordings/
      K1XXX/  # healthy female subjects
      K2XXX/  # healthy male subjects
      P1XXX/  # Parkinson's female subjects
      P2XXX/  # Parkinson's male subjects

Columns in the output CSV:
  - filename
  - speaker_id (e.g. 'K1234')
  - label (0 = healthy, 1 = Parkinson's)
  - gender ('F' or 'M')
  - durmad, durmed, duv, hnr, jitter, mpt, ppr,
    relF0SD, relF1SD, relF2SD, relSEOSD, shimmer, spir

Usage:
    python extract_parcz_features.py

Before running, update the WAV_DIR and OUTPUT_CSV paths below.
"""

import os
import VuVoPy as vp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm

# ===== User Configuration =====
WAV_DIR    = r"user_wav_dir_path"  
#WAV_DIR    = r"C:/Users/Richard Ladislav/Desktop/final countdown/DP-knihovna pro parametrizaci reci - kod/PARCZ_complet"
OUTPUT_CSV = r"user_output_csv_path"
# Number of worker threads
N_WORKERS  = min(os.cpu_count() or 1, 8)
# Timeout per file (seconds)
TIMEOUT    = 12


def process_entry(wav_path, speaker_id, label, gender):
    try:
        return {
            "filename":    os.path.basename(wav_path),
            "speaker_id":  speaker_id,
            "label":       label,
            "gender":      gender,
            "durmad":      vp.durmad(wav_path),
            "durmed":      vp.durmed(wav_path),
            "duv":         vp.duv(wav_path),
            "hnr":         vp.hnr(wav_path),
            "jitter":      vp.jitterPPQ(wav_path),
            "mpt":         vp.mpt(wav_path),
            "ppr":         vp.ppr(wav_path),
            "relF0SD":     vp.relF0SD(wav_path),
            "relF1SD":     vp.relF1SD(wav_path),
            "relF2SD":     vp.relF2SD(wav_path),
            "relSEOSD":    vp.relSEOSD(wav_path),
            "shimmer":     vp.shimmerAPQ(wav_path),
            "spir":        vp.spir(wav_path),
        }
    except Exception as e:
        print(f"Failed on {wav_path}: {e}")
        return None


def main():
    # 1) Gather all WAV entries
    entries = []
    for root, _, files in os.walk(WAV_DIR):
        for filename in files:
            if not filename.lower().endswith('.wav'):
                continue
            wav_path = os.path.join(root, filename)
            speaker_id = os.path.basename(root)
            prefix = speaker_id[0].upper()
            if prefix == 'P':
                label = 1
            elif prefix == 'K':
                label = 0
            else:
                continue
            gender = 'F' if len(speaker_id) > 1 and speaker_id[1] == '1' else (
                     'M' if len(speaker_id) > 1 and speaker_id[1] == '2' else None)
            entries.append((wav_path, speaker_id, label, gender))

    # 2) Parallel extraction with progress via as_completed and timeout
    records = []
    # Map futures to file paths for error handling
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        future_to_path = {
            executor.submit(process_entry, wav, sid, lbl, gdr): wav
            for wav, sid, lbl, gdr in entries
        }
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Extracting features", unit="file"):
            wav_path = future_to_path[future]
            try:
                res = future.result(timeout=TIMEOUT)
            except TimeoutError:
                print(f"Timeout processing {wav_path}")
                continue
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                continue
            if res:
                records.append(res)

    # 3) Build DataFrame and save locally
    df = pd.DataFrame.from_records(records)
    out_dir = os.path.dirname(OUTPUT_CSV)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Now safe to open
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        df.to_csv(f, index=False)

    print(f"Feature extraction complete: {len(df)} records saved to {OUTPUT_CSV}.")


if __name__ == "__main__":
    main()
