#!/usr/bin/env python3
"""
This script extracts speech features from the PARCZ database using VuVoPy,
reading WAV files organized in subdirectories by speaker code, in parallel,
with a real-time progress bar using concurrent.futures.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ===== User Configuration =====
WAV_DIR    = r"C:/Users/Richard Ladislav/Desktop/final countdown/DP-knihovna pro parametrizaci reci - kod/supershort_database"
OUTPUT_CSV = r"C:/Users/Richard Ladislav/Desktop/final countdown/DP-knihovna pro parametrizaci reci - kod/parcz_features.csv"
# Number of worker threads (-1 for cpu_count)
N_WORKERS  = os.cpu_count() or 1


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

    # 2) Parallel extraction with progress via as_completed
    records = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(process_entry, wav, sid, lbl, gdr) for wav, sid, lbl, gdr in entries]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting features", unit="file"):
            res = future.result()
            if res:
                records.append(res)

    # 3) Build DataFrame and save locally (avoid fsspec issues)
    df = pd.DataFrame.from_records(records)
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        df.to_csv(f, index=False)

    print(f"Feature extraction complete: {len(df)} records saved to {OUTPUT_CSV}.")


if __name__ == "__main__":
    main()
