import os
import librosa
import numpy as np
import argparse


def preprocess_audio(input_path, output_path, sr=16000):
    y, _ = librosa.load(input_path, sr=sr, mono=True)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y)
    np.save(output_path, y)


def process_directory(input_dir, output_dir, sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            in_path = os.path.join(input_dir, fname)
            out_name = os.path.splitext(fname)[0] + '.npy'
            out_path = os.path.join(output_dir, out_name)
            print(f"Processing {in_path} -> {out_path}")
            preprocess_audio(in_path, out_path, sr=sr)


def main():
    parser = argparse.ArgumentParser(description="Preprocess beatboxing audio files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with raw audio files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed files')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate (default: 16000)')
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, sr=args.sr)


if __name__ == "__main__":
    main()
