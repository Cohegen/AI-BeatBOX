# AI Beatboxer

Generate beatboxing sounds with deep learning! This project trains a WaveNet-based vocoder on beatboxing audio clips and generates new, realistic beatbox audio samples.

---

## 🚀 Features
- **End-to-end pipeline:** Preprocessing, training, and audio generation
- **WaveNet model:** Powerful generative model for raw audio
- **Checkpointing:** Resume training anytime
- **Colab notebook:** Run everything in the cloud

---

## 📦 Project Structure
```
beatboxerAI/
├── data/
│   ├── raw/                # Raw audio files (WAV, MP3, etc.)
│   └── processed/          # Preprocessed .npy files
├── src/
│   ├── preprocess.py       # Audio preprocessing
│   ├── dataset.py          # PyTorch dataset loader
│   ├── wavenet.py          # WaveNet model
│   ├── train.py            # Training script
│   └── generate.py         # Audio generation script
├── notebooks/
│   └── BeatboxerAI_Generation.ipynb  # Colab notebook
├── requirements.txt
├── README.md
└── wavenet_checkpoint.pth  # (Generated) Model checkpoint
```

---

## 🛠️ Setup
1. **Clone the repo:**
   ```sh
   git clone https://github.com/yourusername/beatboxerAI.git
   cd beatboxerAI
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Prepare your data:**
   - Place your raw audio files in `data/raw/beatboxset1/` (or another subfolder).

---

## ⚙️ Usage
### 1. Preprocess Audio
```sh
python src/preprocess.py --input_dir data/raw/beatboxset1 --output_dir data/processed/beatboxset1
```

### 2. Train the Model
```sh
python src/train.py
```
- Model checkpoints are saved automatically after each epoch.

### 3. Generate Audio
```sh
python src/generate.py
```
- The generated audio will be saved as `generated.wav`.
- To generate longer audio, edit `GEN_LENGTH` in `src/generate.py`.

---

## 🟢 Run on Google Colab
- Use the notebook: `notebooks/BeatboxerAI_Generation.ipynb`
- Upload your `wavenet.py` and `wavenet_checkpoint.pth` or mount Google Drive.
- Follow the notebook cells to generate and listen to audio in the browser.

---

## 🤝 Contributing
Contributions are welcome! Feel free to open issues or pull requests for improvements, new features, or bug fixes.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements
- [WaveNet: A Generative Model for Raw Audio](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)
- [PyTorch](https://pytorch.org/)
- [Librosa](https://librosa.org/)

---

## 🎧 Example Results
*Add your generated audio samples or spectrogram images here!*
