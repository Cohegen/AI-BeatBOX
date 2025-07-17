import torch
import numpy as np
import soundfile as sf
from wavenet import WaveNet

SAMPLE_RATE = 16000
GEN_LENGTH = 16000 * 5  # 5 seconds
CHECKPOINT_PATH = 'wavenet_checkpoint.pth'

if __name__ == "__main__":
    # Load model
    model = WaveNet(in_channels=1, channels=32, kernel_size=2, num_blocks=2, num_layers=4)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Start with a seed (zeros)
    seed = torch.zeros(1, 1, 1)  # (batch, channel, length)
    generated = []
    input_seq = seed
    with torch.no_grad():
        for i in range(GEN_LENGTH):
            out = model(input_seq)
            next_sample = out[:, :, -1].unsqueeze(-1)  # Take last predicted sample
            generated.append(next_sample.item())
            # Append to input_seq for next step
            input_seq = torch.cat([input_seq, next_sample], dim=2)
            # For efficiency, you can limit input_seq to the last 1024 samples or so
            if (i+1) % 1000 == 0:
                print(f"Generated {i+1} samples...")

    generated = np.array(generated, dtype=np.float32)
    sf.write('generated.wav', generated, SAMPLE_RATE)
    print("Saved generated audio to 'generated.wav'") 