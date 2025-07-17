import torch
from torch.utils.data import DataLoader
from dataset import BeatboxDataset
from wavenet import WaveNet
import torch.nn as nn
import torch.optim as optim
import os

# Custom collate function to truncate/pad waveforms to 16,000 samples
FIXED_LENGTH = 16000

def collate_fn(batch):
    waveforms, filenames = zip(*batch)
    processed = []
    for w in waveforms:
        if len(w) >= FIXED_LENGTH:
            processed.append(torch.tensor(w[:FIXED_LENGTH], dtype=torch.float32))
        else:
            pad = torch.zeros(FIXED_LENGTH, dtype=torch.float32)
            pad[:len(w)] = torch.tensor(w, dtype=torch.float32)
            processed.append(pad)
    waveforms_tensor = torch.stack(processed)
    return waveforms_tensor, filenames

if __name__ == "__main__":
    dataset = BeatboxDataset('data/processed/beatboxset1')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # Instantiate the model
    model = WaveNet(in_channels=1, channels=32, kernel_size=2, num_blocks=2, num_layers=4)
    model.train()

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    checkpoint_path = 'wavenet_checkpoint.pth'
    start_epoch = 0
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")

    num_epochs = 20
    max_grad_norm = 1.0  # For gradient clipping

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        for batch_idx, (waveforms, filenames) in enumerate(dataloader):
            # waveforms: (batch, samples)
            # Add channel dimension: (batch, 1, samples)
            waveforms = waveforms.unsqueeze(1)
            # For demonstration, predict the next sample (shifted by 1)
            inputs = waveforms[:, :, :-1]
            targets = waveforms[:, :, 1:]
            outputs = model(inputs)
            # Match output and target shapes
            outputs = outputs[..., :targets.shape[-1]]
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Epoch {epoch+1} Batch {batch_idx+1}, Loss: {loss.item():.6f}")
        avg_loss = epoch_loss / (batch_idx + 1)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}\n")
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}\n")
