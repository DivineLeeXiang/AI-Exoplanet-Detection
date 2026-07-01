import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Scsv.gzet random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Define your list of URLs here
file_urls = [
    "https://archive.stsci.edu/missions/tess/catalogs/tic_v82/tic_dec90_00S__88_00S.csv.gz",
    "https://archive.stsci.edu/missions/tess/catalogs/tic_v82/tic_dec88_00S__86_00S.csv.gz"
    # Add as many links as you need here
]

# 2. Iterate through the list
for url in file_urls:
    # --- PUT YOUR EXISTING CODE HERE ---
    # Make sure to replace the hardcoded URL in your original code 
    # with the variable 'url' so it uses the current link in the loop.
    print(f"Processing: {url}")
    # Example: df = pd.read_csv(url) 
    # -----------------------------------


# ==========================================
# 1. PREPROCESSING HELPERS
# ==========================================
def clean_and_normalize(flux_array):
    """
    Cleans individual light curves by applying robust median scaling 
    and outlier clipping.
    """
    # Normalize by median
    median = np.median(flux_array)
    normalized = flux_array / median
    
    # Simple Outlier Clipping (Clip values that are extremely noisy)
    std = np.std(normalized)
    mean = np.mean(normalized)
    clipped = np.clip(normalized, mean - 3*std, mean + 3*std)
    
    return clipped

# ==========================================
# 2. DUMMY DATA GENERATOR (For instant testing!)
# ==========================================
def generate_synthetic_dataset(num_samples=1000, seq_len=400):
    """
    Generates a mock Kaggle-style Kepler dataset.
    Class 1: True Exoplanets (flat baseline with a periodic U-shaped dip)
    Class 0: Noise / Non-transits (random walk or stellar variability)
    """
    X = []
    y = []
    t = np.linspace(0, 1, seq_len)
    
    for _ in range(num_samples):
        # 1. Generate Gaussian baseline noise
        noise = np.random.normal(0, 0.02, seq_len)
        
        # 2. Determine class
        is_planet = np.random.choice([0, 1], p=[0.7, 0.3]) # Imbalanced class
        
        if is_planet == 1:
            # Create a physical transit dip (U-shape block)
            flux = np.ones(seq_len)
            dip_start = np.random.randint(50, seq_len - 150)
            dip_duration = np.random.randint(30, 80)
            
            # Form a smooth rounded dip
            depth = np.random.uniform(0.04, 0.12)
            for j in range(dip_duration):
                idx = dip_start + j
                # Formula to round the edges of the transit (limb-darkening effect)
                phase = (j / dip_duration) * np.pi
                flux[idx] -= depth * np.sin(phase)
            
            flux += noise
            flux = clean_and_normalize(flux)
            X.append(flux)
            y.append(1)
        else:
            # Just noise plus mock stellar rotation patterns (sine wave)
            stellar_rotation = 0.03 * np.sin(2 * np.pi * t * np.random.uniform(1, 5))
            flux = np.ones(seq_len) + stellar_rotation + noise
            flux = clean_and_normalize(flux)
            X.append(flux)
            y.append(0)
            
    return np.array(X), np.array(y)

# ==========================================
# 3. PYTORCH DATASET SETUP
# ==========================================
class LightCurveDataset(Dataset):
    """
    Converts Scikit-learn/Numpy arrays into structured PyTorch Tensors.
    1D CNN expects shape: (Batch_Size, Channels, Sequence_Length)
    """
    def __init__(self, X, y):
        # Insert a channel dimension: shape becomes (N, 1, seq_len)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        # Binary target: shape becomes (N, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 4. PYTORCH 1D CNN MODEL ARCHITECTURE
# ==========================================
class Exoplanet1DCNN(nn.Module):
    def __init__(self, seq_len):
        super(Exoplanet1DCNN, self).__init__()
        
        # Conv layer 1: scans the timeline looking for short edges/slopes
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, stride=1, padding=4)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Conv layer 2: captures combinations of slopes (the dip shape)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # After two max pooling layers (kernel_size=2), sequence length is divided by 4
        reduced_len = seq_len // 4
        
        # Fully connected decision network
        self.fc1 = nn.Linear(32 * reduced_len, 64)
        self.fc2 = nn.Linear(64, 1) # Single output for binary probability
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Prevents memorizing noise (overfitting)

    def forward(self, x):
        # Feature Extraction
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        
        # Flattening representation
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(self.relu(self.fc1(x)))
        logits = self.fc2(x) # Raw prediction scores
        return logits

# ==========================================
# 5. MODEL TRAINING ENGINE
# ==========================================
def train_model(model, train_loader, val_loader, class_weight, epochs=15):
    # Use BCEWithLogitsLoss to handle raw scores directly with numerical stability
    # Use pos_weight to penalize misclassifying the rare exoplanet class
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Beginning Training Phase...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad() # Reset memory
            outputs = model(inputs) # Forward propagation
            loss = criterion(outputs, targets) # calculate errors
            loss.backward() # Backpropagation (calculate gradients)
            optimizer.step() # Update model weights
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Epoch Validation Check
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ==========================================
# 6. PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Obtain data (Replace this with real CSV files once downloaded)
    # Target dataset dimension: sequence length = 400 bins
    SEQ_LENGTH = 400
    X_raw, y_raw = generate_synthetic_dataset(num_samples=1200, seq_len=SEQ_LENGTH)
    
    # 2. Split into Train / Test sets using Scikit-Learn
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    
    # Split training set further to generate a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # 3. Fit standard scaler using Scikit-Learn (keeps all inputs at a uniform scale)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 4. Handle class imbalance (exoplanets are rare!) [cite: 4]
    # Class weights tell the loss function to care more about the rare exoplanet signals
    num_non_planets = np.sum(y_train == 0)
    num_planets = np.sum(y_train == 1)
    imbalance_multiplier = num_non_planets / num_planets
    print(f"Dataset summary: Planets = {num_planets}, Non-Planets = {num_non_planets}")
    print(f"Loss penalty multiplier for rare class: {imbalance_multiplier:.2f}")
    
    # 5. Pack into PyTorch DataLoaders
    train_dataset = LightCurveDataset(X_train, y_train)
    val_dataset = LightCurveDataset(X_val, y_val)
    test_dataset = LightCurveDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 6. Initialize the Model
    model = Exoplanet1DCNN(seq_len=SEQ_LENGTH)
    
    # 7. Train model
    train_model(model, train_loader, val_loader, class_weight=imbalance_multiplier, epochs=15)
    
    # 8. Evaluation using Scikit-learn metrics
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            logits = model(inputs)
            # Standard sigmoid mapping to calculate probability [0.0, 1.0]
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float() # Threshold classification
            
            all_preds.extend(preds.numpy().flatten())
            all_probs.extend(probs.numpy().flatten())
            
    # Calculate performance scores
    print("\n" + "="*40)
    print("EVALUATION SCORES (TEST DATASET)")
    print("="*40)
    print(classification_report(y_test, all_preds, target_names=["Non-Planet", "Planet"]))
    
    auc = roc_auc_score(y_test, all_probs)
    print(f"ROC-AUC Performance: {auc:.4f}")
    
    # 9. Visualizing a prediction sample
    sample_idx = np.random.randint(0, len(X_test))
    raw_flux = X_test[sample_idx]
    true_label = y_test[sample_idx]
    predicted_prob = all_probs[sample_idx]
    
    plt.figure(figsize=(10, 4))
    plt.plot(raw_flux, color='black', label='Normalized Light Curve')
    plt.title(f"Target Stellar Object | True Class: {true_label} | AI Prediction Score: {predicted_prob:.2%}")
    plt.xlabel("Phase Bin")
    plt.ylabel("Relative Standardized Flux")
    plt.legend()
    plt.show()