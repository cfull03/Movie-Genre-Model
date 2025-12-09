"""
Example script demonstrating PyTorch preprocessing for movie genre classification.

This script shows how to:
1. Prepare data for PyTorch training
2. Create DataLoaders
3. Save/load preprocessors
4. Iterate through batches

Run this script to verify the preprocessing pipeline works correctly.
"""

from pathlib import Path
import torch

from descriptions.modeling.pytorch_preprocess import (
    prepare_pytorch_data,
    create_dataloaders,
    save_pytorch_preprocessors,
    load_pytorch_preprocessors,
)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Prepare data
print("\n" + "=" * 70)
print("Step 1: Preparing PyTorch data")
print("=" * 70)

train_dataset, val_dataset, test_dataset, vectorizer, mlb = prepare_pytorch_data(
    data_path=Path("data/interim/cleaned_movies.csv"),
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    device=device,
)

# Create DataLoaders
print("\n" + "=" * 70)
print("Step 2: Creating DataLoaders")
print("=" * 70)

train_loader, val_loader, test_loader = create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    shuffle_train=True,
    num_workers=0,  # Set to 0 for Windows compatibility
    pin_memory=(device == "cuda"),
)

# Test iterating through batches
print("\n" + "=" * 70)
print("Step 3: Testing batch iteration")
print("=" * 70)

print("\nTraining batches:")
for batch_idx, (features, labels) in enumerate(train_loader):
    print(
        f"  Batch {batch_idx + 1}: features shape={features.shape}, "
        f"labels shape={labels.shape}"
    )
    if batch_idx >= 2:  # Only show first 3 batches
        print("  ...")
        break

print("\nValidation batches:")
for batch_idx, (features, labels) in enumerate(val_loader):
    print(
        f"  Batch {batch_idx + 1}: features shape={features.shape}, "
        f"labels shape={labels.shape}"
    )
    if batch_idx >= 2:  # Only show first 3 batches
        print("  ...")
        break

# Save preprocessors
print("\n" + "=" * 70)
print("Step 4: Saving preprocessors")
print("=" * 70)

save_pytorch_preprocessors(vectorizer, mlb)
print("✓ Preprocessors saved successfully")

# Test loading preprocessors
print("\n" + "=" * 70)
print("Step 5: Testing preprocessor loading")
print("=" * 70)

loaded_vectorizer, loaded_mlb = load_pytorch_preprocessors()
print(f"✓ Preprocessors loaded: {loaded_vectorizer.max_features} features, "
      f"{len(loaded_mlb.classes_)} labels")

# Display dataset statistics
print("\n" + "=" * 70)
print("Dataset Statistics")
print("=" * 70)
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Feature dimension: {train_dataset.features.shape[1]}")
print(f"Label dimension: {train_dataset.labels.shape[1]}")
print(f"Genre classes: {list(mlb.classes_)[:10]}..." if len(mlb.classes_) > 10 else f"Genre classes: {list(mlb.classes_)}")

print("\n" + "=" * 70)
print("✓ PyTorch preprocessing pipeline verified successfully!")
print("=" * 70)















