import torch
import torch.nn as nn
import numpy as np
import time
import os

class Trainer:
    """
    Handles the training loop for the GCN-Transformer Autoencoder.
    """
    def __init__(self, model, optimizer, criterion, device, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader, all_state_embeddings, all_cluster_labels):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (signals, _, global_indices) in enumerate(train_loader):
            signals = signals.to(self.device)
            
            # Lookup cluster labels for this batch using global indices
            # all_cluster_labels is a numpy array [N_total]
            batch_cluster_labels = all_cluster_labels[global_indices]
            state_indices = torch.from_numpy(batch_cluster_labels).long().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward Pass: Conditioned on GCN embeddings
            output = self.model(
                src=signals, 
                state_indices=state_indices, 
                all_state_embeddings=all_state_embeddings
            )
            
            loss = self.criterion(output, signals)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def validate(self, val_loader, all_state_embeddings, all_cluster_labels):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for signals, _, global_indices in val_loader:
                signals = signals.to(self.device)
                
                batch_cluster_labels = all_cluster_labels[global_indices]
                state_indices = torch.from_numpy(batch_cluster_labels).long().to(self.device)
                
                output = self.model(
                    src=signals, 
                    state_indices=state_indices, 
                    all_state_embeddings=all_state_embeddings
                )
                
                loss = self.criterion(output, signals)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

    def fit(self, train_loader, val_loader, all_state_embeddings, all_cluster_labels):
        """
        Main training loop.
        """
        save_dir = self.config['training']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        patience = self.config['training']['patience']
        epochs = self.config['training']['epochs']
        patience_counter = 0
        
        print(f"   [Trainer] Starting training on {self.device}...")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader, all_state_embeddings, all_cluster_labels)
            val_loss = self.validate(val_loader, all_state_embeddings, all_cluster_labels)
            
            epoch_time = time.time() - start_time
            
            print(f"   Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s")
            
            # Save Best Model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.config
                }, save_path)
                print(f"   --> Saved best model to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   [Trainer] Early stopping triggered after {epoch} epochs.")
                    break