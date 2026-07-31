import os
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Ignore sklearn undefined metric warnings when a class is not present in early batches
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. COLORFUL LOGGING SYSTEM
# ==============================================================================
class Colors:
    """ANSI Escape Codes for Colorful Terminal Logs"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'

class Logger:
    @staticmethod
    def info(msg): print(f"{Colors.CYAN}[INFO]{Colors.RESET} {msg}")
    @staticmethod
    def success(msg): print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {msg}")
    @staticmethod
    def warn(msg): print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")
    @staticmethod
    def error(msg): print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}")
    @staticmethod
    def epoch(ep, total, eta=None): 
        eta_str = f" | {Colors.YELLOW}ETA: {eta}{Colors.RESET}" if eta else ""
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}=== Epoch [{ep}/{total}]{eta_str} ==={Colors.RESET}")
    @staticmethod
    def metric(name, val, is_best=False):
        star = f" {Colors.YELLOW}⭐ BEST{Colors.RESET}" if is_best else ""
        if isinstance(val, float):
            print(f"  ↳ {Colors.BLUE}{name}:{Colors.RESET} {val:.4f}{star}")
        else:
            print(f"  ↳ {Colors.BLUE}{name}:{Colors.RESET} {val}{star}")


# ==============================================================================
# 2. DYNAMIC EXPERIMENT FOLDER GENERATOR
# ==============================================================================
def create_experiment_dir(module_name: str, model_name: str, base_path: str = "core/models"):
    """
    Creates a dynamic folder structure for tracking experiments.
    Format: core/models/{module_name}/experiments/{model_name}_{YYYYMMDD_HHMMSS}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = f"{model_name}_{timestamp}"
    
    full_path = Path(base_path) / module_name / "experiments" / exp_folder
    full_path.mkdir(parents=True, exist_ok=True)
    
    Logger.success(f"Dynamic Experiment Folder Created: {full_path}")
    return str(full_path)


# ==============================================================================
# 3. ROOT TEMPLATE: S.O.T.A. BASE TRAINER
# ==============================================================================
class BaseTrainer:
    """
    State-Of-The-Art Universal Training Engine with:
    ✅ Real-Time TQDM Progress Bars & Automatic Graphing
    ✅ Fault-Tolerant Resuming (Load-shedding proof)
    ✅ Mixed Precision Training (AMP)
    ✅ Gradient Accumulation & Clipping
    ✅ Learning Rate Scheduler Support
    ✅ Multi-Metric Tracking (F1, Precision, Recall)
    """
    def __init__(self, 
                 model: nn.Module, 
                 train_loader, 
                 val_loader, 
                 criterion, 
                 optimizer, 
                 device: str, 
                 exp_dir: str,
                 scheduler=None,
                 patience: int = 5,
                 accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0):
        
        # 1. Core Setup
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.exp_dir = exp_dir
        
        # 2. Overfitting Prevention
        self.patience = patience
        self.epochs_no_improve = 0
        
        # 3. Advanced Engineering (AMP, Accumulation, Clipping)
        self.accumulation_steps = max(1, accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        if self.use_amp:
            Logger.info("Automatic Mixed Precision (AMP) is ENABLED 🚀")
        if self.accumulation_steps > 1:
            Logger.info(f"Gradient Accumulation is ENABLED (Steps: {self.accumulation_steps}) 🔋")
        if self.max_grad_norm > 0:
            Logger.info(f"Gradient Clipping is ENABLED (Max Norm: {self.max_grad_norm}) ✂️")
        
        # 4. History Tracking
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [],
            'val_f1': []
        }
        self.best_val_loss = float('inf')
        
    def _compute_metrics(self, all_targets, all_preds):
        """Calculates secondary metrics like F1, Precision, Recall."""
        if not HAS_SKLEARN or len(all_targets) == 0:
            return 0.0, 0.0, 0.0
            
        # Move to CPU for sklearn
        t_cpu = torch.cat(all_targets).cpu().numpy()
        p_cpu = torch.cat(all_preds).cpu().numpy()
        
        # Macro average handles class imbalance better
        f1 = f1_score(t_cpu, p_cpu, average='macro', zero_division=0)
        prec = precision_score(t_cpu, p_cpu, average='macro', zero_division=0)
        rec = recall_score(t_cpu, p_cpu, average='macro', zero_division=0)
        return f1, prec, rec

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}", leave=False, dynamic_ncols=True, 
                    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.RESET))
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 1. Forward Pass with AMP (Mixed Precision)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            # Normalize loss for Gradient Accumulation
            loss = loss / self.accumulation_steps
            
            # 2. Backward Pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # 3. Optimizer Step (with Accumulation & Clipping)
            if ((batch_idx + 1) % self.accumulation_steps == 0) or ((batch_idx + 1) == len(self.train_loader)):
                # Gradient Clipping
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Step Optimizer
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Metrics Logging
            running_loss += (loss.item() * self.accumulation_steps)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets).item()
            total += targets.size(0)
            
            # Update Progress Bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = correct / total
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc*100:.2f}%"})
            
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(self.train_loader)
        acc = correct / total
        return avg_loss, acc, epoch_time

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_targets = []
        all_preds = []
        
        pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", leave=False, dynamic_ncols=True,
                    bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.CYAN, Colors.RESET))
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets).item()
            total += targets.size(0)
            
            all_targets.append(targets)
            all_preds.append(preds)
            
            current_loss = running_loss / (batch_idx + 1)
            current_acc = correct / total
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc*100:.2f}%"})
            
        avg_loss = running_loss / len(self.val_loader)
        acc = correct / total
        
        # Calculate Multi-Metrics
        f1, prec, rec = self._compute_metrics(all_targets, all_preds)
        
        return avg_loss, acc, f1, prec, rec

    def generate_graphs(self):
        """Generates and saves Training curves for Loss and Accuracy"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. Loss Graph
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        loss_path = os.path.join(self.exp_dir, 'loss_curve.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Accuracy Graph
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, self.history['val_acc'], 'g-', label='Validation Accuracy', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        acc_path = os.path.join(self.exp_dir, 'accuracy_curve.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()

    def format_time(self, seconds: float) -> str:
        """Helper to format seconds into readable string"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"

    def fit(self, num_epochs: int, resume: bool = False):
        start_epoch = 1
        
        # 🛠️ FAULT-TOLERANT RESUME LOGIC (Now includes Scheduler & Scaler)
        if resume:
            latest_model_path = os.path.join(self.exp_dir, "latest_model.pt")
            if os.path.exists(latest_model_path):
                Logger.info(f"Resuming training from checkpoint: {latest_model_path}")
                checkpoint = torch.load(latest_model_path, map_location=self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if self.scaler and 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    
                start_epoch = checkpoint['epoch'] + 1
                self.best_val_loss = checkpoint['best_val_loss']
                self.history = checkpoint['history']
                self.epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                
                Logger.success(f"Resumed successfully! Continuing from Epoch {start_epoch}")
            else:
                Logger.warn("No checkpoint found to resume from. Starting from scratch.")
                
        if start_epoch > num_epochs:
            Logger.warn("Model has already finished training for the requested number of epochs.")
            return

        Logger.info(f"Starting Training for {num_epochs} Epochs on {str(self.device).upper()}...")
        
        total_training_start = time.time()
        epoch_times = []
        
        for epoch in range(start_epoch, num_epochs + 1):
            # Calculate ETA
            eta_str = "Calculating..."
            if len(epoch_times) > 0:
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = num_epochs - epoch + 1
                eta_str = self.format_time(avg_epoch_time * remaining_epochs)
            
            Logger.epoch(epoch, num_epochs, eta=eta_str)
            
            # Print Current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            Logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Train
            t_loss, t_acc, t_time = self.train_one_epoch(epoch)
            epoch_times.append(t_time)
            
            Logger.metric("Train Loss", t_loss)
            Logger.metric("Train Acc", f"{t_acc*100:.2f}%")
            
            # Validate
            v_loss, v_acc, v_f1, v_prec, v_rec = self.validate(epoch)
            is_best = v_loss < self.best_val_loss
            
            Logger.metric("Valid Loss", v_loss, is_best=is_best)
            Logger.metric("Valid Acc", f"{v_acc*100:.2f}%")
            if HAS_SKLEARN:
                Logger.metric("Valid F1", f"{v_f1:.4f}")
            
            print(f"  {Colors.BOLD}⏱️  Epoch Time: {self.format_time(t_time)}{Colors.RESET}")
            
            # Step Scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(v_loss)
                else:
                    self.scheduler.step()
            
            # Save metrics to history
            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['train_acc'].append(t_acc)
            self.history['val_acc'].append(v_acc)
            if HAS_SKLEARN:
                self.history['val_f1'].append(v_f1)
            
            # 📦 SAVE FULL STATE DICT FOR FAULT-TOLERANCE
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'history': self.history,
                'epochs_no_improve': self.epochs_no_improve
            }
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            if self.scaler:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            # Save Checkpoint if it's the best model
            if is_best:
                self.best_val_loss = v_loss
                self.epochs_no_improve = 0
                best_model_path = os.path.join(self.exp_dir, "best_model.pt")
                torch.save(checkpoint, best_model_path)
            else:
                self.epochs_no_improve += 1
                Logger.warn(f"Validation Loss did not improve. (Patience: {self.epochs_no_improve}/{self.patience})")
                
            # Always save the latest model (OVERWRITES PREVIOUS LATEST)
            latest_model_path = os.path.join(self.exp_dir, "latest_model.pt")
            torch.save(checkpoint, latest_model_path)
            
            # Save Logs to JSON dynamically
            with open(os.path.join(self.exp_dir, "training_logs.json"), "w") as f:
                json.dump(self.history, f, indent=4)
                
            # Generate Real-time Graphs
            self.generate_graphs()
            
            # 🛑 EARLY STOPPING TRIGGER
            if self.epochs_no_improve >= self.patience:
                Logger.error(f"EARLY STOPPING TRIGGERED! Model is starting to overfit. Halting at Epoch {epoch}.")
                break
                
        total_time = time.time() - total_training_start
        print("\n" + "="*50)
        Logger.success(f"Training Completed in {self.format_time(total_time)}!")
        Logger.info(f"All logs, weights, and graphs are saved in: {self.exp_dir}")
        print("="*50 + "\n")
