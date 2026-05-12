import time
from datetime import timedelta
import torch
from torch.amp import autocast, GradScaler


class EarlyStopping:
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-7,
                 verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_score is None:
            self.best_score = val_loss
            return False
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return True
        else:
            self.best_score = val_loss
            self.counter = 0
            return False


def train_one_epoch(model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    clip_norm=1.0,
                    scaler: GradScaler = None):
    model.train()
    total_loss = 0.0
    total_steps = len(train_loader)

    for i, (xb, yb) in enumerate(train_loader, start=1):
        xb = xb.to(device, non_blocking=True).float().contiguous()
        yb = yb.to(device, non_blocking=True).float().contiguous()

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training with torch.amp (new unified API)
        if scaler is not None:
            with autocast(device_type=device.type):
                y_hat = model(xb)
                loss = criterion(y_hat, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            y_hat = model(xb)
            loss = criterion(y_hat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

        total_loss += loss.item() * xb.size(0)

        # Print step progress
        if i % max(1, total_steps // 10) == 0 or i == total_steps:
            print(f"  Train step {i}/{total_steps}")

    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device, scaler: GradScaler = None):
    model.eval()
    total_loss = 0.0
    total_steps = len(val_loader)

    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_loader, start=1):
            xb = xb.to(device, non_blocking=True).float().contiguous()
            yb = yb.to(device, non_blocking=True).float().contiguous()

            # Use autocast for validation too if using AMP
            if scaler is not None:
                with autocast(device_type=device.type):
                    y_hat = model(xb)
                    loss = criterion(y_hat, yb)
            else:
                y_hat = model(xb)
                loss = criterion(y_hat, yb)

            total_loss += loss.item() * xb.size(0)

            # Print step progress
            if i % max(1, total_steps // 10) == 0 or i == total_steps:
                print(f"  Val step {i}/{total_steps}")

    return total_loss / len(val_loader.dataset)


def fit(model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs: int,
        patience: int,
        model_save_path: str,
        clip_norm: float = 1.0,
        gui=None,
        epoch_end_callback=None,
        scheduler=None,
        use_amp: bool = False):
    """
    Train the model for the given number of epochs with early stopping.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler. If using OneCycleLR, call scheduler.step() after
        each batch. If using ReduceLROnPlateau, call scheduler.step(val_loss).
    use_amp : bool, default=False
        Enable automatic mixed precision training with torch.amp.
    epoch_end_callback : callable, optional
        Called at the end of every epoch after the GUI update.
        Signature: callback(epoch: int, model: nn.Module, device)
        Intended use: spectra preview train_plots, custom logging, etc.
    val_loader : DataLoader or callable
        Either a DataLoader object or a callable that returns one.
        If callable, it will be called once to get the validation loader.
    """

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-7)
    best_val_loss = float('inf')
    history = {"train": [], "val": []}

    # Create GradScaler if using AMP

    scaler = GradScaler() if use_amp else None

    # Resolve val_loader if it's a callable (lazy loading)
    if callable(val_loader):
        print("📦 Initializing validation loader...")
        val_loader = val_loader()
        print("✅ Validation loader ready")

    for epoch in range(1, epochs + 1):
        print(f"\nepoch {epoch}/{epochs}")
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, clip_norm, scaler
        )
        val_loss = validate(model, val_loader, criterion, device, scaler)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        epoch_time = time.time() - t0
        eta = (epochs - epoch) * epoch_time

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        status = "Training..."

        print(f"[{epoch:03d}] train {train_loss:.6e} | val {val_loss:.6e} | "
              f"time {timedelta(seconds=int(epoch_time))} | "
              f"ETA {timedelta(seconds=int(eta))}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save best model
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            early_stopping.counter = 0
            status = "Improved! Model saved."
        else:
            early_stopping(val_loss)
            status = f"No improvement ({early_stopping.counter}/{patience})"

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                if gui:
                    gui.set_status("Early stopping triggered", "red")
                # Fire callback one last time before breaking
                if epoch_end_callback is not None:
                    epoch_end_callback(epoch, model, device)
                break

        # GUI update
        if gui:
            gui.update(epoch, train_loss, val_loss, status)

        # Spectra / custom callback
        if epoch_end_callback is not None:
            epoch_end_callback(epoch, model, device)

    model.load_state_dict(torch.load(model_save_path, map_location=device))

    if gui:
        gui.set_status("Training completed!", "green")

    return history, best_val_loss
