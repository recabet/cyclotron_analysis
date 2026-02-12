import time
from datetime import timedelta
import torch



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
                    clip_norm=1.0):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True).float().contiguous()
        yb = yb.to(device, non_blocking=True).float().contiguous()
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(xb)
        loss = criterion(y_hat, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True).float().contiguous()
            yb = yb.to(device, non_blocking=True).float().contiguous()
            y_hat = model(xb)
            loss = criterion(y_hat, yb)
            total_loss += loss.item() * xb.size(0)
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
        gui=None):

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-7)
    best_val_loss = float('inf')
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, clip_norm
        )
        val_loss = validate(model, val_loader, criterion, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = time.time() - t0
        eta = (epochs - epoch) * epoch_time

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        status = "Training..."

        print(f"[{epoch:03d}] train {train_loss:.6e} | val {val_loss:.6e} | "
              f"time {timedelta(seconds=int(epoch_time))} | "
              f"ETA {timedelta(seconds=int(eta))}")

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
                break

        # GUI update
        if gui:
            gui.update(epoch, train_loss, val_loss, status)

    model.load_state_dict(torch.load(model_save_path, map_location=device))

    if gui:
        gui.set_status("Training completed!", "green")

    return history, best_val_loss
