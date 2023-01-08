import gc
import time
import torch
import numpy as np
from merlin.loader.torch import Loader
from sklearn.metrics import roc_auc_score
from transformers import get_linear_schedule_with_warmup

from training.losses import ClsLoss
from data.loader import define_loaders
from training.optim import define_optimizer
from params import WEIGHTS


def evaluate(model, val_loader, data_config, loss_config, loss_fct, use_fp16=False):
    model.eval()
    avg_val_loss = 0.0
    preds, preds_aux, tgts = [], [], []

    with torch.no_grad():
        for x, _ in val_loader:
            y = torch.cat([x[k] for k in data_config['target']], 1)
            x = torch.cat([x[k] for k in data_config['features']], 1)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred = model(x)
                loss = loss_fct(y_pred.detach(), y)

            avg_val_loss += loss / len(val_loader)

            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                y_pred = y_pred.softmax(-1)

            preds.append(y_pred.detach().cpu().numpy())
            tgts.append(y.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(tgts), avg_val_loss


def fit(
    model,
    train_dataset,
    val_dataset,
    data_config,
    loss_config,
    optimizer_config,
    epochs=1,
    verbose_eval=1,
    device="cuda",
    use_fp16=False,
    run=None,
    fold=0,
):
    """
    Training function.
    TODO
    Args:
    Returns:
    """
    scaler = torch.cuda.amp.GradScaler()

    optimizer = define_optimizer(
        optimizer_config["name"],
        model.parameters(),
        lr=optimizer_config["lr"],
        betas=optimizer_config["betas"],
    )

    loss_fct = ClsLoss(loss_config)

    train_loader = Loader(train_dataset, batch_size=data_config["batch_size"], shuffle=True)
    val_loader = Loader(val_dataset, batch_size=data_config["val_bs"], shuffle=False)

    # LR Scheduler
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(optimizer_config["warmup_prop"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    auc = 0
    step = 1
    preds = None
    avg_losses, aucs = [], []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for x, _ in train_loader:
            y = torch.cat([x[k] for k in data_config['target']], 1)
            x = torch.cat([x[k] for k in data_config['features']], 1)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred = model(x)
                loss = loss_fct(y_pred, y)

            scaler.scale(loss).backward()
            avg_losses.append(loss.detach().cpu())

            scaler.unscale_(optimizer)

            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()

            if scale == scaler.get_scale():
                scheduler.step()

            model.zero_grad(set_to_none=True)

            step += 1
            if (step % verbose_eval) == 0 or step - 1 == epochs * len(train_loader):
                if 0 <= epochs * len(train_loader) - step < verbose_eval:
                    continue

                if val_loader is not None:
                    preds, tgts, avg_val_loss = evaluate(
                        model, val_loader, data_config, loss_config, loss_fct, use_fp16=use_fp16
                    )
                    aucs = [roc_auc_score(tgts[:, i], preds[:, i]) for i in range(preds.shape[-1])]
                    auc = np.average(aucs, weights=WEIGHTS)

                dt = time.time() - start_time
                lr = scheduler.get_last_lr()[0]

                s = f"Epoch {epoch:02d}/{epochs:02d} (step {step:04d})\t"
                s = s + f"lr={lr:.1e}\t t={dt:.0f}s \t loss={np.mean(avg_losses):.3f}"
                s = s + f"\t val_loss={avg_val_loss:.3f}" if avg_val_loss else s
                s = s + f"\t auc={auc:.4f}  " if auc else s
                s = s + f"{tuple(list(np.round(aucs, 3)))}" if len(aucs) else s
                print(s)

                if run is not None:
                    run[f"fold_{fold}/train/epoch"].log(epoch, step=step)
                    run[f"fold_{fold}/train/loss"].log(np.mean(avg_losses), step=step)
                    run[f"fold_{fold}/val/loss"].log(avg_val_loss, step=step)
                    run[f"fold_{fold}/val/auc"].log(auc, step=step)

                start_time = time.time()
                avg_losses = []
                model.train()

        del (x, y_pred, loss, y)
        torch.cuda.empty_cache()
        gc.collect()
                

    del (train_loader, val_loader, optimizer)
    torch.cuda.empty_cache()
    gc.collect()

    return preds