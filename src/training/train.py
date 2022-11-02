import gc
import time
import torch
import numpy as np

from torchcontrib.optim import SWA
from transformers import get_linear_schedule_with_warmup

from utils.metric import compute_metric

from data.loader import define_loaders
from training.losses import NBMELoss
from training.optim import custom_params, define_optimizer, trim_tensors, AWP


def evaluate(
    model, val_loader, data_config, loss_config, loss_fct
):
    model.eval()
    avg_val_loss = 0.
    preds = []
    with torch.no_grad():
        for data in val_loader:
            y_batch = data["target"]
            ids, token_type_ids = trim_tensors(
                [data["ids"], data["token_type_ids"]],
                pad_token=data_config['pad_token']
            )

            y_pred = model(ids.cuda(), token_type_ids.cuda())

            loss = loss_fct(y_pred.detach(), y_batch.cuda()).mean()
            avg_val_loss += loss / len(val_loader)

            if loss_config['activation'] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config['activation'] == "softmax":
                y_pred = y_pred.softmax(-1)

            preds.append(y_pred.detach().cpu().numpy())

    return np.concatenate(preds), avg_val_loss


def fit(
    model,
    train_dataset,
    val_dataset,
    data_config,
    loss_config,
    optimizer_config,
    epochs=5,
    acc_steps=1,
    verbose_eval=1,
    device="cuda",
    use_fp16=False,
    gradient_checkpointing=False,
):
    """
    Training functiong.
    TODO

    Args:

    Returns:
    """
    scaler = torch.cuda.amp.GradScaler()

    if gradient_checkpointing:
        model.transformer.gradient_checkpointing_enable()

    opt_params = custom_params(
        model,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
        lr_transfo=optimizer_config["lr_transfo"],
        lr_decay=optimizer_config["lr_decay"],
    )
    optimizer = define_optimizer(
        optimizer_config["name"],
        opt_params,
        lr=optimizer_config["lr"],
        betas=optimizer_config["betas"],
    )
    optimizer.zero_grad()

    if optimizer_config['use_swa']:
        optimizer = SWA(
            optimizer,
            swa_start=optimizer_config['swa_start'],
            swa_freq=optimizer_config['swa_freq']
        )

    loss_fct = NBMELoss(loss_config, device=device)

    if optimizer_config['use_awp']:
        awp = AWP(model,
            optimizer,
            loss_fct,
            adv_lr=optimizer_config['awp_lr'],
            adv_eps=optimizer_config['awp_eps'],
            start_step=optimizer_config['awp_start_step'],
            scaler=scaler,
        )
    
    train_loader, val_loader = define_loaders(
        train_dataset,
        val_dataset,
        batch_size=data_config["batch_size"],
        val_bs=data_config["val_bs"],
        use_len_sampler=data_config["use_len_sampler"],
        pad_token=data_config["pad_token"],
    )

    # LR Scheduler
    num_training_steps = epochs * len(train_loader) // acc_steps
    num_warmup_steps = int(optimizer_config["warmup_prop"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    step = 1
    avg_losses = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for data in train_loader:
            ids, token_type_ids = trim_tensors(
                [data["ids"], data["token_type_ids"]],
                pad_token=data_config['pad_token']
            )
            y_batch = data["target"]

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred = model(ids.cuda(), token_type_ids.cuda())
                loss = loss_fct(y_pred, y_batch.cuda()).mean() / acc_steps

            scaler.scale(loss).backward()
            avg_losses.append(loss.item() * acc_steps)

            if optimizer_config['use_awp'] and (step % optimizer_config['awp_period']) == 0:
                awp.attack_backward(ids, token_type_ids, y_batch, step, use_fp16=use_fp16)

            if step % acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    optimizer_config["max_grad_norm"],
                    error_if_nonfinite=False
                )

                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()

                if scale == scaler.get_scale():
                    scheduler.step()

                for param in model.parameters():
                    param.grad = None

            step += 1
            if step % (verbose_eval * acc_steps) == 0 or step - 1 == epochs * len(train_loader):
                if 0 <= epochs * len(train_loader) - step < verbose_eval * acc_steps:
                    continue

                if step // acc_steps > optimizer_config['swa_start'] + optimizer_config['swa_freq'] and optimizer_config['use_swa']:
                    optimizer.swap_swa_sgd()
                
                preds, avg_val_loss = evaluate(
                    model, val_loader, data_config, loss_config, loss_fct
                )

                if step // acc_steps > optimizer_config['swa_start'] + optimizer_config['swa_freq'] and optimizer_config['use_swa']:
                    optimizer.swap_swa_sgd()

                score= compute_metric(preds, val_dataset.targets)

                dt = time.time() - start_time
                lr = scheduler.get_last_lr()[0]

                s = f"Epoch {epoch:02d}/{epochs:02d}  (step {step // acc_steps:04d})\t"
                s = s + f"lr={lr:.1e}\t t={dt:.0f}s\t loss={np.mean(avg_losses):.3f}"
                s = s + f"\t val_loss={avg_val_loss:.3f}" if avg_val_loss else s
                s = s + f"\t score={score:.3f}" if score else s
                print(s)

                start_time = time.time()
                avg_losses = []
                model.train()

    if optimizer_config['use_swa']:
        optimizer.swap_swa_sgd()

    del (train_loader, val_loader, optimizer)
    torch.cuda.empty_cache()
    gc.collect()

    return preds
