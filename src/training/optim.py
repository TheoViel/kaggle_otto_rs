import torch


def define_optimizer(name, params, lr=1e-3, betas=(0.9, 0.999)):
    """
    Defines the loss function associated to the name.
    Supports optimizers from torch.nn.
    Args:
        name (str): Optimizer name.
        params (torch parameters): Model parameters
        lr (float, optional): Learning rate. Defaults to 1e-3.
    Raises:
        NotImplementedError: Specified optimizer name is not supported.
    Returns:
        torch optimizer: Optimizer
    """
    try:
        optimizer = getattr(torch.optim, name)(params, lr=lr, betas=betas)
    except AttributeError:
        raise NotImplementedError

    return optimizer


def custom_params(model, weight_decay=0, lr=1e-3, lr_transfo=3e-5, lr_decay=1):
    """
    Custom parameters for Bert Models to handle weight decay and differentiated learning rates.

    Args:
        model (torch model]): Transformer model
        weight_decay (int, optional): Weight decay. Defaults to 0.
        lr (float, optional): LR of layers not belonging to the transformer. Defaults to 1e-3.
        lr_transfo (float, optional): LR of the last layer of the transformer. Defaults to 3e-5.
        lr_decay (float, optional): Factor to multiply lr_transfo when going deeper. Defaults to 1.

    Returns:
        list: Optimizer params.
    """

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    transformer_name = "transformer"
    opt_params = []

    if not any([n in model.name for n in ["albert", "funnel", "distil"]]):
        try:
            nb_blocks = len(model.transformer.encoder.layer)
        except AttributeError:
            nb_blocks = len(model.transformer.encoder.layers)

        for n, p in model.named_parameters():
            wd = 0 if any(nd in n for nd in no_decay) else weight_decay

            if transformer_name in n and "pooler" not in n:
                lr_ = None
                for i in range(nb_blocks):  # for bert base
                    if f"layer.{i}." in n or f"layers.{i}." in n:
                        lr_ = lr_transfo * lr_decay ** (nb_blocks - 1 - i)
                        break

                if lr_ is None:  # embedding related layers
                    # print(n)
                    lr_ = lr_transfo * lr_decay ** (nb_blocks)

            else:
                lr_ = lr

            opt_params.append(
                {
                    "params": [p],
                    "weight_decay": wd,
                    "lr": lr_,
                }
            )
    else:
        for n, p in model.named_parameters():
            wd = 0 if any(nd in n for nd in no_decay) else weight_decay

            if transformer_name in n:
                lr_ = lr_transfo
            else:
                lr_ = lr

            opt_params.append(
                {
                    "params": [p],
                    "weight_decay": wd,
                    "lr": lr_,
                }
            )

    return opt_params


def trim_tensors(to_trim, pad_token=0, min_len=10):
    """
    Trim tensors so that within a batch, padding is shortened.
    This speeds up training for RNNs and Transformers

    Args:
        to_trim (list of torch tensors): Tokens to trim. First element has to be ids.
        model_name (str, optional): [description]. Defaults to 'bert'.
        min_len (int, optional): Minimum trimming size. Defaults to 10.

    Returns:
        list of torch tensors: Trimmed tokens.
    """
    max_len = (to_trim[0] != pad_token).sum(1).max()
    max_len = max(max_len, min_len)
    return [tokens[:, :max_len] for tokens in to_trim]


class AWP:
    """
    https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """
    def __init__(
        self,
        model,
        optimizer,
        loss_fct,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.001,
        start_step=1,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fct = loss_fct

        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_step = start_step
        self.adv_step = adv_step
    
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, ids, token_type_ids, y_batch, step, use_fp16=False):
        if (self.adv_lr == 0) or (step < self.start_step):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step() 
            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred = self.model(ids.cuda(), token_type_ids.cuda())
                adv_loss = self.loss_fct(y_pred, y_batch.cuda()).mean()
 
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}