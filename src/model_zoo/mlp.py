import torch
import torch.nn as nn

    
def define_model(name="mlp", nb_ft=50, d=512, p=0., num_layers=4, num_classes=1):
    if name == "mlp":
        return MlpModel(
            nb_ft=nb_ft,
            d=d,
            p=p,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    elif name == "res":
        return ResModel(
            nb_ft=nb_ft,
            d=d,
            p=p,
            num_layers=num_layers,
            num_classes=num_classes,
        )
    else:
        raise NotImplementedError

        
class MlpModel(nn.Module):
    def __init__(self, nb_ft=50, d=1024, num_layers=4, p=0., num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.layer0 = nn.Sequential(
            nn.BatchNorm1d(nb_ft),
            nn.Linear(nb_ft, d),
            nn.LeakyReLU(),
        )
        
        self.mlp = []
        for i in range(num_layers):
            self.mlp.append(nn.Sequential(
                nn.BatchNorm1d(d),
                nn.Dropout(p=p),
                nn.Linear(d, d // 2),
                nn.LeakyReLU(),
            ))
            d = d // 2
        self.mlp = nn.Sequential(*self.mlp)

        self.logits = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Dropout(p=p),
            nn.Linear(d, num_classes),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.mlp(x)
        return self.logits(x)
    

class ResBlock(nn.Module):
    def __init__(self, h, p=0, use_bn=True):
        super().__init__()
        
        if use_bn:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(h),
                nn.Dropout(p),
                nn.Linear(h, h),
                nn.LeakyReLU(),
                nn.BatchNorm1d(h),
                # nn.Dropout(p),
                nn.Linear(h, h),
                nn.LeakyReLU(),
            )
        else:
            self.layers = nn.Sequential(
                nn.BatchNorm1d(h),
                nn.Dropout(p),
                nn.Linear(h, h),
                nn.LeakyReLU(),
                nn.BatchNorm1d(h),
                # nn.Dropout(p),
                nn.Linear(h, h),
                nn.LeakyReLU(),
            )
                
    def forward(self, x):
        return self.layers(x) + x

    
class ResModel(nn.Module):
    def __init__(self, nb_ft=50, d=1024, num_layers=4, p=0., num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.layer0 = nn.Sequential(
            nn.BatchNorm1d(nb_ft),
            nn.Dropout(p),
            nn.Linear(nb_ft, d),
            nn.LeakyReLU(),
        )

        self.resnet = nn.Sequential(
            *[ResBlock(d, p=p, use_bn=True) for _ in range(num_layers)]
        )

        self.logits = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Dropout(p=p),
            nn.Linear(d, num_classes),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.resnet(x)
        return self.logits(x)