import time
import torch
from tqdm import tqdm

x = torch.rand((512, 1024, 1024)).cuda()
y = torch.rand((512, 1024, 1024)).cuda()

minutes = 600

for _ in tqdm(range(100 * 60 * minutes)):
    time.sleep(0.01)
    y = (x * y)
