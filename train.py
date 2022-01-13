import torch
import numpy as np
import random
import os
from PUNet_Pytorch.model.punet import get_model
from PUNet_Pytorch.utils.utils import get_emd_loss, get_repulsion_loss
from dataset.ModelNet40 import get_data_loader
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
blue = lambda x: '\033[94m' + x + '\033[0m'


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def main():
    seed_torch(1234)
    batch_size = 8
    train_loader = get_data_loader(train=True, attack=False, batch_size=batch_size)
    model = get_model(npoint=2048, up_ratio=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)  # 对学习率进行调整
    epochs = 201
    for epoch in range(epochs):
        for data, label in train_loader:
            optimizer.zero_grad()
            data = data.float().to(device).contiguous()
            re_data = model(data)

            emd_loss = get_emd_loss(re_data.transpose(1, 2).contiguous(), data.transpose(1, 2).contiguous(), 1.0)

            re_loss = get_repulsion_loss(re_data)

            print('[%d] emd loss: %f re_loss: %f' % (epoch, emd_loss, re_loss))

            loss = emd_loss + re_loss

            loss.backward()

            optimizer.step()

        scheduler.step(epoch)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'out/model_epoch%d.pth' % epoch)


if __name__ == '__main__':
    main()
