import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
# add
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from timeit import default_timer as timer


class VAE(torch.nn.Module):
    """
    VAE model
    """

    def __init__(self, input_dim=28 * 28, hidden_dim=256, mu_logvar_dim=20):
        super(VAE, self).__init__()

        # encode 编码器  -> [batch_size, (1, 28, 28)]
        self.en_linear1 = torch.nn.Linear(input_dim, hidden_dim)  # -> (1, 256)
        self.en_linear11 = torch.nn.Linear(hidden_dim, mu_logvar_dim)
        self.en_linear12 = torch.nn.Linear(hidden_dim, mu_logvar_dim)

        # 解码器
        self.de_linear1 = torch.nn.Linear(mu_logvar_dim, hidden_dim)
        self.de_linear2 = torch.nn.Linear(hidden_dim, input_dim)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def encode(self, x):
        en_h1 = self.relu(self.en_linear1(x))
        return self.en_linear11(en_h1), self.en_linear12(en_h1)

    # 重新参数化
    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        de_h3 = self.relu(self.de_linear1(z))
        return self.sigmoid(self.de_linear2(de_h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        x_recont = self.decode(z)
        return x_recont, mu, logvar


###
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            #print(words)
            imgs.append((words[0], int(words[1][7])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)
###


if __name__ == '__main__':

    # 起始时间
    tic = timer()

    # 超参数设置
    input_dim = 28 * 28  # 输入图片的 长 * 宽，高默认为 1
    hidden_dim = 256  # 隐藏层 256
    mu_logvar_dim = 20  # 普通神经网络的输出层

    epochs = 15  # 15轮 训练
    batch_size = 2  # 每一轮中每次训练单次投入的数据
    learning_rate = 1e-3
    N_TEST_IMG = 5  # 展示的图片
                                          )

    ##
    dataset = MyDataset(txt='./mydata/train.txt', transform=transforms.ToTensor())
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)
    ##

    # 创建一个实例
    example_model = VAE()

    # 优化器
    optimizer = optim.Adam(example_model.parameters(), lr=learning_rate)

    # 看动态图
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
    plt.ion()

    # 原始数据图
    # view_data = dataset.data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    raw_mydata = ['./mydata/train/0.jpg','./mydata/train/1.jpg','./mydata/train/2.jpg',\
                  './mydata/train/3.jpg','./mydata/train/4.jpg']
    for i in range(N_TEST_IMG):
    #     a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
        a[0][i].imshow(Image.open(raw_mydata[i]),cmap="gray")
        a[0][i].set_xticks(());
        a[0][i].set_yticks(())

    # 训练模型
    for epoch in range(epochs):
        for epochs_index, (train_data, _) in enumerate(data_loader):
            # 获取样本，并前向传播
            x = train_data.view(-1, input_dim)
            x_reconst, mu, log_var = example_model(x)

            # 计算重构损失和KL散度（KL散度用于衡量两种分布的相似程度）损失
            reconst_loss = F.mse_loss(x_reconst, x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # 反向传播和优化
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epochs_index + 1) % 1 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                      .format(epoch + 1, epochs, epochs_index + 1, len(data_loader), reconst_loss.item(),
                              kl_div.item()))

        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(x_reconst.data.numpy()[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(());
            a[1][i].set_yticks(())
        plt.draw();
        plt.pause(0.05)

    # 结束时间
    toc = timer()

    print(toc - tic)  # 输出的时间，秒为单位


