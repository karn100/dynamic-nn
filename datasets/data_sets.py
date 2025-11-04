import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

def load_mnist(batch_size = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train = datasets.MNIST("./data",train=True,download=True,transform=transform)
    test = datasets.MNIST("./data",train=False,download=True,transform=transform)

    train_loader = DataLoader(train,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test,batch_size=batch_size,shuffle=False)

    return train_loader,test_loader,784,10   #train_data,test_data, in_dims,out_dims
