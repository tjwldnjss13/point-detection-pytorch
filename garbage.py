import torch


if __name__ == '__main__':
    a = torch.linspace(0, 11, 12).reshape(-1, 2)
    b = torch.Tensor([1.3])
    print(torch.ceil(b))