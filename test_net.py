import torch

tensor = torch.tensor([0.], device='cuda:0')
tensor4 = torch.tensor([0., 0., 0., 0.], device='cuda:0')
if tensor4.size() == torch.Size([4]):
    print("The tensor has size [4].")
else:
    print("The tensor does not have size [4].")