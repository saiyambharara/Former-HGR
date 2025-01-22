import torch
from torch import nn
from models.Pa_Former_2 import spatial_transformer
# from models.S_Former import spatial_transformer
from models.T_Former import temporal_transformer


class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pa_former = spatial_transformer()
        # self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        self.drop =nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(18816, 512)
        # self.fc12 = nn.Linear(512, 16)
        
        self.fc = nn.Linear(512, 25)

    def forward(self, x):
        # print("x_Input:",x.size())
        x = self.pa_former(x)
        # x = self.s_former(x)
        # print("x_S_Output:",x.size())
        x = self.t_former(x)
        # print("x_T_Output:",x.size())
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc(x)
        # exit()
        return x


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)
