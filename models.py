import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


import math
import pdb

#############################################################################################################

class UNet(nn.Module):
	def __init__(self, hidden_size = 32):
		super(UNet, self).__init__()

		class Encoder(nn.Module):
			def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
				super(Encoder, self).__init__()

				self.seq = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.ReLU()
				)
				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				return self.seq(x) * self.GateConv(x)

		class Decoder(nn.Module):
			def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
				super(Decoder, self).__init__()
				
				self.seq = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
				)

				if tanh:
					self.activ = nn.Tanh()
				else:
					self.activ = nn.ReLU()
				
				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				s = self.seq(x)
				s = self.activ(s)
				return s * self.GateConv(x)


		self.enc0 = Encoder(16, hidden_size, stride=1)
		self.enc1 = Encoder(hidden_size, hidden_size, stride=2)
		self.enc2 = Encoder(hidden_size, hidden_size, stride=2)
		self.enc3 = Encoder(hidden_size, hidden_size, stride=2)

		self.dec0 = Decoder(hidden_size, hidden_size, stride=1)
		# up-scaling + concat
		self.dec1 = Decoder(hidden_size * 2, hidden_size, stride=1)
		self.dec2 = Decoder(hidden_size * 2, hidden_size, stride=1)
		self.dec3 = Decoder(hidden_size * 2, hidden_size , stride=1)

		self.dec4 = Decoder(hidden_size, 3, stride=1, tanh=True)
		'''for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()'''
    

	def forward(self, w1, w2, flo1, flo2, fr1, fr2):
		s0 = self.enc0(torch.cat([w1, w2, flo1, flo2, fr1, fr2],1).cuda())
		s1 = self.enc1(s0)
		s2 = self.enc2(s1)
		s3 = self.enc3(s2)

		s4 = self.dec0(s3)
		# up-scaling + concat
		s4 = F.interpolate(s4, scale_factor=2, mode='nearest')
		s5 = self.dec1(torch.cat([s4, s2],1).cuda())
		s5 = F.interpolate(s5, scale_factor=2, mode='nearest')
		s6 = self.dec2(torch.cat([s5, s1],1).cuda())
		s6 = F.interpolate(s6, scale_factor=2, mode='nearest')
		s7 = self.dec3(torch.cat([s6, s0],1).cuda())

		out = self.dec4(s7)
		return out

class ResNet(nn.Module):
	def __init__(self, hidden_size = 32):
		super(ResNet, self).__init__()

		class ConvBlock(nn.Module):
			def __init__(self, in_ch, out_ch):
				super(ConvBlock, self).__init__()

				self.seq = nn.Sequential(
					nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
					nn.LeakyReLU(0.2)
				)

				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(1),
					nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=0),
					nn.ReflectionPad2d(1),
					nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				return self.seq(x) * self.GateConv(x)


		class ResBlock(nn.Module):
			def __init__(self, num_ch):
				super(ResBlock, self).__init__()

				self.seq = nn.Sequential(
					nn.Conv2d(num_ch, num_ch, kernel_size=1, stride=1, padding=0),
					nn.ReLU()
				)

				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(1),
					nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=0),
					nn.ReflectionPad2d(1),
					nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				return self.seq(x) * self.GateConv(x) + x


		self.seq = nn.Sequential(
			ConvBlock(6, hidden_size),
			ResBlock(hidden_size),
			ResBlock(hidden_size),
			ResBlock(hidden_size),
			ResBlock(hidden_size),
			ResBlock(hidden_size),
			ConvBlock(hidden_size, 3),
			nn.Tanh()
		)
	#fint, warped3,flo3, ft
	def forward(self, fint, warped3):
		return self.seq(torch.cat([fint, warped3],1).cuda())


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x
	
class Critic(nn.Module):
    # Initializers
    def __init__(self, d=64):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1)#128
        self.relu1 = nn.LeakyReLU(0.2)
        self.norm1 = nn.InstanceNorm2d(d)

        self.conv2 = nn.Conv2d(d, d * 2, kernel_size=4, stride=2, padding=1)#64
        self.relu2 = nn.LeakyReLU(0.2)
        self.norm2 = nn.InstanceNorm2d(d * 2)

        self.conv3 = nn.Conv2d(d * 2, d * 4, kernel_size=4, stride=2, padding=1)#32
        self.relu3 = nn.LeakyReLU(0.2)
        self.norm3 = nn.InstanceNorm2d(d * 4)

        self.conv4 = nn.Conv2d(d * 4, d * 8, kernel_size=4, stride=2, padding=1)#16
        self.relu4 = nn.LeakyReLU(0.2)
        self.norm4 = nn.InstanceNorm2d(d * 8)
        
        self.conv5 = nn.Conv2d(d * 8, d * 8, kernel_size=4, stride=2, padding=1)#8
        self.relu5 = nn.LeakyReLU(0.2)


        self.flatten = nn.Flatten()
        self.linear = nn.Linear(d * 8 * 8 * 8, 1)  # Adjust the linear layer input size accordingly

        # Weight initialization
        self._initialize_weights()

    # Weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    # Forward method
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.norm4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.flatten(x)
        x = self.linear(x)

        return x