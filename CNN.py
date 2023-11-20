class CNNBlock(nn.Module):
    def __init__(self,in_channels, out_channels ,kernel_size = 3, stride =1  , padding = 1):## if groups = in_channels then this is depth_wise convolution operation
        super(CNNBlock , self).__init__()
        self.conv2d = nn.Conv2d(in_channels = in_channels , out_channels = out_channels , kernel_size = kernel_size , stride = stride , padding = padding , bias = False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
    def forward(self,x):
        return self.leaky_relu(self.batchnorm(self.conv2d(x)))

class Yolo(nn.Module):
    def __init__(self,S=7,B=2 , C=1 , input_channels = 3):
        super(Yolo , self).__init__()        
        self.darknet = nn.Sequential(
                     CNNBlock(in_channels = input_channels , out_channels = 64,kernel_size = 7 , stride = 2 , padding = 3),
                     nn.MaxPool2d(kernel_size = 2 , stride=2),
                     CNNBlock(in_channels = 64 , out_channels = 192,kernel_size = 3 , stride = 2 , padding = 1),   
                     nn.MaxPool2d(kernel_size = 2 , stride=2),
                     CNNBlock(in_channels = 192 , out_channels = 128,kernel_size = 1 , stride = 1 , padding = 0), 
                     CNNBlock(in_channels = 128 , out_channels = 256,kernel_size = 3 , stride = 1 , padding = 1),    
                     CNNBlock(in_channels = 256 , out_channels = 256,kernel_size = 1 , stride = 1 , padding = 0), 
                     CNNBlock(in_channels = 256 , out_channels = 512,kernel_size = 3 , stride = 1 , padding = 1),   
                     nn.MaxPool2d(kernel_size = 2 , stride=2),
                     CNNBlock(in_channels = 512 , out_channels = 256,kernel_size = 1 , stride = 1 , padding = 0), 
                     CNNBlock(in_channels = 256 , out_channels = 512,kernel_size = 3 , stride = 1 , padding = 1),    
                     CNNBlock(in_channels = 512 , out_channels = 256,kernel_size = 1 , stride = 1 , padding = 0), 
                     CNNBlock(in_channels = 256 , out_channels = 512,kernel_size = 3 , stride = 1 , padding = 1),      
                     CNNBlock(in_channels = 512 , out_channels = 256,kernel_size = 1 , stride = 1 , padding = 0), 
                     CNNBlock(in_channels = 256 , out_channels = 512,kernel_size = 3 , stride = 1 , padding = 1),    
                     CNNBlock(in_channels = 512 , out_channels = 256,kernel_size = 1 , stride = 1 , padding = 0), 
                     CNNBlock(in_channels = 256 , out_channels = 512,kernel_size = 3 , stride = 1 , padding = 1),
                     CNNBlock(in_channels = 512 , out_channels = 512,kernel_size = 1 , stride = 1 , padding = 0), 
                     CNNBlock(in_channels = 512 , out_channels = 1024,kernel_size = 3 , stride = 1 , padding = 1), 
                     nn.MaxPool2d(kernel_size = 2 , stride=2),
                     CNNBlock(in_channels = 1024 , out_channels = 512,kernel_size = 1 , stride = 1 , padding = 0), 
                     CNNBlock(in_channels = 512 , out_channels = 1024,kernel_size = 3 , stride = 1 , padding = 1),
                     CNNBlock(in_channels = 1024 , out_channels = 512,kernel_size = 1 , stride = 1 , padding = 0), 
                     CNNBlock(in_channels = 512 , out_channels = 1024,kernel_size = 3 , stride = 1 , padding = 1), 
                     CNNBlock(in_channels = 1024 , out_channels = 1024,kernel_size = 3 , stride = 1 , padding = 1), 
                     CNNBlock(in_channels = 1024 , out_channels = 1024,kernel_size = 3 , stride = 2 , padding = 2), 
                     CNNBlock(in_channels = 1024 , out_channels = 1024,kernel_size = 3 , stride = 1 , padding = 2), 
                     CNNBlock(in_channels = 1024 , out_channels = 1024,kernel_size = 3 , stride = 1 , padding = 1),   
                        )
        self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear((1024*S*S) , 4096), 
                nn.Dropout(0.4),
                nn.LeakyReLU(0.1),
                nn.Linear(4096 , S*S*(C + B*5)) 
                )
        
    def forward(self,x):
        x = self.darknet(x)
        return self.fc(x)
    