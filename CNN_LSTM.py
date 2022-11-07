class CNN_LSTM(nn.Module):
    def __init__(self, device):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.pool1 = nn.MaxPool2d(2, 2)            
        self.conv2 = nn.Conv2d(20, 40, 3)       
        self.conv3 = nn.Conv2d(40, 60, 3)      
        self.conv4 = nn.Conv2d(60, 80, 3)   
        # self.conv5 = nn.Conv2d(80, 100, 3)   
        # self.conv6 = nn.Conv2d(100, 120, 3)   
        # self.conv7 = nn.Conv2d(120, 140, 3)   
 
        
        self.fc1 = nn.Linear(3000, 1000)       
        self.fc3 = nn.Linear(1000, 500)               
        self.fc4 = nn.Linear(500, 250)             
        self.fc5 = nn.Linear(250, 100)               
        self.fc6 = nn.Linear(100, 1)               
                                              
       

        self.lstm1 = nn.LSTM(80*12*12,3000)
        self.sigmoid = nn.Sigmoid()
        self.device = device


    def forward(self, x, hidden):
        for img in x:
            # y = torch.permute(img,(2,0,1)).float()
            y = img
            y = self.pool1(F.relu(self.conv1(y))) 
            y = self.pool1(F.relu(self.conv2(y))) 
            y = self.pool1(F.relu(self.conv3(y)))
            y = self.pool1(F.relu(self.conv4(y))) 
            # y = self.pool1(F.relu(self.conv5(y))) 
            # y = self.pool1(F.relu(self.conv6(y))) 
            # y = self.pool1(F.relu(self.conv7(y))) 
            out, hidden = self.lstm1(y.view(-1, 80*12*12), hidden)                     
        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = self.sigmoid(self.fc6(out))
        out = out.view(1, -1)
        out = out[:,-1]
        return out
        # x = torch.permute(x,(2,0,1)).float()
        # x = self.pool1(F.relu(self.conv1(x)))  
        # x = self.pool1(F.relu(self.conv2(x)))  
        # x = self.pool1(F.relu(self.conv3(x)))
        # x = self.pool1(F.relu(self.conv4(x)))
        # x = self.lstm_layer(x.view(-1, 120*65*118))
        # x = x.view(-1, 30*89*48)            # -> n, 400
        # return x
    def init_hidden(self):
        # c0 = torch.rand(100).float().to(self.device)
        # h0 = torch.rand(100).float().to(self.device)
        c0 = torch.zeros((1, 3000))
        h0 = torch.zeros((1, 3000))
        return (h0,c0)
