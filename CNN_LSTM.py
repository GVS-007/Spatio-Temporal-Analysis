class CNN_LSTM(nn.Module):
    def __init__(self, device):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 3)
        self.pool1 = nn.MaxPool2d(2, 2)            
        self.conv2 = nn.Conv2d(20, 40, 3)       
        self.conv3 = nn.Conv2d(40, 60, 3)      
        self.conv4 = nn.Conv2d(60, 80, 3)   
        
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
            y = img
            y = self.pool1(F.relu(self.conv1(y))) 
            y = self.pool1(F.relu(self.conv2(y))) 
            y = self.pool1(F.relu(self.conv3(y)))
            y = self.pool1(F.relu(self.conv4(y))) 
            out, hidden = self.lstm1(y.view(-1, 80*12*12), hidden)                     
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = self.sigmoid(self.fc6(out))
        out = out.view(1, -1)
        out = out[:,-1]
        return out
    def init_hidden(self):
        c0 = torch.zeros((1, 3000))
        h0 = torch.zeros((1, 3000))
        return (h0,c0)
