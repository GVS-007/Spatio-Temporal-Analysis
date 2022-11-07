class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, device):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.device = device
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        h_cur, c_cur = h_cur.to(self.device), c_cur.to(self.device)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, height, width).to(self.device))
        
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size,device,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []

        cur_input_dim = self.input_dim 
        self.sigmoid = nn.Sigmoid()
        self.device = device


        self.conv_lstm = ConvLSTMCell(input_dim=cur_input_dim,
                                      hidden_dim=self.hidden_dim,
                                      kernel_size=self.kernel_size,
                                          bias=self.bias, device = self.device)
        
        self.conv1 = nn.Conv2d(self.hidden_dim, 20, 3)
        self.pool1 = nn.MaxPool2d(2, 2)             
        self.conv2 = nn.Conv2d(20, 40, 3)       
        self.conv3 = nn.Conv2d(40, 60, 3)       
        self.conv4 = nn.Conv2d(60, 80, 3)       
        self.fc1 = nn.Linear(80*12*12, 3000)
        self.fc2 = nn.Linear(3000, 700)
        self.fc3 = nn.Linear(700, 150)
        self.fc4 = nn.Linear(150, 50)
        self.fc5 = nn.Linear(50, 1)
    def forward(self, input_tensor):
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        h, c = self._init_hidden(input_tensor.size(0), (input_tensor.size(3), input_tensor.size(4)))
        h,c = h.to(self.device), c.to(self.device)
        for t in range(seq_len):
            h, c = self.conv_lstm(input_tensor=cur_layer_input[:, t, :, :, :],
                                              cur_state=[h, c])
            
        y = h
        y = self.pool1(F.relu(self.conv1(y))) 
        y = self.pool1(F.relu(self.conv2(y))) 
        y = self.pool1(F.relu(self.conv3(y)))
        y = self.pool1(F.relu(self.conv4(y)))
        # print(y.shape, 80*12*12)
        out = y.view(-1, 80*12*12)
        # print(out.shape)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.sigmoid(self.fc5(out))
        return out[0]
        out = out.view(1, -1)
        out = out[:,-1]
        # print(out[0][0][0].shape)
        return out[0][0][0]       # out, hidden = self.lstm1(y.view(-1, 10920), hidden)                     
        # out = F.relu(self.fc1(out))

        # out = self.sigmoid(self.fc6(out))
        # out = out.view(1, -1)
        # out = out[:,-1]
        # return out
            
        # return h

    def _init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, height, width).to(self.device))
