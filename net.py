import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import accuracy_score, f1_score


cls_loss_fn = nn.CrossEntropyLoss()
reg_loss_fn = nn.MSELoss()

reg_metric = nn.L1Loss()
def cls_metric(logit, label):
    pred_label = torch.argmax(logit, dim=-1)
    acc = accuracy_score(pred_label.cpu(), label.cpu())
    f1 = f1_score(pred_label.cpu(), label.view(-1).cpu(), average="macro")
    return acc, f1

# ------------------ LSTM ------------------------
class MultitaskLSTM(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(MultitaskLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_1, output_size)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        cls_out = out.contiguous()
        cls_out = cls_out.view(-1, out.shape[2])
        cls_out = self.cls(cls_out)

        out, _ = self.lstm2(out[:, -1, :].view(x.size(0), 1, -1))
        out = self.dropout(out)
        reg_out = self.reg(out[:, -1, :]).flatten()

        return reg_out, cls_out
    

class ClassifyLSTM(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(ClassifyLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_1, output_size)

    def forward(self, x):
        print(x.shape)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        cls_out = out.contiguous()
        cls_out = cls_out.view(-1, out.shape[2])
        cls_out = self.cls(cls_out)

        return cls_out

class RegressionLSTM(nn.Module):
    
    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(RegressionLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.reg = nn.Linear(hidden_size_2, 1)

        
    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out[:, -1, :].view(x.size(0), 1, -1))
        out = self.dropout(out)
        reg_out = self.reg(out[:, -1, :]).flatten()

        return reg_out

# ------------------ MLP ------------------------

class ClassifyMLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(ClassifyMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_2, output_size)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.cls(x)
        x = x.view(-1, x.shape[-1])
        return x
    

class ClassifyGRU(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(ClassifyGRU, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size_1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out = self.dropout(out)
        out = out.contiguous().view(-1, out.size(-1))  
        out = self.cls(out)
        return out

# ------------------ RNN ------------------------

class ClassifyRNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(ClassifyRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size_1, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out = self.dropout(out)
        out = out.contiguous().view(-1, out.size(-1)) 
        out = self.cls(out)
        return out

class RegressionRNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(RegressionRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size_1, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out = self.dropout(out)
        out = out[:, -1, :]
        reg_out = self.reg(out).squeeze()  
        return reg_out

class MultitaskRNN(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(MultitaskRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size_1, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_1, output_size)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        cls_out = out.contiguous()
        cls_out = cls_out.view(-1, cls_out.size(-1)) 
        cls_out = self.cls(cls_out)

        out, _ = self.rnn2(out)
        out = self.dropout(out)
        out = out[:, -1, :]
        reg_out = self.reg(out).squeeze()  

        return reg_out, cls_out
    
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MLSTMfcn2(nn.Module):
    def __init__(self, *, num_classes, max_seq_len, num_features,
                    num_lstm_out=128, num_lstm_layers=1, 
                    conv1_nf=128, conv2_nf=256, conv3_nf=128,
                    lstm_drop_p=0.8, fc_drop_p=0.3):
        super(MLSTMfcn2, self).__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf
        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=num_lstm_out, num_layers=num_lstm_layers, batch_first=True)
        self.conv1 = nn.Conv1d(num_features, conv1_nf, 8)
        self.conv2 = nn.Conv1d(conv1_nf, conv2_nf, 5)
        self.conv3 = nn.Conv1d(conv2_nf, conv3_nf, 3)
        self.bn1 = nn.BatchNorm1d(conv1_nf)
        self.bn2 = nn.BatchNorm1d(conv2_nf)
        self.bn3 = nn.BatchNorm1d(conv3_nf)
        self.se1 = SELayer(conv1_nf)
        self.se2 = SELayer(conv2_nf)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstm_drop_p)
        self.convDrop = nn.Dropout(fc_drop_p)
        self.fc = nn.Linear(conv3_nf + num_lstm_out, num_classes)

    def forward(self, x, seq_lens):
        x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        x1, (ht,ct) = self.lstm(x1)
        x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, padding_value=0.0)
        
        # Flatten x1 to have shape [B*T, num_lstm_out]
        x1_flattened = x1.contiguous().view(-1, self.num_lstm_out)
        
        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)
        
        # Assuming x2 needs to be expanded to match x1's flattening
        x2_expanded = x2.repeat(1, seq_lens[0]).view(-1, self.conv3_nf)
        
        x_all = torch.cat((x1_flattened, x2_expanded), dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)
        
        return x_out
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MLSTMfcn(nn.Module):
    def __init__(self, *, num_classes, max_seq_len, num_features,
                    num_lstm_out=128, num_lstm_layers=1, 
                    conv1_nf=128, conv2_nf=256, conv3_nf=128,
                    lstm_drop_p=0.8, fc_drop_p=0.3):
        super(MLSTMfcn, self).__init__()

        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features, 
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        
        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc = nn.Linear(self.conv3_nf+self.num_lstm_out, self.num_classes)
    
    def forward(self, x, seq_lens):
        ''' input x should be in size [B,T,F], where 
            B = Batch size
            T = Time samples
            F = features
        '''
        x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens, 
                                                batch_first=True, 
                                                enforce_sorted=False)
        x1, (ht,ct) = self.lstm(x1)
        x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, 
                                                    padding_value=0.0)
        x1 = x1[:,-1,:]
        x2 = x.transpose(2,1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2,2)
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out
    
    
    
class ClassifyCNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(ClassifyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size_1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size_1, out_channels=hidden_size_2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_size_2 * 75, 512)  # Tính toán kích thước đầu ra từ Convolutional Layers
        self.fc2 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
class ClassifyFCN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(ClassifyFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_size_2, output_size)  # Output layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Reshape input to treat each [x, y, z] data point as an independent sample
        x = x.view(-1, 3)  # Reshape from [48, 300, 3] to [14400, 3]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
