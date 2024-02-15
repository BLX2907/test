import torch
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