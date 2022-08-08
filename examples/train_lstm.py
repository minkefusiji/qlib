from lstm import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

import qlib
#from qlib.constant import REG_CN
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Fillna
from qlib.data.dataset import DatasetH

def train_lstm_benefit():
    # load dataset
    with open('dataset/selected_csi300_bin/features.txt', 'r') as fp:
        lines = fp.readlines()
        fields = []
        names = []
        for line in lines:
            fields.append('$' + line.strip().lower())
            names.append(line.strip().lower())

    provider_uri = "./dataset/benchmark_bin/"  # target_dir
    qlib.init(provider_uri=provider_uri)
    qdl = QlibDataLoader(config=[fields, names])
    qdl.load(start_time='2008-01-01', end_time='2020-08-01')
    dh = DataHandlerLP(start_time='2008-01-01', end_time='2020-08-01',
                infer_processors=[Fillna()],
                data_loader=qdl)
    ds = DatasetH(dh, segments={"train": ("2008-01-01", "2016-12-31"), "valid": ("2017-01-01", "2020-08-01")})

    X_names = names.copy()
    X_names.remove('benefit')
    X_names.remove('risk')
    X_names.remove('benefit2')
    X_names.remove('risk2')
    X_names.remove('benefit3')
    X_names.remove('risk3')

    x_train, y_train = ds.prepare('train')[X_names], ds.prepare('train')[['benefit']]
    x_valid, y_valid = ds.prepare('valid')[X_names], ds.prepare('valid')[['benefit']]
    benefit_datasets = [x_train, y_train, x_valid, y_valid]

    # wrap up to dataset
    benefit_train_dataset = SequenceDataset(ds.prepare('train'), 'benefit', X_names)
    benefit_train_loader = DataLoader(benefit_train_dataset, batch_size=5, shuffle=True)

    benefit_valid_dataset = SequenceDataset(ds.prepare('valid'), 'benefit', X_names)
    benefit_valid_loader = DataLoader(benefit_valid_dataset, batch_size=5, shuffle=True)

    # train model
    learning_rate = 1e-3
    num_hidden_units = 20

    model = ShallowRegressionLSTM(num_sensors=x_train.shape[1], hidden_units=num_hidden_units)
    # model.load_state_dict(torch.load('model/lstm_benefit_model2.pkl'))
    # model.eval()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(benefit_valid_loader, model, loss_function)
    print()

    for ix_epoch in range(1000):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(benefit_train_loader, model, loss_function, optimizer=optimizer)
        test_model(benefit_valid_loader, model, loss_function)
        print()
    
    # save model
    lstm_benefit = model
    torch.save(lstm_benefit.state_dict(), 'model/lstm_benefit_model2.pkl')

def train_lstm_risk():
    # load dataset
    with open('dataset/selected_csi300_bin/features.txt', 'r') as fp:
        lines = fp.readlines()
        fields = []
        names = []
        for line in lines:
            fields.append('$' + line.strip().lower())
            names.append(line.strip().lower())

    provider_uri = "./dataset/benchmark_bin/"  # target_dir
    qlib.init(provider_uri=provider_uri)
    qdl = QlibDataLoader(config=[fields, names])
    qdl.load(start_time='2008-01-01', end_time='2020-08-01')
    dh = DataHandlerLP(start_time='2008-01-01', end_time='2020-08-01',
                infer_processors=[Fillna()],
                data_loader=qdl)
    ds = DatasetH(dh, segments={"train": ("2008-01-01", "2016-12-31"), "valid": ("2017-01-01", "2020-08-01")})

    X_names = names.copy()
    X_names.remove('benefit')
    X_names.remove('risk')
    X_names.remove('benefit2')
    X_names.remove('risk2')
    X_names.remove('benefit3')
    X_names.remove('risk3')

    x_train, y_train = ds.prepare('train')[X_names], ds.prepare('train')[['risk']]
    x_valid, y_valid = ds.prepare('valid')[X_names], ds.prepare('valid')[['risk']]
    risk_datasets = [x_train, y_train, x_valid, y_valid]


    # wrap up to dataset
    risk_train_dataset = SequenceDataset(ds.prepare('train'), 'risk', X_names)
    risk_train_loader = DataLoader(risk_train_dataset, batch_size=5, shuffle=True)

    risk_valid_dataset = SequenceDataset(ds.prepare('valid'), 'risk', X_names)
    risk_valid_loader = DataLoader(risk_valid_dataset, batch_size=5, shuffle=True)

    # train model
    learning_rate = 1e-3
    num_hidden_units = 20

    model = ShallowRegressionLSTM(num_sensors=x_train.shape[1], hidden_units=num_hidden_units)
    # model.load_state_dict(torch.load('model/lstm_risk_model2.pkl'))
    # model.eval()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(risk_valid_loader, model, loss_function)
    print()

    for ix_epoch in range(1000):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(risk_train_loader, model, loss_function, optimizer=optimizer)
        test_model(risk_valid_loader, model, loss_function)
        print()

    # save model
    lstm_risk = model
    torch.save(lstm_risk.state_dict(), 'model/lstm_risk_model2.pkl')

def train_lstm_benefit2():
    # load dataset
    with open('dataset/selected_csi300_bin/features.txt', 'r') as fp:
        lines = fp.readlines()
        fields = []
        names = []
        for line in lines:
            fields.append('$' + line.strip().lower())
            names.append(line.strip().lower())

    provider_uri = "./dataset/benchmark_bin/"  # target_dir
    qlib.init(provider_uri=provider_uri)
    qdl = QlibDataLoader(config=[fields, names])
    qdl.load(start_time='2008-01-01', end_time='2020-08-01')
    dh = DataHandlerLP(start_time='2008-01-01', end_time='2020-08-01',
                infer_processors=[Fillna()],
                data_loader=qdl)
    ds = DatasetH(dh, segments={"train": ("2008-01-01", "2016-12-31"), "valid": ("2017-01-01", "2020-08-01")})

    X_names = names.copy()
    X_names.remove('benefit')
    X_names.remove('risk')
    X_names.remove('benefit2')
    X_names.remove('risk2')
    X_names.remove('benefit3')
    X_names.remove('risk3')

    x_train, y_train = ds.prepare('train')[X_names], ds.prepare('train')[['benefit2']]
    x_valid, y_valid = ds.prepare('valid')[X_names], ds.prepare('valid')[['benefit2']]
    benefit_datasets = [x_train, y_train, x_valid, y_valid]

    # wrap up to dataset
    benefit_train_dataset = SequenceDataset(ds.prepare('train'), 'benefit2', X_names)
    benefit_train_loader = DataLoader(benefit_train_dataset, batch_size=5, shuffle=True)

    benefit_valid_dataset = SequenceDataset(ds.prepare('valid'), 'benefit2', X_names)
    benefit_valid_loader = DataLoader(benefit_valid_dataset, batch_size=5, shuffle=True)

    # train model
    learning_rate = 1e-3
    num_hidden_units = 20

    model = ShallowRegressionLSTM(num_sensors=x_train.shape[1], hidden_units=num_hidden_units)
    # model.load_state_dict(torch.load('model/lstm_benefit_model2.pkl'))
    # model.eval()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(benefit_valid_loader, model, loss_function)
    print()

    for ix_epoch in range(1000):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(benefit_train_loader, model, loss_function, optimizer=optimizer)
        test_model(benefit_valid_loader, model, loss_function)
        print()
    
    # save model
    lstm_benefit = model
    torch.save(lstm_benefit.state_dict(), 'model/lstm_benefit2_model2.pkl')

def train_lstm_risk2():
    # load dataset
    with open('dataset/selected_csi300_bin/features.txt', 'r') as fp:
        lines = fp.readlines()
        fields = []
        names = []
        for line in lines:
            fields.append('$' + line.strip().lower())
            names.append(line.strip().lower())

    provider_uri = "./dataset/benchmark_bin/"  # target_dir
    qlib.init(provider_uri=provider_uri)
    qdl = QlibDataLoader(config=[fields, names])
    qdl.load(start_time='2008-01-01', end_time='2020-08-01')
    dh = DataHandlerLP(start_time='2008-01-01', end_time='2020-08-01',
                infer_processors=[Fillna()],
                data_loader=qdl)
    ds = DatasetH(dh, segments={"train": ("2008-01-01", "2016-12-31"), "valid": ("2017-01-01", "2020-08-01")})

    X_names = names.copy()
    X_names.remove('benefit')
    X_names.remove('risk')
    X_names.remove('benefit2')
    X_names.remove('risk2')
    X_names.remove('benefit3')
    X_names.remove('risk3')

    x_train, y_train = ds.prepare('train')[X_names], ds.prepare('train')[['risk2']]
    x_valid, y_valid = ds.prepare('valid')[X_names], ds.prepare('valid')[['risk2']]
    risk_datasets = [x_train, y_train, x_valid, y_valid]


    # wrap up to dataset
    risk_train_dataset = SequenceDataset(ds.prepare('train'), 'risk2', X_names)
    risk_train_loader = DataLoader(risk_train_dataset, batch_size=5, shuffle=True)

    risk_valid_dataset = SequenceDataset(ds.prepare('valid'), 'risk2', X_names)
    risk_valid_loader = DataLoader(risk_valid_dataset, batch_size=5, shuffle=True)

    # train model
    learning_rate = 1e-3
    num_hidden_units = 20

    model = ShallowRegressionLSTM(num_sensors=x_train.shape[1], hidden_units=num_hidden_units)
    # model.load_state_dict(torch.load('model/lstm_risk_model2.pkl'))
    # model.eval()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(risk_valid_loader, model, loss_function)
    print()

    for ix_epoch in range(1000):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(risk_train_loader, model, loss_function, optimizer=optimizer)
        test_model(risk_valid_loader, model, loss_function)
        print()

    # save model
    lstm_risk = model
    torch.save(lstm_risk.state_dict(), 'model/lstm_risk2_model2.pkl')

def train_lstm_benefit3():
    # load dataset
    with open('dataset/selected_csi300_bin/features.txt', 'r') as fp:
        lines = fp.readlines()
        fields = []
        names = []
        for line in lines:
            fields.append('$' + line.strip().lower())
            names.append(line.strip().lower())

    provider_uri = "./dataset/benchmark_bin/"  # target_dir
    qlib.init(provider_uri=provider_uri)
    qdl = QlibDataLoader(config=[fields, names])
    qdl.load(start_time='2008-01-01', end_time='2020-08-01')
    dh = DataHandlerLP(start_time='2008-01-01', end_time='2020-08-01',
                infer_processors=[Fillna()],
                data_loader=qdl)
    ds = DatasetH(dh, segments={"train": ("2008-01-01", "2016-12-31"), "valid": ("2017-01-01", "2020-08-01")})

    X_names = names.copy()
    X_names.remove('benefit')
    X_names.remove('risk')
    X_names.remove('benefit2')
    X_names.remove('risk2')
    X_names.remove('benefit3')
    X_names.remove('risk3')

    x_train, y_train = ds.prepare('train')[X_names], ds.prepare('train')[['benefit3']]
    x_valid, y_valid = ds.prepare('valid')[X_names], ds.prepare('valid')[['benefit3']]
    benefit_datasets = [x_train, y_train, x_valid, y_valid]

    # wrap up to dataset
    benefit_train_dataset = SequenceDataset(ds.prepare('train'), 'benefit3', X_names)
    benefit_train_loader = DataLoader(benefit_train_dataset, batch_size=5, shuffle=True)

    benefit_valid_dataset = SequenceDataset(ds.prepare('valid'), 'benefit3', X_names)
    benefit_valid_loader = DataLoader(benefit_valid_dataset, batch_size=5, shuffle=True)

    # train model
    learning_rate = 1e-3
    num_hidden_units = 20

    model = ShallowRegressionLSTM(num_sensors=x_train.shape[1], hidden_units=num_hidden_units)
    # model.load_state_dict(torch.load('model/lstm_benefit_model2.pkl'))
    # model.eval()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(benefit_valid_loader, model, loss_function)
    print()

    for ix_epoch in range(1000):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(benefit_train_loader, model, loss_function, optimizer=optimizer)
        test_model(benefit_valid_loader, model, loss_function)
        print()
    
    # save model
    lstm_benefit = model
    torch.save(lstm_benefit.state_dict(), 'model/lstm_benefit3.pkl')

def train_lstm_risk3():
    # load dataset
    with open('dataset/selected_csi300_bin/features.txt', 'r') as fp:
        lines = fp.readlines()
        fields = []
        names = []
        for line in lines:
            fields.append('$' + line.strip().lower())
            names.append(line.strip().lower())

    provider_uri = "./dataset/benchmark_bin/"  # target_dir
    qlib.init(provider_uri=provider_uri)
    qdl = QlibDataLoader(config=[fields, names])
    qdl.load(start_time='2008-01-01', end_time='2020-08-01')
    dh = DataHandlerLP(start_time='2008-01-01', end_time='2020-08-01',
                infer_processors=[Fillna()],
                data_loader=qdl)
    ds = DatasetH(dh, segments={"train": ("2008-01-01", "2016-12-31"), "valid": ("2017-01-01", "2020-08-01")})

    X_names = names.copy()
    X_names.remove('benefit')
    X_names.remove('risk')
    X_names.remove('benefit2')
    X_names.remove('risk2')
    X_names.remove('benefit3')
    X_names.remove('risk3')

    x_train, y_train = ds.prepare('train')[X_names], ds.prepare('train')[['risk3']]
    x_valid, y_valid = ds.prepare('valid')[X_names], ds.prepare('valid')[['risk3']]
    risk_datasets = [x_train, y_train, x_valid, y_valid]


    # wrap up to dataset
    risk_train_dataset = SequenceDataset(ds.prepare('train'), 'risk3', X_names)
    risk_train_loader = DataLoader(risk_train_dataset, batch_size=5, shuffle=True)

    risk_valid_dataset = SequenceDataset(ds.prepare('valid'), 'risk3', X_names)
    risk_valid_loader = DataLoader(risk_valid_dataset, batch_size=5, shuffle=True)

    # train model
    learning_rate = 1e-3
    num_hidden_units = 20

    model = ShallowRegressionLSTM(num_sensors=x_train.shape[1], hidden_units=num_hidden_units)
    # model.load_state_dict(torch.load('model/lstm_risk_model2.pkl'))
    # model.eval()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(risk_valid_loader, model, loss_function)
    print()

    for ix_epoch in range(1000):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(risk_train_loader, model, loss_function, optimizer=optimizer)
        test_model(risk_valid_loader, model, loss_function)
        print()

    # save model
    lstm_risk = model
    torch.save(lstm_risk.state_dict(), 'model/lstm_risk3.pkl')

def train_lstm_benefit4():
    # load dataset
    with open('dataset/selected_csi300_bin/features.txt', 'r') as fp:
        lines = fp.readlines()
        fields = []
        names = []
        for line in lines:
            fields.append('$' + line.strip().lower())
            names.append(line.strip().lower())

    provider_uri = "./dataset/benchmark_bin/"  # target_dir
    qlib.init(provider_uri=provider_uri)
    qdl = QlibDataLoader(config=[fields, names])
    qdl.load(start_time='2008-01-01', end_time='2020-08-01')
    dh = DataHandlerLP(start_time='2008-01-01', end_time='2020-08-01',
                infer_processors=[Fillna()],
                data_loader=qdl)
    ds = DatasetH(dh, segments={"train": ("2008-01-01", "2016-12-31"), "valid": ("2017-01-01", "2020-08-01")})

    X_names = names.copy()
    X_names.remove('benefit')
    X_names.remove('risk')
    X_names.remove('benefit2')
    X_names.remove('risk2')
    X_names.remove('benefit3')
    X_names.remove('risk3')
    X_names.remove('benefit4')
    X_names.remove('risk4')
    X_names.remove('label0_y')

    x_train, y_train = ds.prepare('train')[X_names], ds.prepare('train')[['benefit4']]
    x_valid, y_valid = ds.prepare('valid')[X_names], ds.prepare('valid')[['benefit4']]
    benefit_datasets = [x_train, y_train, x_valid, y_valid]

    # wrap up to dataset
    benefit_train_dataset = SequenceDataset(ds.prepare('train'), 'benefit4', X_names)
    benefit_train_loader = DataLoader(benefit_train_dataset, batch_size=5, shuffle=True)

    benefit_valid_dataset = SequenceDataset(ds.prepare('valid'), 'benefit4', X_names)
    benefit_valid_loader = DataLoader(benefit_valid_dataset, batch_size=5, shuffle=True)

    # train model
    learning_rate = 1e-5
    num_hidden_units = 20

    model = ShallowRegressionLSTM(num_sensors=x_train.shape[1], hidden_units=num_hidden_units)
    # model.load_state_dict(torch.load('model/lstm_benefit_model4.pkl'))
    # model.eval()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(benefit_valid_loader, model, loss_function)
    print()

    for ix_epoch in range(2000):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(benefit_train_loader, model, loss_function, optimizer=optimizer)
        avg_loss = test_model(benefit_valid_loader, model, loss_function)
        with open("model/benefit4_loss.txt", "a+") as fp:
            fp.write(str(avg_loss) + "\n")
        print()
    
    # save model
    lstm_benefit = model
    torch.save(lstm_benefit.state_dict(), 'model/lstm_benefit4.pkl')

def train_lstm_risk4():
    # load dataset
    with open('dataset/selected_csi300_bin/features.txt', 'r') as fp:
        lines = fp.readlines()
        fields = []
        names = []
        for line in lines:
            fields.append('$' + line.strip().lower())
            names.append(line.strip().lower())

    provider_uri = "./dataset/benchmark_bin/"  # target_dir
    qlib.init(provider_uri=provider_uri)
    qdl = QlibDataLoader(config=[fields, names])
    qdl.load(start_time='2008-01-01', end_time='2020-08-01')
    dh = DataHandlerLP(start_time='2008-01-01', end_time='2020-08-01',
                infer_processors=[Fillna()],
                data_loader=qdl)
    ds = DatasetH(dh, segments={"train": ("2008-01-01", "2016-12-31"), "valid": ("2017-01-01", "2020-08-01")})

    X_names = names.copy()
    X_names.remove('benefit')
    X_names.remove('risk')
    X_names.remove('benefit2')
    X_names.remove('risk2')
    X_names.remove('benefit3')
    X_names.remove('risk3')
    X_names.remove('benefit4')
    X_names.remove('risk4')
    X_names.remove('label0_y')

    x_train, y_train = ds.prepare('train')[X_names], ds.prepare('train')[['risk4']]
    x_valid, y_valid = ds.prepare('valid')[X_names], ds.prepare('valid')[['risk4']]
    risk_datasets = [x_train, y_train, x_valid, y_valid]


    # wrap up to dataset
    risk_train_dataset = SequenceDataset(ds.prepare('train'), 'risk4', X_names)
    risk_train_loader = DataLoader(risk_train_dataset, batch_size=5, shuffle=True)

    risk_valid_dataset = SequenceDataset(ds.prepare('valid'), 'risk4', X_names)
    risk_valid_loader = DataLoader(risk_valid_dataset, batch_size=5, shuffle=True)

    # train model
    learning_rate = 1e-5
    num_hidden_units = 20

    model = ShallowRegressionLSTM(num_sensors=x_train.shape[1], hidden_units=num_hidden_units)
    # model.load_state_dict(torch.load('model/lstm_risk_model4.pkl'))
    # model.eval()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(risk_valid_loader, model, loss_function)
    print()

    for ix_epoch in range(2000):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(risk_train_loader, model, loss_function, optimizer=optimizer)
        avg_loss = test_model(risk_valid_loader, model, loss_function)
        with open("model/risk4_loss.txt", "a+") as fp:
            fp.write(str(avg_loss) + "\n")
        print()

    # save model
    lstm_risk = model
    torch.save(lstm_risk.state_dict(), 'model/lstm_risk4.pkl')

def train_lstm_ic():
    # load dataset
    with open('dataset/selected_csi300_bin/features.txt', 'r') as fp:
        lines = fp.readlines()
        fields = []
        names = []
        for line in lines:
            fields.append('$' + line.strip().lower())
            names.append(line.strip().lower())

    provider_uri = "./dataset/benchmark_bin/"  # target_dir
    qlib.init(provider_uri=provider_uri)
    qdl = QlibDataLoader(config=[fields, names])
    qdl.load(start_time='2008-01-01', end_time='2020-08-01')
    dh = DataHandlerLP(start_time='2008-01-01', end_time='2020-08-01',
                infer_processors=[Fillna()],
                data_loader=qdl)
    ds = DatasetH(dh, segments={"train": ("2008-01-01", "2016-12-31"), "valid": ("2017-01-01", "2020-08-01")})

    X_names = names.copy()
    X_names.remove('benefit')
    X_names.remove('risk')
    X_names.remove('benefit2')
    X_names.remove('risk2')
    X_names.remove('benefit3')
    X_names.remove('risk3')
    X_names.remove('benefit4')
    X_names.remove('risk4')
    X_names.remove('label0_y')

    x_train, y_train = ds.prepare('train')[X_names], ds.prepare('train')[['label0_y']]
    x_valid, y_valid = ds.prepare('valid')[X_names], ds.prepare('valid')[['label0_y']]
    benefit_datasets = [x_train, y_train, x_valid, y_valid]

    # wrap up to dataset
    benefit_train_dataset = SequenceDataset(ds.prepare('train'), 'label0_y', X_names)
    benefit_train_loader = DataLoader(benefit_train_dataset, batch_size=5, shuffle=True)

    benefit_valid_dataset = SequenceDataset(ds.prepare('valid'), 'label0_y', X_names)
    benefit_valid_loader = DataLoader(benefit_valid_dataset, batch_size=5, shuffle=True)

    # train model
    learning_rate = 1e-5
    num_hidden_units = 20

    model = ShallowRegressionLSTM(num_sensors=x_train.shape[1], hidden_units=num_hidden_units)
    # model.load_state_dict(torch.load('model/lstm_benefit_model4.pkl'))
    # model.eval()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Untrained test\n--------")
    test_model(benefit_valid_loader, model, loss_function)
    print()

    for ix_epoch in range(2000):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(benefit_train_loader, model, loss_function, optimizer=optimizer)
        avg_loss = test_model(benefit_valid_loader, model, loss_function)
        with open("model/ic_score_loss.txt", "a+") as fp:
            fp.write(str(avg_loss) + "\n")
        print()
    
    # save model
    lstm_benefit = model
    torch.save(lstm_benefit.state_dict(), 'model/lstm_icscore.pkl')


if __name__ == "__main__":
    # train_lstm_benefit()
    # train_lstm_risk()
    # train_lstm_benefit2()
    # train_lstm_risk2()
    # train_lstm_benefit3()
    # train_lstm_risk3()
    train_lstm_ic()
    train_lstm_risk4()
    train_lstm_benefit4()

    with open("model/benefit4_loss.txt", "r") as fp:
        lines_benefit4 = fp.readlines()
        new_lines_benefit4 = []
        for line in lines_benefit4:
            new_lines_benefit4.append(float(line.strip()))

    with open("model/risk4_loss.txt", "r") as fp:
        lines_risk4 = fp.readlines()
        new_lines_risk4 = []
        for line in lines_risk4:
            new_lines_risk4.append(float(line.strip()))
    
    print(new_lines_benefit4)
    print(new_lines_risk4)

    plt.figure()
    plt.title("Benefit4 (avg(Close_i, ..., Close_{i + 30}) - Close_i) model loss")
    plt.plot(range(len(new_lines_benefit4)), new_lines_benefit4)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("model/benefit4_loss.png")

    plt.figure()
    plt.title("Risk4 (avg(ATR_i, ..., ATR_{i + 30}) model loss")
    plt.plot(range(len(new_lines_risk4)), new_lines_risk4)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("model/risk4_loss.png")

    plt.figure()
    plt.title("IC score model loss")
    plt.plot(range(len(new_lines_risk4)), new_lines_risk4)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("model/icscore_loss.png")
