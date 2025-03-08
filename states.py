import os
import yaml
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from FeatureCloud.app.engine.app import AppState, app_state, Role

from src.classes import Client, ISUPDataset, ISUPPredictor
from src.utils import fed_avg, test



@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('distribute_model', Role.COORDINATOR)
        self.register_transition('pre_training', Role.PARTICIPANT)


    def run(self):

        # read in the config
        config_file = os.path.join(os.getcwd(), "mnt", "input", "config.yaml")
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # initialize client
        client_name = self.id
        train_df = pd.read_csv(os.path.join("/mnt", "input", "train.csv"), index_col=0).T.reset_index().rename(columns={'index': 'image_id'})
        train_loader = DataLoader(ISUPDataset(train_df), batch_size=config["batch_size"], shuffle=True)
        val_df = pd.read_csv(os.path.join("/mnt", "input", "valid.csv"), index_col=0).T.reset_index().rename(columns={'index': 'image_id'})
        val_loader = DataLoader(ISUPDataset(val_df), batch_size=config["batch_size"], shuffle=False)
        model = ISUPPredictor(input_channels=config["input_channels"], n_classes=6)

        # initialise client with model
        client = Client(client_name, train_loader, val_loader, model, config["lr"])

        # store client class and config
        self.store(key='client', value=client)
        self.store(key='config', value=config)
        
        if self.is_coordinator:
            return 'distribute_model'
        return 'pre_training'
    

@app_state('distribute_model')
class DistributeModelState(AppState):

    def register(self):
        self.register_transition('pre_training', Role.COORDINATOR)


    def run(self):
        client = self.load('client')

        # send initialized model of coordinator to all clients
        self.broadcast_data(client.model.state_dict(), send_to_self=True)

        self.store(key="best_model", value=client.model.state_dict())
        self.store(key="best_val_acc", value=0.0)

        return 'pre_training'


@app_state('pre_training')
class PreTrainingState(AppState):

    def register(self):
        self.register_transition('train', Role.BOTH)


    def run(self):
        client = self.load('client')

        # receive initial model by the server
        server_state = self.await_data(n=1, is_json=False)
        client.model.load_state_dict(server_state)

        # store client class
        self.store(key='client', value=client)

        return 'train'


@app_state('train')
class TrainState(AppState):

    def register(self):
        self.register_transition('aggregate', Role.COORDINATOR)
        self.register_transition('validate', Role.PARTICIPANT)


    def run(self):
        client = self.load('client')

        # run training for n epochs on each client
        client.train()
    
        # send model state for aggregation into central model
        model_state = client.model.state_dict()
        n_data = client.n_train_data
        self.send_data_to_coordinator([model_state, n_data], send_to_self=True, use_smpc=False)
        
        self.store(key='client', value=client)

        if self.is_coordinator:
            return 'aggregate'
        return 'validate'


@app_state('aggregate')
class AggregateState(AppState):

    def register(self):
        self.register_transition('validate', Role.COORDINATOR)


    def run(self):
        # await weights of clients
        list_of_lists = self.gather_data(is_json=False)
        list_of_states = list()
        n_data_clients = list()
        for l in list_of_lists:
            list_of_states.append(l[0])
            n_data_clients.append(l[1])

        # aggregate the weights of the clients by basic fed_avg
        server_state = fed_avg(list_of_states, n_data_clients)

        # distribute aggregated model to the clients
        self.broadcast_data(server_state, send_to_self=True)

        return 'validate'


@app_state('validate')
class ValidateState(AppState):

    def register(self):
        self.register_transition('train', Role.PARTICIPANT)
        self.register_transition('terminal', Role.PARTICIPANT)
        self.register_transition('save_model', Role.COORDINATOR)


    def run(self):
        client = self.load('client')
        config = self.load('config')

        # receive aggregated model by the server
        new_weights = self.await_data(n=1, is_json=False)
        client.model.load_state_dict(new_weights)

        # start validation routine
        client.validate()
        client.epoch = client.epoch + 1

        self.send_data_to_coordinator(client.train_metrics["val_acc"], send_to_self=True, use_smpc=False)

        self.store(key='client', value=client)

        # if training is not finished based on number of epochs
        if self.is_coordinator:
            return 'save_model'
        else:
            if client.epoch != config['epochs']:
                return 'train'
            # training is finished, final testing only on coordinator
            else:
                return 'terminal'


@app_state('save_model')
class SaveModelState(AppState):

    def register(self):
        self.register_transition('train', Role.COORDINATOR)
        self.register_transition('test', Role.COORDINATOR)


    def run(self):
        client = self.load('client')
        config = self.load('config')

        best_val_acc = self.load("best_val_acc")

        # get monitored values per client
        list_of_metrics = self.gather_data(is_json=False)

        # average over all clients and get highest value
        mean_val_acc = np.mean(list_of_metrics)

        # check if newest model is better and save if True
        if mean_val_acc > best_val_acc:
           self.store(key="best_model", value=client.model.state_dict())
           self.store(key="best_val_acc", value=mean_val_acc)

        if client.epoch != config['epochs']:
            return 'train'
        else:
            return 'test'


@app_state('test')
class TestState(AppState):

    def register(self):
        self.register_transition('terminal', Role.BOTH)  

    def run(self):
        config = self.load('config')
        client = self.load('client')
        model_state = self.load("best_model")

        client.model.load_state_dict(model_state)

        # start final evaluation of the model
        test_df = pd.read_csv(os.path.join("/mnt", "input", "test.csv"), index_col=0).T.reset_index().rename(columns={'index': 'image_id'})
        test_loader = DataLoader(ISUPDataset(test_df), batch_size=config["batch_size"], shuffle=True)
        
        path = os.path.join('/mnt', 'output', self.id)
        test(client.model, test_loader, path)

        return 'terminal'
        