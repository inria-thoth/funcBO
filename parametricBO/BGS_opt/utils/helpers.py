import os
import torch
import importlib
import omegaconf
import nvidia_smi

#   Adapted from: https://github.com/AntheaL/bilevel_opt_code/blob/main/bilevel_opt/helpers.py

class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self

def to_type(data,dtype):
    if dtype==torch.double:
        return data.double()
    else:
        return data.float()

def config_to_instance(config_module_name="name",**config):
    module_name = config.pop(config_module_name)
    attr = import_module(module_name)
    if config:
        attr = attr(**config)
    return attr


def config_to_dict(config):
    done = False
    out_dict = {}
    for key, value in config.items():
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            out_dict[key] = config_to_dict(value)
        else:
            out_dict[key] = value
    return Config(out_dict)


def import_module(module_name):
    module, attr = os.path.splitext(module_name)
    try:
        module = importlib.import_module(module)
        return getattr(module, attr[1:])
    except:
        try:
            module = import_module(module)
            return getattr(module, attr[1:])
        except:
            return eval(module+'.'+attr[1:])

def init_model(model, model_path,dtype, device):
    if model_path:
        #state_dict_model = torch.load(model_path, map_location='cpu')
        #model = model.load_state_dict(state_dict_model).to('cpu')
        model = torch.load(model_path)

    model = to_type(model, dtype)
    model = model.to(device)

    return model

def assign_device(device):
    if device >-1:
        device = (
            'cuda:'+str(device) 
            if torch.cuda.is_available() and device>-1 
            else 'cpu'
        )
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device


def get_dtype(dtype):
    if dtype==64:
        return torch.double
    elif dtype==32:
        return torch.float
    else:
        raise NotImplementedError('Unkown type')



def get_gpu_usage(device):
    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(device, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

    nvidia_smi.nvmlShutdown()

