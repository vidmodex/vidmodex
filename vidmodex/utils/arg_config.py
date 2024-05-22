import os
import inspect
import argparse
import yaml
import glob
import re
        
def make_parser(parser, config_args):
    for k in dir(config_args):
        if k.startswith("_"):continue
        v = getattr(config_args, k)
        if isinstance(v, (float, int, str, list, dict, tuple)):
            parser.add_argument(f"--{k}", type=type(v), default=v)
    return parser

def config_update(parsed_args, config_args, experiment_name="victim_train"):
    with open(parsed_args.config, "r") as f:
        custom_config = yaml.load(f, Loader=yaml.FullLoader)
    
    if experiment_name == "victim_train":
        custom_config["experiment"] = f'{custom_config["model"]["victim"]["name"]}_target_{custom_config["train_dataset"]["name"]}'
        
    if custom_config.get("config_args"):
        yaml_config_args = custom_config["config_args"]
        for k in yaml_config_args.keys():
            setattr(config_args, k, yaml_config_args[k])
    
    for k in dir(parsed_args):
        if k not in dir(config_args):continue
        v = getattr(parsed_args, k)
        setattr(config_args, k, v)
        
    if config_args.resume_ckpt is None and config_args.resume:
        resume_ckpt = f'runs/{custom_config["experiment"]}'
        
        subfolders = glob.glob(f'{resume_ckpt}/version_*')
        version_numbers = [int(re.search(r'version_(\d+)', folder).group(1)) for folder in subfolders]
        version_numbers.sort(reverse=True)
        for version in version_numbers:
            checkpoint_folder = f'runs/{custom_config["experiment"]}/version_{version}/checkpoints'
            if os.path.exists(checkpoint_folder) and glob.glob(checkpoint_folder + '/*.ckpt'):
                resume_ckpt = glob.glob(checkpoint_folder + '/*.ckpt')[0]
                break
        config_args.resume_ckpt = resume_ckpt
        
    return custom_config

