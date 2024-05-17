import yaml
from easydict import EasyDict as edict


def get_config(args):
    config_dir = f'./config/{args.config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = args.seed
    config.ae_ckpt = f'{args.ae_ckpt}'
    config.run_name = args.run_name
    config.stage = args.stage
    
    return config