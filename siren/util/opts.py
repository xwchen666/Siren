import argparse
from ruamel.yaml import YAML
import importlib

def parse_args_and_load_module(configuration_file, key, description=''):
    parser = argparse.ArgumentParser(description=description)
    yaml = YAML(typ='safe')
    with open(configuration_file) as f:
        params = yaml.load(f)
        params = params[key]
        param_dict = {}
        for param_name, vals in params['parameters'].items():
            if vals.get('short_name', None):
                names = ['-' + vals['short_name'], '--' + param_name]
            else:
                names = ['--' + param_name]
            
            param_dict["action"]   = 'store_true' if vals['type'] is 'bool' else 'store'
            param_dict["required"] = 'default' not in vals
            param_dict["default"]  = vals.get('default', None)
            param_dict["choices"]  = vals.get('choices', None)
            param_dict["type"]     = getattr(__import__('builtins'), vals['type'])
            param_dict["help"]     = vals.get("help", None)

            parser.add_argument(*names, **param_dict)

        options = parser.parse_known_args()[0]
        module_name = params['module_name']
        class_name  = params['class_name']
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(**vars(options))
