import yaml

yaml_file = 'config/config.yaml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)
    
print(config)