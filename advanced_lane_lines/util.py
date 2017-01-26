from configparser import ConfigParser
config = ConfigParser()
config.read('config.cfg')

def get_config_tuples(name):
    tuples = config.items(name)
    return {item[0]: tuple(map(float, item[1].split())) for item in tuples}
