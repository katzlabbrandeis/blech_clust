import sys
import argparse
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

parser = argparse.ArgumentParser(description = 'Test unit locations')
parser.add_argument('dir_name',  help = 'Directory containing data files')
args = parser.parse_args()

if args.dir_name:
    dir_path = args.dir_name
    if dir_path[-1] != '/':
        dir_path += '/'
else:
    raise Exception('dir_name not provided')

#dir_path = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191029_171714/'

dat = ephys_data(dir_path)
dat.get_region_units()
print(dict(zip(dat.region_names, dat.region_units)))
