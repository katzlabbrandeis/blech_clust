# Mark blech_exp_info.py as having been successfully completed
import sys
import os

# Get blech_clust path
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
sys.path.append(blech_clust_dir)  # noqa
from utils.blech_utils import imp_metadata, pipeline_graph_check  # noqa
logging_script_path = os.path.join(blech_clust_dir, 'blech_exp_info.py')

metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

# Perform pipeline graph check
this_pipeline_check = pipeline_graph_check(dir_name)

# Write success to log
log_dir = os.path.join(dir_name, 'logs')
os.makedirs(log_dir, exist_ok=True)
this_pipeline_check.write_to_log(os.path.join(log_dir, 'execution.log'), 'attempted')
this_pipeline_check.write_to_log(logging_script_path, 'completed')
