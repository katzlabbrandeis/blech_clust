# Mark blech_exp_info.py as having been successfully completed
import sys
import os

# Get blech_clust path
script_path = os.path.realpath(__file__)
blech_clust_dir = os.path.dirname(os.path.dirname(script_path))
logging_script_path = os.path.join(blech_clust_dir, 'blech_exp_info.py')

sys.path.append(blech_clust_dir)
from utils.blech_utils import imp_metadata, pipeline_graph_check

metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name

# Perform pipeline graph check
this_pipeline_check = pipeline_graph_check(dir_name)

# Write success to log
this_pipeline_check.write_to_log(logging_script_path, 'attempted')
this_pipeline_check.write_to_log(logging_script_path, 'completed')
