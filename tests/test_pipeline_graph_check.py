
"""
Tests for the pipeline_graph_check functionality in blech_utils.py.
Verifies that --overwrite_dependencies argument has been added to all scripts.
"""
import os
import sys

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestOverwriteDependenciesArgument:
    """Test that scripts accept --overwrite_dependencies argument"""

    def test_blech_init_has_argument(self):
        """Test that blech_init.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/blech_init.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_blech_make_arrays_has_argument(self):
        """Test that blech_make_arrays.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/blech_make_arrays.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_blech_process_has_argument(self):
        """Test that blech_process.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/blech_process.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_blech_post_process_has_argument(self):
        """Test that blech_post_process.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/blech_post_process.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_blech_common_avg_reference_has_argument(self):
        """Test that blech_common_avg_reference.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/blech_common_avg_reference.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_blech_exp_info_has_argument(self):
        """Test that blech_exp_info.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/blech_exp_info.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_blech_units_plot_has_argument(self):
        """Test that blech_units_plot.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/blech_units_plot.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_blech_units_characteristics_has_argument(self):
        """Test that blech_units_characteristics.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/blech_units_characteristics.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_utils_infer_rnn_rates_has_argument(self):
        """Test that utils/infer_rnn_rates.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/utils/infer_rnn_rates.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_utils_qa_utils_drift_check_has_argument(self):
        """Test that utils/qa_utils/drift_check.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/utils/qa_utils/drift_check.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_utils_qa_utils_elbo_drift_has_argument(self):
        """Test that utils/qa_utils/elbo_drift.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/utils/qa_utils/elbo_drift.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content

    def test_utils_qa_utils_unit_similarity_has_argument(self):
        """Test that utils/qa_utils/unit_similarity.py accepts --overwrite_dependencies argument"""
        with open('/workspace/project/blech_clust/utils/qa_utils/unit_similarity.py', 'r') as f:
            content = f.read()

        assert '--overwrite_dependencies' in content


class TestPipelineGraphCheckClass:
    """Test that pipeline_graph_check class has the new functionality"""

    def test_blech_utils_has_overwrite_dependencies_in_init(self):
        """Test that blech_utils.py pipeline_graph_check __init__ has overwrite_dependencies parameter"""
        with open('/workspace/project/blech_clust/utils/blech_utils.py', 'r') as f:
            content = f.read()

        # Check that the __init__ method has the parameter
        assert 'def __init__(self, data_dir, overwrite_dependencies=False):' in content
        assert 'self.overwrite_dependencies = overwrite_dependencies' in content

    def test_blech_utils_check_previous_has_override_logic(self):
        """Test that check_previous method has the override logic"""
        with open('/workspace/project/blech_clust/utils/blech_utils.py', 'r') as f:
            content = f.read()

        # Check that the check_previous method has the override logic
        assert 'if self.overwrite_dependencies:' in content
        assert "input('Continue anyway? ([y]/n) :: ')" in content
