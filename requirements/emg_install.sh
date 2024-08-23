# Get path of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
conda install -c r r-base -y
pip install rpy2
# BaSAR is archived on CRAN, so we need to install it from a local file
conda install -c r r-polynom r-orthopolynom -y
INSTALL_PATH="${SCRIPT_DIR}/requirements/BaSAR_1.3.tar.gz"
INSTALL_STR="install.packages(\"${INSTALL_PATH}\", repos=NULL)"
Rscript -e "${INSTALL_STR}"
