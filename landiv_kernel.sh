# Load mamba (conda) module
module load mamba

# Create and activate your environment
mamba create --name landiv
source activate landiv

# Add packages to your environment
mamba install --name landiv --file requirements.txt -qy

# Install our package
mamba install --name landiv pip -qy
# Make sure ~/.local/bin is picked up
export PATH="$HOME/.local/bin:$PATH"
# install landiv in developer mode
mamba run pip install -e . -qy

# Install the tools to add a custom kernel
mamba install --name landiv ipykernel -qy

# Add your environment to the kernel list
ipython kernel install --user --name landiv
