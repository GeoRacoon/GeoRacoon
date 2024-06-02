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
mamba run pip install -e .

# ###
# The remaining two lines are only needed if you intend to use `landiv` with
# jupyter, which is not what we currently do.

#mamba install --name landiv ipykernel -qy
#ipython kernel install --user --name landiv
