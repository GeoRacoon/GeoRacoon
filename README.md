# Landiv Blur

Package to compute diversity measures in land-cover type maps

## Installation

_Note:_
_This package relies on [rasterio](https://rasterio.readthedocs.io/en/latest/index.html)_
_which partially depends on [libgdal](https://gdal.org/)._
_If you follow the installation instructions below you will attempt to install_
_rasterio from the Python Package Index in which chase the libgdal library_
_will be shipped along._
_However, if you encounter any issues with the installaiton of rasterio, head_
_over to the [rasterio installation insructions](https://rasterio.readthedocs.io/en/stable/installation.html) for more details._


To install `landiv_blur`:

1. Clone this repository
1. `cd` into the repository
1. Run `pip3 install .`


## Usage

Head over to the [examples/](examples/) folder for some usage examples.

Installed along with the `landiv_blur` package is also a command line executable
`landiv` that can be used to process `.tif` files in a parallelized manner.
After installation, type `landiv --help` in your terminal for further details
on how to use it.

<!--- quickstart --->

### Running on a SLURM cluster (like cluster.s3it.uzh.ch)

In order to apply the filters to sizeable maps we are in need of adequate
resources that are provided, for example, by the SLURM cluster maintained by
s3it.uzh.ch.

The approach we chose here is as follows:

- We submit a single job to the cluster that requires multiple CPU's and enough RAM to process a map in parallel.
- `landiv` then takes care of splitting up the map into multiple blocks (or `views`) and uses python's
  `multiprocessing` library to efficiently make use of the reserved CPUs. `landiv` also handles the recombination
  of the resulting blocks back into a single file.

This approach distributes the workload much like one would parallelize on a single multi-core
server and can, in fact, be used in an identical approach on a laptop or workstation.

#### Setup environment on the cluster

1. Login to cluster.s3it.uzh.ch with `ssh -l shortname cluster.s3it.uzh.ch`
1. Create ssh key-pair with `ssh-keygen`. Make sure to **set a password
   protection** for the key.
1. Add the just generate public key (the \<something\>.pub) to your ssh keys
   on https://git.math.uzh.ch/-/profile/keys
1. Clone the landiv project into your home folder on the cluster
1. Build and configure the python environment to use the `landiv_blur` package
   and the `landiv` command in particular:
   ```
   chmod +x landiv_env_setup.sh
   bash ./landiv_env_setup.sh
   ```

#### Data and storage of output files

We have to deal with two constraints on the cluster:

- storage capacity
- accessibility

Given that we deal with several GBs of data, produce even more as output
files and we are at least two to use the cluster, the best option seems to use
the groups share under `/shares/niklaus.ieu.uzh`.

How exactly we want to structure the data and output files there remains to be decided.
For now the land-cover type maps are located under `/shares/niklaus.ieu.uzh/first_approach/Europe/landcover`
and the output files are put under `/shares/niklaus.ieu.uzh/first_approach/Europe/output`, if we use
the [example script](examples/console_launcher.sh).


#### Launching jobs on the cluster
1. Create a launcher script that reserves the desired resources and calls `landiv`.
   _Have a look at [examples/console_launcher.sh](examples/console_launcher.sh) if you
   are uncertain how such a script should look like._

   The recommended configuration (which is also used in the mentioned example
   script) is as follows:

   - 32 CPUs
   - 120GB RAM
   - Max duration: 4 Hours
   - Block size 4000x4000 pixels

   _For further information see issue #27_
1. Launch your script (here we assume it's called `console_launcher.sh`) with

   ```
   sbatch console_launcher.sh
   ```
1. Now you can monitor the progress.
   
   In the folder you launched the previous command a file called `slurm-<job_id>.out` will be created
   that will gather all the output of the `landiv` script.

   Here are a few options to monitor your job:

   - Run `squeue --me` to see the status of your job.
   - Run `tail -f slurm-<job_id>.out` to have a live view of the output that the job is generating
   - Run `sacct -j <job_id>` to see an overview of the consumed resources, job state, etc.
     
     For a better formatting of the output, I recommend configuring the `sacct` command with:

     ```
     export SACCT_FORMAT="JobID%20,JobName,Elapsed,CPUTime,State,MaxRSS,AllocTRES%32,ExitCode,User,Partition,NodeList"
     ```
   Once the jobs are terminated you will find the resulting maps in the location you specified in the `--output`
   parameter of the `landiv` command.


---
---


## Exemplary output

![France-CH border](./results/test_france.png)

### Individual layers

![France-CH border](./results/test_france_layers.png)

#### Individual layers with Gaussian filter

![France-CH border](./results/test_france_layers_filtered_1.0.png)
_sigma = 1_
![France-CH border](./results/test_france_layers_filtered_10.0.png)
_sigma = 10_
![France-CH border](./results/test_france_layers_filtered_40.0.png)
_sigma = 40_

### Entropy after diffusion

![France-CH border](./results/test_france.png)
![France-CH border](./results/test_france_layers_entropy_1.0.png)
_sigma = 1_

---

![France-CH border](./results/test_france_layers_entropy_10.0.png)
_sigma = 10_

---

![France-CH border](./results/test_france_layers_entropy_40.0.png)
_sigma = 40_

---

<br>

<p align="center">
<img 
   alt="Test area FR-CH border"
   src="./results/test_france.png"
   height="900"
/>
<img 
  alt="Test area FR-CH border - entropy"
  src="./results/test_france_layers_entropy_40.0.png"
  height="900"
/>
</p>

<br>

---
---

### Bigger map

<br>

<p align="center">
<img 
   alt="all france"
   src="./results/all_france.png" 
   height="900"
/>
<img 
  alt="All france - entropy sigma 200"
  src="./results/all_france_layers_entropy_200.0.png"
  height="900"
/>
</p>

<br>


In principle this approach can be adapted also for landscape blocks consisting of a block of pixels and thus an initial distributions with resulting entropy.
Therefore, there are two way to include scale effects:

- the standard deviation of the diffusion kernel
- the landscape block size
