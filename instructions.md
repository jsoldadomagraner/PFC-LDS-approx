The models can be fitted locally or using the cluster scheduler SLURM

To fit the models locally, run for example in the command line:\
python trainlds.py 2 1\
This fits a model of latent dimensionality 2 and will save the fitted output with the label 1 (run 1), so that you can save multiple fitted model instances with random initializations.\
python cvlds.py 2 1\
This fits a model of latent dimensionality 2 with the data condition 1 left out. This script also generates model predictions for the left-out condition.\
For all other specs, such as input dimensionality of the models, model type (with different contextual constraints), cost penalties (e.g. input norm penalty), etc, specify this within the trainlds.py and cvlds.py scripts.

To fit the models using SLURM, run in the command line:\
bash trainSLURMwrapper\
bash cvSLURMwrapper\
This will send parallel jobs to the SLURM scheduler to fit multiple models, e.g. with different latent dimensionalies, multiple runs (random inits), and different left-out conditions for cross-validation.\
Specify those numbers within the SLURM files. Note that screen outputs and errors are saved, to facilitate debuging. Add your path to save such files at the beginning of the scripts trainSLURM and cvSLURM.

The fiting results are saved in '/yourhomepath/fitted_models/', to change the paths edit mypaths.py and pathstfr.py
