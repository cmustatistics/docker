from run import demo_experiments,\
    GP,VFE,LIA,LIA_TO_LIF,LIA_TO_PP,LIF,SVGP

from plot import plot_syn_demo
seed = 123

###########################################
# In order to have a demo that can be quickly performed
# in computers without a GPU, each method will be run once
# for a low dimensional dataset
###########################################

dimensions = 3 # set the dimension of the problem
number_of_inducing_points = 50 # set the number of inducing points

for experiment in demo_experiments(dimensions,number_of_inducing_points,seed):

        LIA_TO_PP(experiment,seed)

        SVGP(experiment,seed)

        # Other methods can be easily run.
        # For instance,
        #   LIA_TO_LIF(experiment,seed)
        #   LIA(experiment,seed)
        #   GP(experiment,seed)
        #   VFE(experiment,seed)

###########################################
# Plot the results
###########################################

plot_syn_demo(dimensions=dimensions,
              number_of_inducing_points=number_of_inducing_points)
