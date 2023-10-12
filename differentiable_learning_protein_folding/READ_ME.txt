SIMULATED ANNEALING (SA):
SA notebook is split into two parts:
	 part 1 - protein structure prediction
	 part 2 - amino acid sequence prediction

The function simulated_annealing() requires all previously defined functions to be ran as it relies on them.
The user can choose a sequence of amino acids, e.g. "HPHPPH" and store this in the "seq" variable near the end of part 1, 
right after the simulated_annealing() function is defined.


DIFFERENTIABLE LEARNING (DL):
DL notebook is all in one notebook cell block. If hyperparameter tuning is desired, the user can leave code as is 
and specify ranges for parameters to test through. If not desired, then the user can comment these out and define the 
parameter variables as single values instead, e.g. as "alpha = 0.01, too_close_pen = 5, ...".

Once hyperparameters are defined, then the user can simply run the code for the desired number of iterations (final for loop, currently set to 2000).

Start and end states of protein structures are plotted as well as energy curves to witness how the funciton is optimizing.
