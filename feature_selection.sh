# Defining a constant to hold the possible meta-heuristics
MH=("pso")

# Defining the classifier to be used
CLF="lr"

# Defining the transfer function to be used
TF="t1"

# Defining the number of agents
N_AGENTS=10

# Defining the number of iterations
N_ITER=3

# Defining the seed
SEED=0

# Creating a loop of meta-heuristics
for M in "${MH[@]}"; do
    # Performs the feature selection procedure
    python select_features.py ${M} ${CLF} ${TF} -n_agents ${N_AGENTS} -n_iter ${N_ITER} -seed ${SEED}

    # Process the optimization history
    python process_history.py ${M}_${CLF}_${TF}_${N_AGENTS}ag_${N_ITER}iter_${SEED} ${TF}

    # Performs the final classification over the testing set using selected features
    python test_selected_features.py ${CLF} ${M}_${CLF}_${TF}_${N_AGENTS}ag_${N_ITER}iter_${SEED} -seed ${SEED}
done