# Defining the dataset to be used
DATA="wine"

# Defining the validation split
# Note that we firstly split train and test, and then train is splitted again into train and val
VAL_SPLIT=0.5

# Defining the test split
TEST_SPLIT=0.2

# Defining a constant to hold the possible meta-heuristics
MH=("bmrfo")

# Defining the classifier to be used
CLF="opf"

# Defining the number of agents
N_AGENTS=10

# Defining the number of iterations
N_ITER=5

# Defining the seed
SEED=0

# Creating a loop of meta-heuristics
for M in "${MH[@]}"; do
    # Performs the feature selection procedure
    python bool_select_features.py ${DATA} ${M} ${CLF} -n_agents ${N_AGENTS} -n_iter ${N_ITER} -val_split ${VAL_SPLIT} -test_split ${TEST_SPLIT} -seed ${SEED}

    # Process the optimization history
    python bool_process_history.py ${DATA}_${VAL_SPLIT}_${TEST_SPLIT}_${M}_${CLF}_${N_AGENTS}ag_${N_ITER}iter_${SEED}

    # Performs the classification over the testing set using selected features
    python classify_selected_features.py ${DATA} ${CLF} ${DATA}_${VAL_SPLIT}_${TEST_SPLIT}_${M}_${CLF}_${N_AGENTS}ag_${N_ITER}iter_${SEED} -val_split ${VAL_SPLIT} -test_split ${TEST_SPLIT} -seed ${SEED}
done

# Performs the classification over the testing set using baseline features
python classify_baseline_features.py ${DATA} ${CLF} -val_split ${VAL_SPLIT} -test_split ${TEST_SPLIT} -seed ${SEED}