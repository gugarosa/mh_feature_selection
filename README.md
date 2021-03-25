# A Survey on Metaheuristic Approaches to Feature Selection

*This repository holds all the necessary code to run the very-same experiments described in the paper "A Survey on Metaheuristic Approaches to Feature Selection".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

 * `data`: Folder containing the OPF file format datasets;
 * `history`: Folder for saving the output files, such as `.pkl` and `.txt`;
 * `models`
   * `classifiers.py`: Defines the possible classifiers that can be used;
   * `heuristics.py`: Defines the possible meta-heuristics that can be used;
   * `transfers.py`: Defines the possible transfer functions that can be used;
 * `utils`
   * `loader.py`: Loads OPF file format datasets;
   * `optimizer.py`: Wraps the optimization task into a single method;
   * `outputter.py`: Converts the optimization history into readable output files;
   * `targets.py`: Implements the objective functions to be optimized.
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

Please [download](https://www.recogna.tech/files/mh_feature_selection/data.tar.gz) the datasets in the OPF file format and then put them on the `data/` folder.

---

## Usage

### Select Features using Meta-Heuristic Optimization

The first step is to select features using a meta-heuristic optimization. To accomplish such a step, one needs to use the following scripts:

```Python
python bool_select_features.py -h
```

or

```Python
python select_features.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Process Optimization History

After conducting the optimization task, one needs to process its history into readable outputs. Please, use the following scripts to accomplish such a procedure:

```Python
python bool_process_history.py -h
```

or

```Python
python process_history.py -h
```

*Note that this script converts the .pkl optimization history into readable .txt outputs.*

### Classify Selected Features

With the readable outputs in hands, one can now classify the testing set using the selected features:

```Python
python classify_selected_features.py -h
```

### Classify Baseline Features (Optional)

Additionally, it is possible to perform the classification task over the testing set using baseline features (all features), as follows:

```Python
python classify_baseline_features.py -h
```

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell scripts, as follows:

```Bash
./bool_pipeline.sh
```

or

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
