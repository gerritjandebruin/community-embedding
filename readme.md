This repository contains our current research into obtaining baseline performance for node embeddings in graphs.

# Requirements
Create a new virtual environment in conda:
```
conda create -n community-embedding node2vec jupyter scikit-learn networkx pandas tqdm seaborn xgboost jupyter_contrib_nbextensions
```

# Generating the Github page
These instructions allow seeing the Jupyter Notebook at http://gerritjandebruin.nl/community-embedding/index.html.
When you are in the virtual environment:
```bash
jupyter nbconvert results.ipynb --to html_toc
mv results.html docs/index.html
mv results_files docs/
```

# Files
The folder `tools/` provide most code used in `results.ipynb`.
The data is stored in `citeseer/`.

# Contributions
- [António Pereira Barata](https://www.universiteitleiden.nl/en/staffmembers/antonio-pereira-barata)
- [Gerrit-Jan de Bruin](http://gerritjandebruin.nl/)