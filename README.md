# PODPAC Examples

Repository that contains files and example Jupyter notebook demonstrating [PODPAC](https://github.com/creare-com/podpac) usage.

## Run Examples on Binder

Binder is a service to run Jupyter Notebooks without any setup.
Click the link below to launch the PODPAC Examples binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/creare-com/podpac-examples/38d2aa570363dab9707f674908d1559edc26314d)

## Run Examples Locally

To run examples locally, you must have a [working version of PODPAC installed](https://podpac.org/install.html)

### Start JupyterLab

1. Clone or download this repository to your computer
2. Browse to the location of the `podpac-examples` repository
3. Launch a JupyterLab session

```bash
$ jupyter lab
```

### Viewing Markdown in JupterLab

To make a Markdown (.md) file more readable in JupyterLab (.e.g this README.md file):
* Right-click the text editor
* Select 'Show Markdown Preview'

A new window will open and show a nicely rendered version of the file.


### Using JupyterLab

The official documentation can be [found here](https://jupyterlab.readthedocs.io/en/stable/user/interface.html), or from the help menu at the top of the JupyterLab browser window.

To launch an example:
* Browse to an example using the file-browser on the left panel
* Double-click the file 
* From the top menu, select `Run --> Run All`

<img src='notebooks/Images/JupyterHowTo.gif' width='80%' />

There are multiple useful keyboard shortcuts, as shown below. For example, for evaluating cells in the notebook:
* `Ctrl-Enter`: Evaluate current cell
* `Shift-Enter`: Evaluate current cell and select the next cell
* `Alt-Enter`: Evaluate current cell and create a new cell after it

<img src='notebooks/Images/notebook-keyboard-shortcuts.jpg' width='80%' />

## Where to go from here

* The [Introduction](notebooks/Introduction.ipynb) notebook introduces the PODPAC library and some Python basics
* The `notebooks/basic_examples` folder contains notebooks with some basic examples
* The `notebooks/demos` folder contains notebooks with some more involved examples
* The `notebooks/developer` folder contains notebooks with technical implementation details for developers
* The `notebooks/presentations` folder contains notebooks of presentations given at various workshops, conferences, science team meetings, and other venues
* The `data` folder contains raw datafiles used for some of the examples.
