
## Visual Analytics Final Project: Understanding mortality in Catalan road accidents

Felipe López, Oriol Pont

### Project description and expected benefits

We have readily available data from all accidents in Catalan roads that involved serious injuries and deaths, thanks to a digitalization task done by our public administration. Direct inspection of this data makes it hard to discover the patterns and causes behind the outcomes, so we want to leverage Visual Analytics and ML techniques to analyze, visualize and use the data to predict death risk in new accidents. In short, we want to understand what factors and features are more relevant to help predicting whether an accident with a specific set of characteristics involves deadly victims.

Not only this would allow running simulations to predict outcomes in **hypothetical scenarios**, or even asses how likely were the observed outcomes of a given **past** accident: the learned criteria from such prediction model would allow us taking intelligent measures **in the present** to reduce the overall risk of deadly accidents, which in turn might save **future** lives.

-----

### Required data sources

  * [https://analisi.transparenciacatalunya.cat/Transport/Accidents-de-tr-nsit-amb-morts-o-ferits-greus-a-Ca/rmgc-ncpb/about\_data](https://www.google.com/search?q=https://analisi.transparenciacatalunya.cat/Transport/Accidents-de-tr-nsit-amb-morts-o-ferits-greus-a-Ca/rmgc-ncpb/about_data)

A 24.5k-row dataset collecting data about traffic accidents involving serious injuries or deaths. It is provided and annually updated by the Interior Department since 2010. Each record conforms a mix of categorical and numerical data, originating from the Accident Police Report.

The snapshot we will be using (exported on 2025-12-04) is already present in this repository as `data/accidents_catalonia_2010_2023.csv`.

### Expected results/delivery/output

We will focus on delivering an interactive Streamlit webapp that enables running mortality simulations on different scenarios, using our model trained from this dataset.

To work with the data and train this model, we will use a Jupyter Notebook (.ipynb), where we will perform an initial Exploratory Data Analysis, prepare the dataset for ML (cleaning, standarizing,...), prepare several training runs and generate the final output, and use XAI techniques to showcase what has been learned by the model.

-----

### Visualization method

We will use plots to showcase the model evals, and an interactive interface that allows running predictions and simulations using a few clicks instead of filling large config files. The specificities of the visualizations used will be determined during the model development according to how valuable the insights each showcase are, discarding all the plots/histograms/heatmaps/... that don't present valuable information to the reader.

-----

### Repository structure

```plaintext
catalan-traffic-accident-analysis/
├── data/
├── notebooks/          # .ipynb notebooks for EDA and model training
├── src/                # Streamlit app
├── models/
├── output/             # for static visualizations
├── README.md
├── LICENSE
└── pyproject.toml      # Project dependencies
```

-----

### Development setup

This project uses `uv` for fast and reliable Python environment management. All dependencies (notebooks, Jupyter, Streamlit, ML libraries) are specified in `pyproject.toml`.

#### Prerequisites

* Python 3.10 or higher
* `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

#### Setting up the environment

1. Clone the repository and navigate to the project directory
2. Create and activate the virtual environment:

```bash
uv sync
source .venv/bin/activate
```

On Windows:

```bash
uv sync
.venv\Scripts\activate
```

This will:

* Create a `.venv` virtual environment
* Install all dependencies including Jupyter, Streamlit, pandas, scikit-learn, and XAI libraries

#### Updating dependencies

To add new packages, edit `pyproject.toml` and run:

```bash
uv sync
```
