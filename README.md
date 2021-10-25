MIMIC-IV-ED Benchmark
=========================

Python workflow for generating benchmark datasets and machine learning models from the MIMIC-IV-ED database.

## Table of contents
* [General info](#general-info)
* [Structure](#structure)
* [Requirements and Setup](#requirements-and-setup)
* [Workflow](#workflow)
    1. [Benchmark Data Generation](#1-benchmark-data-generation)
    2. [Cohort Filtering](#2-cohort-filtering)
    3. [Outcome and Model Selection](#3-outcome-and-model-selection)
    4. [Model Evaluation](#4-model-evaluation)
* [Acknowledgements](#acknowledgements)
* [Citation](#citation)

## General info

Clinical decisions in the emergency department are  . Unsurprisingly, machine learning based clinical prediction models have been widely adopted in the field of 

In parellel to the rise of clinical prediction models, there has also been a rapid increase in adoption of Electronic Health records (EHR) for patient data. The Medical Information Mart for Intensive Care ([MIMIC)-IV]((https://physionet.org/content/mimiciv/1.0/)) and [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/1.0/) are examples of EHR databases that contain a vast amount of patient information.

There is therefore a need for publicly available benchmark datasets and models that allow researchers to produce comparable and reproducible results. 

In 2019, a [MIMIC-III benchmark](https://github.com/YerevaNN/mimic3-benchmarks) was published for the previous iteration of the MIMIC database. 

Here, we present a workflow that generates a benchmark dataset from the MIMIC-IV-ED database and constructs benchmark models for three ED-based prediction tasks.


## Structure

The structure of this repository is detailed as follows:

- `Benchmark_scripts/...` contains the scripts for benchmark dataset generation (master_data.csv).
- `Benchmark_scripts/...` contains the scripts for building the various task-specific benchmark models.
-  

## Requirements and Setup
MIMIC-IV-ED and MIMIC-IV databases are not provided with this repository and are **required** for this workflow. MIMIC-IV-ED can be downloaded from [https://physionet.org/content/mimic-iv-ed/1.0/](https://physionet.org/content/mimic-iv-ed/1.0/) and MIMIC-IV can be downloaded from [https://physionet.org/content/mimiciv/1.0/](https://physionet.org/content/mimiciv/1.0/).

***NOTE** It should be noted that upon downloading and extracting the MIMIC databases from their compressed files, the directory `/mimic-iv-ed-1.0/ed` should be moved/copied to the directory containing MIMIC-IV data `/mimic-iv-1.0`.

## Workflow

The following sub-sections describe the sequential modules within the MIMIC-IV-ED workflow and how the should ideally be run.

Prior to these steps, this repository, MIMIC-IV-ED and MIMIC-IV should be downloaded and set up locally. 

### 1. Benchmark Data Generation
~~~
python extract_master_dataset.py {mimic_iv_path} {output_path} {icu_transfer_timerange} {next_ed_visit_timerange}
~~~
**Arguements**:

- `mimic_iv_path` : Path to directory containing MIMIC-IV data. Refer to [Requirements and Setup](#requirements-and-setup) for details.
- `output_path ` : Path to output directory.
- `icu_transfer_timerange` : Timerange in hours for ICU transfer outcome. Default set to 12. 
- `next_ed_visit_timerange` : Timerange in days days for next ED visit outcome. Default set to 3.


### 2. Cohort Filtering
### 3. Outcome and Model Selection
### 4. Model Evaluation

## Acknowledgements

## Citation


