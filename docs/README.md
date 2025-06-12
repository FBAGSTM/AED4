# SageMaker Pipelines (AED Version)

## Introduction
This repository provides a `buildspec.yml` file to help you configure and execute SageMaker pipelines for audio event detection pipeline. The guide is designed to be flexible and easy to use, with placeholders for customization. Additionally, this README includes a **Dataset Catalogue** for managing datasets like **ESC-50** and **FSD50K**, and instructions for integrating them into your pipeline.

> [!IMPORTANT]
**File Relocation**:\
This `buildspec.yml` and `pipeline_parameters.py` files will be relocated and executed within the main repository [here](https://github.com/TBALSTM/STM32_AWS_CDK/tree/main/lib/ml/buildspec.yml)\
    - The buildspec.yml file will be placed under the path `<STM32_AWS_IaC>/lib/ml`.
    - The pipeline_parameters.py file will be placed under the path `<STM32_AWS_IaC>/mlops/AED`.<br/><br/>
A placeholder buildspec.yml file currently exists in the STM32_AWS_IaC repoundersitory, which will be replaced by this file. Please keep this in mind when configuring the buildspec commands and their functionality.<br/><br/>
**Pipeline Naming Convention**\
If the user calls this pipeline AED (Audio Event Detection), ensure that the dataset and configurations align with this use case.


## Buildspec File
The `buildspec.yml` file is a critical component of the pipeline, defining the steps required to:

1. Install dependencies.
2. Prepare datasets.
3. Execute the SageMaker pipeline.

This file is structured into phases and includes placeholders for customization, such as dataset names and download commands.

### Artifacts
Artifacts specify the files or directories to be saved after the build process. These are typically outputs generated during the pipeline execution.
```yml
artifacts:
  files:
    - /tmp/ml/**/*
  name: ml
```
* files: Specifies the location of the files to be saved.
* name: Defines the name of the artifact.

### Phase
The pipeline is divided into three main phases: **Install, Pre-Build, and Build**.

---

1. **Install Phase**

This phase sets up the environment and installs dependencies.

```yml
install:
  runtime-versions:
    python: 3.8
  commands:
    - echo "Installing dependencies..."
    - pip install --upgrade .
```
* Purpose: Installs the required Python dependencies by running the [setup.py file](https://github.com/TBALSTM/STM32_AWS_CDK/tree/main/mlops/setup.py) located in the main repository.
* Key Command: `pip install --upgrade .` ensures that the latest version of the dependencies is installed.

---

2. **Pre-Build Phase**

This phase prepares the dataset for the pipeline. It includes downloading the dataset and uploading it to an S3 bucket if it doesn't already exist.

```yml
pre_build:
  commands:
    - echo "Retrieving ML Dataset configuration for use case $USECASE_NAME"

    - DATASET_NAME="<YOUR_DATASET_NAME>"
    - DOWNLOAD_COMMAND="<YOUR_DOWNLOAD_COMMAND>"

    - aws s3api head-object --bucket ${ARTIFACT_BUCKET} --key train/datasets/$DATASET_NAME || NOT_EXIST=true
    - |
      if [ "$NOT_EXIST" = true ]; then
        mkdir -p temp
        echo "Downloading $DATASET_NAME Dataset..."
        eval $DOWNLOAD_COMMAND
        echo "Uploading to S3 bucket under /train/datasets/$DATASET_NAME/ ..."
        aws s3 cp temp s3://${ARTIFACT_BUCKET}/train/datasets/$DATASET_NAME/ --recursive --quiet
      fi
    - echo "Pre-build phase completed."
```

* Purpose: Ensures the dataset is available in the S3 bucket for the pipeline to use.
* Key Placeholders:
  - `DATASET_NAME`: The name of your dataset
  - `DOWNLOAD_COMMAND`: The command to download your dataset\
  [For example, see the Dataset Catalogue](#dataset-catalogue)
* Logic:
  - Checks if the dataset already exists in the S3 bucket.
  - If not, downloads the dataset to a temporary folder (temp) and uploads it to the S3 bucket.

---

3. **Build Phase**

This phase executes the SageMaker pipeline using the prepared dataset.

```yml
build:
  commands:
    - export PYTHONUNBUFFERED=TRUE
    - export SAGEMAKER_PROJECT_NAME_ID="${SAGEMAKER_PROJECT_NAME}-${SAGEMAKER_PROJECT_ID}"
    - |
      run-pipeline --module-name pipelines.stm.pipeline \
        --role-arn $SAGEMAKER_PIPELINE_ROLE_ARN \
        --tags "[{\"Key\":\"sagemaker:project-name\", \"Value\":\"${SAGEMAKER_PROJECT_NAME}\"}, {\"Key\":\"sagemaker:project-id\", \"Value\":\"${SAGEMAKER_PROJECT_ID}\"}]" \
        --kwargs "{\"usecase_name\":\"${USECASE_NAME}\",\"region\":\"${AWS_REGION}\",\"role\":\"${SAGEMAKER_PIPELINE_ROLE_ARN}\",\"default_bucket\":\"${ARTIFACT_BUCKET}\",\"pipeline_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"model_package_group_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"base_job_prefix\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"sagemaker_project_name\":\"${SAGEMAKER_PROJECT_NAME}\"}"
    - aws s3 cp s3://${ARTIFACT_BUCKET}/evaluation/build/ /tmp/ml --recursive
    - echo "Create/Update of the SageMaker Pipeline and execution completed."
```

* Purpose: Runs the SageMaker pipeline with the dataset prepared in the pre-build phase.
* Key Commands:
  - run-pipeline: Executes the pipeline with the specified parameters.
  - aws s3 cp: Copies the evaluation results back to the local directory for further analysis.

> [!NOTE]
Other Environnments Variables are already defined in the **repository STM32_AWS_IaC** during the[SageMaker Pipeline Definition](https://github.com/TBALSTM/STM32_AWS_CDK/tree/main/lib/ml/sagemaker-pipeline.ts). **You do not need to modified them in the file without touching the   all architecture.**

| Variable Name | Description
| :- | :- |
| ``SAGEMAKER_PROJECT_NAME``| The name of the SageMaker project.
| ``SAGEMAKER_PROJECT_ID``| The unique ID of the SageMaker project.
| ``ARTIFACT_BUCKET``| The S3 bucket where artifacts are stored.
| ``SAGEMAKER_PIPELINE_ROLE_ARN``| The ARN of the IAM role used by SageMaker pipelines.
| ``AWS_REGION``| The AWS region where the pipeline is executed.
| ``USECASE_NAME``| The name of the use case (e.g., pipeline-specific identifier).
| ``SAGEMAKER_PROJECT_NAME_ID``| A combination of SAGEMAKER_PROJECT_NAME and SAGEMAKER_PROJECT_ID.

## How to Use the Buildspec
1. **Locate the Buildspec**
  
    The buildspec file is located in this folder as mlops/buildspec.yml.

2. **Customize the Buildspec**
    Open your [YAML builspec file](../mlops/buildspec.yml) and replace the following placeholders with your specific values:

    - `DATASET_NAME`: The name of the dataset you are working with.
    - `DOWNLOAD_COMMAND`: The command to download the dataset in a folder called **./temp**. It can be a multiple line command.

3. **Save the File**

    Save the modified file as `buildspec.yml` in your project directory.

4. **Adjust AI configurations file**

    After modifying the `buildspec.yml` file, ensure that the `user_config.yaml` file is updated to match your new dataset configuration.
    The `user_config.yaml` file is located in:
    - [Here for training](../mlops/audio_event_detection/scripts/training/user_config.yaml)
    - [Here for evaluation](../mlops/audio_event_detection/scripts/evaluate/user_config.yaml)

    > Refer to the README files in those sections for more details on how to configure the `user_config.yaml` file.

5. **Run the Pipeline**

    By pushing your changes to this repository, AWS CodePipeline will automatically be triggered. The pipeline will:
    - Retrieve the new dataset for your specified use case.
    - Execute the SageMaker pipeline with your changes.

## Dataset Catalogue
This section provides a catalogue of datasets that can be used in the pipeline. Each dataset includes its name, download command, and a brief description.

1. `ESC-50`

    `Dataset Name`: ESC-50\
    `Download Command`:
    ```bash
    git clone https://github.com/karolpiczak/ESC-50.git ./temp
    ```
    > A dataset for sound event classification, containing 2,000 labeled environmental audio recordings across 50 classes. [See More Infos](https://github.com/karolpiczak/ESC-50)

2. `FSD50K`

    `Dataset Name`: FSD50K\
    `Download Command`:
    ```bash
    echo 'Downloading FSD50K Dataset...' && \
    echo 'Downloading Dev Audio...' && \
    wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z01 -q && \
    wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z02 -q && \
    wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z03 -q && \
    wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z04 -q && \
    wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z05 -q && \
    wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip -q && \
    zip -q -s 0 FSD50K.dev_audio.zip --out FSD50K.dev_audio-unsplit.zip && \
    echo 'Downloading Eval Audio...' && \
    wget https://zenodo.org/records/4060432/files/FSD50K.eval_audio.z01 -q && \
    wget https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip -q && \
    zip -q -s 0 FSD50K.eval_audio.zip --out FSD50K.eval_audio-unsplit.zip && \
    echo 'Downloading Ground Truth...' && \
    wget https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip -q && \
    echo 'Unzipping...' && \
    unzip -q FSD50K.ground_truth.zip -d temp/ && \
    unzip -q FSD50K.eval_audio-unsplit.zip -d temp/ && \
    unzip -q FSD50K.dev_audio-unsplit.zip -d temp/"
    ```
    > The FSD50K dataset is a large-scale, open dataset for sound event detection and classification. It contains over 51,000 audio recordings sourced from Freesound, a collaborative sound database. The dataset is organized into two main subsets: development (Dev) and evaluation (Eval), and it covers a wide range of sound events across 200 classes based on the AudioSet ontology. [See More Infos](https://zenodo.org/records/4060432)

> [!TIP]
You can find an example of a completed buildspec.yml file for the dataset ESC-50 in the [buildspec-example.yml file](./buildspec-example.yml).

## SageMaker Pipeline Parameters

### Parameters Overview
1. `Dataset Name`
    * Parameter: Extracted dynamically from the `user_config.yaml` file.
    * Location: <usecase_folder>/scripts/training/user_config.yaml
    * Key in YAML: dataset.name
    * Default Value: "AED_Dataset" (if the file or key is missing).

    You can update the name field in the user_config.yaml file to specify a different dataset.

2. `Validation Metrics`

    **Clip-Level Accuracy**
    - Parameter Name: q_clip_level_acc_threshold
    - Default Threshold: 0.5 (can be overridden via default_threshold)
    - fail_step_msg: Execution failed due to clip level acc <
    - json_path: multiclass_classification_metrics.clip_acc.value
    - Name: clip:accuracy
    - Regex: Clip-level accuracy on test set : ([0-9\\.]+)

    **Patch-Level Accuracy**
    * Parameter Name: q_patch_level_acc_threshold
    * Default Threshold: 0.5 (can be overridden via default_threshold).
    * fail_step_msg: Execution failed due to patch level acc <
    * json_path: multiclass_classification_metrics.patch_acc.value
    * Name: patch:accuracy
    * Regex: Patch-level accuracy on test set : ([0-9\\.]+)

    The pipeline uses validation metrics to evaluate the model's performance. These metrics are defined dynamically and include thresholds for success.

    For both parameters, you can:
    - modify the default threshold by overwritting or providing a new value to the default_threshold argument in the get_pipeline_parameters function.
    - modify the fail message step since it's just a SageMaker log.

    > [!WARNING]
    > You can remove these validation parameters entirely; however, doing so may result in an inaccurate pipeline and AI model passing the validation step.

3. `Other Parameter`
    For other parameters or nested parameters, it is recommended not to modify them unless you intend to change the core architecture or functionality of the pipeline.

    | Parameter	| Default Value	| Description	|Can Be Modified?
    | :-: | :-: | :-: | :-:
    | AED_FOLDER_NAME | audio_event_detection |	The folder name for the use case, used to locate the dataset and scripts	| No
    | processing_step_name| Preprocess_`dataset_name`	| The name of the preprocessing step in the pipeline in SageMaker Studio, dynamically includes the dataset name |	Yes
    | modelzoo_version | v1 |	The version of the STM model zoo being used | No