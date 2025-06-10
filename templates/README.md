# Buildspec Template for SageMaker Pipelines
This repository provides a buildspec template (buildspec-template.yml) to help you configure and execute SageMaker pipelines. The template is designed to be flexible and easy to use, with placeholders for customization.

## How to Use the Template
1. **Locate the Template**

The template file is located in this folder as buildspec-template.yml. Copy this file to your project directory.

2. **Customize the Template**

Open the file mlops/buildspec.yml and replace the following placeholders with your specific values:

- <DATASET_NAME>: The name of the dataset you are working with (e.g., ESC-50).
- <DOWNLOAD_COMMAND>: The command to download the dataset (e.g., git clone <REPO_URL>).
- **Note**: All other variables (e.g., SAGEMAKER_PROJECT_NAME, SAGEMAKER_PROJECT_ID, ARTIFACT_BUCKET, etc.) are already defined in the **common repository STM32_AWS_CDK** in the file lib/ml/sagemaker-pipeline.ts. You do not need to modify or define them manually here.

3. **Save the File**

Save the modified file as buildspec.yml in your project directory.

4. **Adjust the user_config.yaml File**

After modifying the buildspec.yml file, ensure that the user_config.yaml file is updated to match your new dataset configuration.
The user_config.yaml file is located in:
- For training: mlops/audio_event_detection/scripts/training/user_config.yaml
- For evaluation: mlops/audio_event_detection/scripts/evaluate/user_config.yaml

Refer to the README files in those sections for more details on how to configure the user_config.yaml file.

5. **Run the Pipeline**

By pushing your changes to this repository, AWS CodePipeline will automatically be triggered. The pipeline will:
- Retrieve the new dataset for your specified use case.
- Execute the SageMaker pipeline with your changes.

## Template Overview
The buildspec-template.yml file is structured as follows:

### Artifacts
Specifies the files or directories to be saved after the build:

```yml
artifacts:
  files:
    - /tmp/ml/**/*
  name: ml
```

### Phases
The pipeline is divided into three phases:

#### Install Phase
Installs dependencies and sets up the environment:

```yml
install:
  runtime-versions:
    python: 3.8
  commands:
    - echo "Installing dependencies..."
    - pip install --upgrade .
```

#### Pre-Build Phase
Prepares the dataset. You need to customize the following placeholders:

- <DATASET_NAME>: The name of the dataset (e.g., ESC-50).
- <DOWNLOAD_COMMAND>: The command to download the dataset (e.g., git clone <REPO_URL>).

Example:
```yml
pre_build:
  commands:
    - echo "Retrieving ML Dataset configuration for use case $USECASE_NAME"

    - DATASET_NAME="<DATASET_NAME>"  # Example: "ESC-50"
    - DOWNLOAD_COMMAND="<DOWNLOAD_COMMAND>"  # Example: "git clone <REPO_URL> ./temp"

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

#### Build Phase
Executes the SageMaker pipeline with the dataset donwloaded in pre-build phase as source.

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

**Variables Defined in Common Repository**:
The following variables are already defined in the repository STM32_AWS_CDK, at location: lib/ml/sagemaker-pipeline.ts and do not need to be modified in the template:

| Variable Name | Description
| :- | :- |
| ``SAGEMAKER_PROJECT_NAME``| The name of the SageMaker project.
| ``SAGEMAKER_PROJECT_ID``| The unique ID of the SageMaker project.
| ``ARTIFACT_BUCKET``| The S3 bucket where artifacts are stored.
| ``SAGEMAKER_PIPELINE_ROLE_ARN``| The ARN of the IAM role used by SageMaker pipelines.
| ``AWS_REGION``| The AWS region where the pipeline is executed.
| ``USECASE_NAME``| The name of the use case (e.g., pipeline-specific identifier).
| ``SAGEMAKER_PROJECT_NAME_ID``| A combination of SAGEMAKER_PROJECT_NAME and SAGEMAKER_PROJECT_ID.
                                                               
**Example Configuration**

Here’s an example of a completed buildspec.yml file for the dataset ESC-50:

```yml
version: 0.2

artifacts:
  files:
    - /tmp/ml/**/*
  name: ml

phases:
  install:
    runtime-versions:
      python: 3.8
    commands:
      - pip install --upgrade .

  pre_build:
    commands:
      - DATASET_NAME="ESC-50"
      - DOWNLOAD_COMMAND="git clone https://github.com/karolpiczak/ESC-50.git ./temp"
      - aws s3api head-object --bucket ${ARTIFACT_BUCKET} --key train/datasets/$DATASET_NAME || NOT_EXIST=true
      - |
        if [ "$NOT_EXIST" = true ]; then
          mkdir -p temp
          eval $DOWNLOAD_COMMAND
          aws s3 cp temp s3://${ARTIFACT_BUCKET}/train/datasets/$DATASET_NAME/ --recursive --quiet
        fi

  build:
    commands:
      - export PYTHONUNBUFFERED=TRUE
      - export SAGEMAKER_PROJECT_NAME_ID="${SAGEMAKER_PROJECT_NAME}-${SAGEMAKER_PROJECT_ID}"
      - |
        run-pipeline --module-name pipelines.stm.pipeline \
          --role-arn $SAGEMAKER_PIPELINE_ROLE_ARN \
          --tags "[{\"Key\":\"sagemaker:project-name\", \"Value\":\"${SAGEMAKER_PROJECT_NAME}\"}, {\"Key\":\"sagemaker:project-id\", \"Value\":\"${SAGEMAKER_PROJECT_ID}\"}]" \
          --kwargs "{\"usecase_name\":\"ESC-50\",\"region\":\"${AWS_REGION}\",\"role\":\"${SAGEMAKER_PIPELINE_ROLE_ARN}\",\"default_bucket\":\"${ARTIFACT_BUCKET}\",\"pipeline_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"model_package_group_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"base_job_prefix\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"sagemaker_project_name\":\"${SAGEMAKER_PROJECT_NAME}\"}"
      - aws s3 cp s3://${ARTIFACT_BUCKET}/evaluation/build/ /tmp/ml --recursive
      - echo "Create/Update of the SageMaker Pipeline and execution completed."
```

---
**Note:**

This template and documentation provide a clear starting point for configuring and running SageMaker pipelines. By leveraging the variables already defined in STM32_AWS_CDK/lib/ml/sagemaker-pipeline.ts, you only need to focus on customizing the dataset-specific details (DATASET_NAME and DOWNLOAD_COMMAND). This ensures consistency and reduces the risk of errors.