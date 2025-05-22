import csv
import json
import logging
from contextlib import suppress
from functools import partial
from typing import Any, Optional

import boto3
import sagemaker
from botocore.exceptions import NoRegionError, ClientError
from sagemaker.s3 import S3Uploader

from insilico_aws.resources import Algorithm
from insilico_aws.utils import validate_parameters, __version__, load_resources_definition

logger = logging.getLogger(__name__)


class AlgorithmClient:
    def __init__(
        self,
        algorithm: str,
        region_name: Optional[str] = None,
        arn: Optional[str] = None,
        default_tags: Optional[dict[str, Any]] = None
    ):
        """
        Create a client for interacting with Sagemaker Algorithm.

        Args:
            algorithm: Algorithm name.
            region_name: The region where the algorithm is deployed.
            arn: Algorithm arn (optional), overwrites algorithm name and region.
            default_tags: Tags to add to training jobs and endpoints.
        """
        region_name = region_name or boto3.session.Session().region_name
        if not region_name:
            raise NoRegionError()
        boto3.setup_default_session(region_name=region_name)
        self.sagemaker_client = boto3.client('sagemaker')
        self.sagemaker_session = sagemaker.Session(sagemaker_client=self.sagemaker_client)
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        self.job_name: Optional[str] = None
        self.endpoint_name: Optional[str] = None
        self.model_name: Optional[str] = None
        self.default_tags = default_tags

        resources = load_resources_definition()

        try:
            self.algorithm = Algorithm(
                name=algorithm,
                region_name=region_name,
                arn=arn,
                **resources['algorithm'][algorithm]
            )
        except KeyError:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}, "
                f"expected: {', '.join(resources['algorithm'].keys())}"
            )

    def upload_train_data(
        self, train_data: str, test_data: str, s3_uri: str
    ) -> dict[str, str]:
        """
        Upload training job input data to s3.
        Args:
            train_data: Source path to train.csv file.
            test_data: Source path to test.csv file.
            s3_uri: Desired s3 path, s3://<bucket>/<prefix>.

        Returns:
            A dict, containing uploaded paths.
        """
        if self.algorithm.training_data_required:
            reqs = set(self.algorithm.training_data_required)
            for input_file in (train_data, test_data):
                with open(input_file, 'r') as f:
                    reader = csv.DictReader(f)
                    if missing_columns := reqs - set(reader.fieldnames):  # type: ignore
                        raise ValueError(
                            f"Train data has missing columns in file {input_file}: "
                            f"{', '.join(missing_columns)}"
                        )

        upload = partial(
            S3Uploader.upload,
            desired_s3_uri=s3_uri,
            sagemaker_session=self.sagemaker_session
        )
        return {
            'train_data': upload(local_path=train_data),
            'test_data': upload(local_path=test_data)
        }

    def create_training_job(
        self,
        input_path: str,
        output_path: str,
        instance_type: str,
        wait: bool = False,
        role: Optional[str] = None,
        max_run_hours: Optional[int] = None,
        tags: Optional[dict[str, Any]] = None,
        training_parameters: Optional[dict[str, Any]] = None,
        training_volume_size_gb: Optional[int] = None,
    ) -> Optional[str]:
        """
        Run a new training job for current algorithm.
        Args:
            input_path: Source s3 path to train data.
            output_path: Destination s3 path where trained model will be saved.
            wait: Whether the call should wait until the job completes.
            role: Job execution role name, will try to use current credentials if not set.
            instance_type: EC2 instance type to use for training.
            max_run_hours: Timeout in seconds for training.
            tags: Tags for labeling the job.
            training_parameters: Parameters to pass into the job.
            training_volume_size_gb: Size of the volume to use for storing input data.

        Returns:
            Created training job name.
        """
        if not role:
            role = sagemaker.get_execution_role(sagemaker_session=self.sagemaker_session)

        algorithm_estimator = sagemaker.algorithm.AlgorithmEstimator(
            algorithm_arn=self.algorithm.arn,
            role=role,
            instance_count=1,
            instance_type=instance_type,
            max_run=(max_run_hours or self.algorithm.training_max_run_hours) * 60 * 60,
            output_path=output_path,
            sagemaker_session=self.sagemaker_session,
            tags=tags or self.default_tags,
            base_job_name=self.algorithm.name,
            volume_size=training_volume_size_gb or self.algorithm.training_volume_size_gb,
            hyperparameters=training_parameters
        )
        algorithm_estimator.fit(inputs={'train': input_path}, wait=wait)
        self.job_name = algorithm_estimator.latest_training_job.name
        return self.job_name

    def find_latest_training_job(self, top_n=1):
        summaries = self.sagemaker_client.list_training_jobs(
            NameContains=self.algorithm.name, SortOrder='Descending'
        )['TrainingJobSummaries']
        summaries = summaries[:top_n]
        return [(s['TrainingJobName'], s['TrainingJobStatus']) for s in summaries]

    def get_training_job_status(self, job_name: str) -> dict[str, str]:
        """
        Calls the DescribeTrainingJob API for the given job name and returns the response.
        See sagemaker session methods for more options

        Args:
            job_name: The name of the training job to describe.

        Returns:
            A dict containing the job status and secondary status.
        """
        job_description = self.sagemaker_session.describe_training_job(job_name)
        return {
            'status': job_description['TrainingJobStatus'],
            'secondary_status': job_description['SecondaryStatus']
        }

    def stop_training_job(self, job_name: str):
        """
        Stops the training job.

        Args:
            job_name (str): The name of the training job to stop.
        """
        self.sagemaker_session.stop_training_job(job_name)

    def create_endpoint(
        self,
        endpoint_name: str,
        instance_type: str,
        n_instances: int = 1,
        exists_ok: bool = False,
        role: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
        model_data_path: Optional[str] = None,
        training_job_name: Optional[str] = None,
        training_job_output_path: Optional[str] = None,
        inference_parameters: Optional[dict[str, Any]] = None
    ):
        """
        Deploy new inference endpoint.

        Args:
            endpoint_name: The endpoint name.
            instance_type: EC2 instance type where the endpoint should be deployed.
            n_instances: Number of instance (1 by default).
            exists_ok: Where to reuse existing endpoint.
            role: The IAM role which is assigned to the service.
            tags: Tags for labeling the endpoint.
            model_data_path: Absolute s3 path to training results (model.tar.gz file)
                if not set, the following parameters will be used to construct the path.
            training_job_name: Use the output of this job for inference.
            training_job_output_path: The path which was defined as the job's output_path.
            inference_parameters: Inference parameters.
        """
        if not model_data_path:
            if not (training_job_name and training_job_output_path):
                raise ValueError(
                    'Model data path or training job name & output path must be provided'
                )
            model_data_path = (
                f"{training_job_output_path.strip('/')}/"
                f"{training_job_name}/output/model.tar.gz"
            )
        if not role:
            role = sagemaker.get_execution_role(sagemaker_session=self.sagemaker_session)
        try:
            endpoint_description = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            if exists_ok:
                if s := endpoint_description['EndpointStatus'] != 'InService':
                    raise RuntimeError(f"Endpoint {endpoint_name} is not ready: {s}")
                self.endpoint_name = endpoint_name
                return
            raise RuntimeError(f"Endpoint {endpoint_name} already exists")
        except ClientError:
            pass

        validate_parameters(self.algorithm.inference_parameters, inference_parameters)
        if isinstance(inference_parameters, dict):
            inference_parameters = {k: str(v) for k, v in inference_parameters}

        model_package = sagemaker.ModelPackage(
            role=role,
            algorithm_arn=self.algorithm.arn,
            model_data=model_data_path,
            name=endpoint_name,
            env=inference_parameters
        )
        self.model_name = endpoint_name
        model_package.deploy(
            initial_instance_count=n_instances,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            tags=tags or self.default_tags  # type: ignore
        )
        self.endpoint_name = endpoint_name
        logger.info('Endpoint with name %s has been created', endpoint_name)

    def invoke_endpoint(
        self,
        request_data: str,
        endpoint_name: Optional[str] = None,
    ):
        """
        Invoke the inference endpoint.
        Args:
            request_data: Csv file path.
            endpoint_name: The endpoint name.

        Returns:

        """
        with open(request_data, 'rb') as f:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name or self.endpoint_name,
                Body=f,
                CustomAttributes=f'client-version={__version__}'
            )
        return json.load(response['Body'])

    def delete_endpoint(self, endpoint_name: Optional[str] = None, quiet: bool = True):
        """
        Delete the inference endpoint.
        Args:
            endpoint_name: The endpoint name.
            quiet: Ignore deletion errors.

        Returns:

        """
        exceptions = [ClientError] if quiet else []
        with suppress(*exceptions):
            self.sagemaker_client.delete_endpoint(
                EndpointName=endpoint_name or self.endpoint_name
            )
        with suppress(*exceptions):
            self.sagemaker_client.delete_endpoint_config(
                EndpointConfigName=endpoint_name or self.endpoint_name
            )
        self.endpoint_name = None

    def delete_model(self, model_name: Optional[str] = None, quiet: bool = True):
        """
        Delete model and model package resources
        Args:
            model_name: The model name
            quiet: Ignore deletion errors.

        Returns:

        """
        exceptions = [ClientError] if quiet else []
        with suppress(*exceptions):
            self.sagemaker_client.delete_model(
                ModelName=model_name or self.model_name
            )
        with suppress(*exceptions):
            self.sagemaker_client.delete_model_package(
                ModelPackageName=model_name or self.model_name
            )

    def close(self):
        """
        Closes underlying endpoint connections.
        """
        self.sagemaker_runtime.close()
        self.sagemaker_client.close()

