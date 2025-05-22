# Insilico Medicine AWS SDK

Insilico-aws is a python package for interacting with Insilico Medicine's products deployed on AWS.
Please check the `examples` directory for more details.

This package is optional, if you a proficient user of the Sagemaker API you might prefer `sagemaker.Session` direct usage.
 

## Installation

```bash
pip install insilico-aws
```

## Quick Start

Before using the package make sure you have configured AWS credentials,
the default region can be overwritten by the client parameter `region_name`.

Use the Product Arn (can be found in your subscription details) to create the client instance:

```python
from insilico_aws import AlgorithmClient
client = AlgorithmClient(algorithm='<name>', arn='<arn>')
```

Check the example [Jupyter notebooks](insilico_aws/examples) of chosen algorithm/model 
for further details (fine-tune jobs, inference endpoints, e.t.c.)

Use the following command to create test notebooks and datafiles in your local workspace:

```bash
python -c "import insilico_aws; insilico_aws.load_examples(overwrite=False)"
```

then open the `examples` directory.

## Support

If you have any questions or need assistance, feel free to reach out at: chemistry42@insillicomedicine.com
