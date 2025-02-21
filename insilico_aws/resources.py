
from typing import Optional, Any


from pydantic import BaseModel


class Algorithm(BaseModel):
    name: str
    region_name: str
    arn: Optional[str] = None
    account_id: Optional[str] = None
    training_instance_type: list[str]
    training_data_required: Optional[list[str]] = None
    inference_instance_type: list[str]
    training_max_run_hours: int
    training_volume_size_gb: int
    inference_parameters: list[dict[str, Any]]

