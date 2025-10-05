import datetime
from typing import List, Annotated, Optional, Dict, Literal
import kubernetes.client.models as k8s
from kubernetes.client import V1EnvFromSource
from pydantic import BaseModel, BeforeValidator, ConfigDict


class TaskConfigAirflow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    namespace: str
    image_pull_secrets: Annotated[
        List[k8s.V1LocalObjectReference],
        BeforeValidator(lambda xs: [k8s.V1LocalObjectReference(x) for x in xs]),
    ]
    image: str

    in_cluster: Optional[bool] = True
    random_name_suffix: Optional[bool] = True
    do_xcom_push: Optional[bool] = True


class RuntimeConfig(BaseModel):
    use_gpu: Optional[bool] = None
    tune_enabled: Optional[bool] = None
    train_enabled: Optional[bool] = None


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_config_airflow: TaskConfigAirflow
    task_env_vars: Dict[str, str]
    task_env_from_secrets: Annotated[
        List[V1EnvFromSource],
        BeforeValidator(lambda xs: [k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(x)) for x in xs]),
    ]


class ModelDevelopmentConfig(Config):
    runtime_config: RuntimeConfig


class ActiveFiresProductionConfig(Config): ...


class CarbonMonoxideProductionConfig(Config): ...

class PrecipitationProductionConfig(Config): ...