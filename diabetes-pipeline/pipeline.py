# pipeline.py
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker import image_uris

bucket = 'main-sagemaker-zohebml'
prefix = 'mlops'
role = 'arn:aws:iam::590183717898:role/service-role/AmazonSageMaker-ExecutionRole-20240716T105741'

sagemaker_session = sagemaker.Session()
pipeline_session = PipelineSession()

input_source = f"s3://{bucket}/{prefix}/diabetes.csv"
train_path = f"s3://{bucket}/{prefix}/train"
test_path = f"s3://{bucket}/{prefix}/test"
val_path = f"s3://{bucket}/{prefix}/val"
model_output_uri = f"s3://{bucket}/{prefix}/output"
evaluation_output_uri = f"s3://{bucket}/{prefix}/output/evaluation"

########## PREPROCESSING STEP ##########
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name='diabetes-preprocess'
)

processing_step = ProcessingStep(
    name='PreprocessingStep',
    processor=sklearn_processor,
    code='preprocess.py',
    inputs=[ProcessingInput(source=input_source, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/output/train", destination=train_path),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/output/test", destination=test_path),
        ProcessingOutput(output_name="val_data", source="/opt/ml/processing/output/validation", destination=val_path),
    ]
)

########## TRAINING STEP ##########
estimator = SKLearn(
    entry_point='train.py',
    framework_version="1.7.2",
    instance_type='ml.m5.xlarge',
    role=role,
    output_path=model_output_uri,
    base_job_name='diabetes-train',
    hyperparameters={'n_estimators': 100, 'max_depth': 5}
)

train_input = TrainingInput(s3_data=train_path, content_type='text/csv')

train_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={"training": train_input}
)

########## EVALUATION STEP ##########
evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

image_uri = image_uris.retrieve(framework="sklearn", region="ap-south-1", version="1.7.2")

evaluation_processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=evaluation_processor,
    code="eval.py",
    inputs=[
        ProcessingInput(source=train_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
        ProcessingInput(source=test_path, destination="/opt/ml/processing/test")
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/output", destination=evaluation_output_uri)
    ],
    property_files=[evaluation_report]
)

########## REGISTER MODEL (IF ACCURACY > 0.6) ##########
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=f"{evaluation_output_uri}/evaluation.json",
        content_type="application/json"
    )
)

model = Model(
    image_uri=image_uri,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=pipeline_session
)

register_args = model.register(
    content_types=["application/x-model"],
    response_types=["application/json"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="diabetes",
    model_metrics=model_metrics
)

step_register = ModelStep(name="RegisterDiabetesModel", step_args=register_args)

cond_gte = ConditionGreaterThan(
    left=evaluation_step.properties.PropertyFiles["EvaluationReport"].accuracy,
    right=0.60
)

condition_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[cond_gte],
    if_steps=[step_register],
    else_steps=[]
)

train_step.add_depends_on([processing_step])
evaluation_step.add_depends_on([train_step])
condition_step.add_depends_on([evaluation_step])

########## PIPELINE ##########
pipeline = sagemaker.workflow.pipeline.Pipeline(
    name="Diabetes-ML-Pipeline",
    steps=[processing_step, train_step, evaluation_step, condition_step],
    sagemaker_session=pipeline_session
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print("🚀 Pipeline started:", execution.arn)