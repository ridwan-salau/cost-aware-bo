import random
from sagemaker.pytorch import PyTorch
import sagemaker
import boto3
from sagemaker.remote_function import remote

# sagemaker_session = sagemaker.Session()
sagemaker_session = sagemaker.Session()

region = sagemaker_session.boto_region_name

bucket = "mbz-hpc-aws-master"

role = sagemaker.get_execution_role()
print(role)

RANDOM = random.randint(1000, 10000)
acqf = "EEIPU"
exp_name="t5-pipe-multi-sgmkr"
trial = 1
root = f"{bucket}/AROARU6TOWKRU3FNVE2PB:Ridwan.Salahuddeen@mbzuai.ac.ae"
data_dir=f"{root}/inputs" 

cache_root=f"{root}/.cachestore/{acqf}/{RANDOM}_trial_{trial}"

settings = dict(
    # entry_point="optimize_multi.py",
    # source_dir="./",
    sagemaker_session=sagemaker_session,
    role=role,
    # py_version="py38",
    # framework_version="1.11.0",
    instance_count=1,
    instance_type="ml.c5.2xlarge",
    # output_path=f"s3://{root}",
    # hyperparameters={"exp-name": exp_name, "trial": trial, "data-dir": data_dir, "acqf": acqf, "cache-root": cache_root},
)

@remote(**settings)
def hello():
    return ("Hello, world")

if __name__=="__main__":
    print(hello())
# estimator = PyTorch(
#     entry_point="optimize_multi.py",
#     source_dir="./",
#     role=role,
#     py_version="py38",
#     framework_version="1.11.0",
#     instance_count=2,
#     instance_type="ml.c5.2xlarge",
#     output_path=f"s3://{root}",
#     hyperparameters={"exp-name": exp_name, "trial": trial, "data-dir": data_dir, "acqf": acqf, "cache-root": cache_root},
# )

# estimator.fit()

'''
python optimize_multi.py \
        --exp-name $exp_name --trial $trial --cache-root \
        $cache_root --acqf $acqf --data-dir $data_dir
'''