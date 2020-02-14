import kfp.dsl as dsl
from kfp import components
from kubernetes import client as k8s_client

import os
import json
from random import randint

dkube_preprocess_op         = components.load_component_from_file("../components/preprocess/component.yaml")
dkube_split_op         = components.load_component_from_file("../components/split/component.yaml")
dkube_training_op           = components.load_component_from_file("../components/training/component.yaml")
dkube_evaluation_op         = components.load_component_from_file("../components/evaluation/component.yaml")
dkube_serving_op            = components.load_component_from_file("../components/serving/component.yaml")
dkube_viewer_op             = components.load_component_from_file('../components/viewer/component.yaml')

@dsl.pipeline(
    name='dkube-regression-pl',
    description='sample regression pipeline with dkube components'
)

def d3pipeline(
    #cellular preprocess
    cellular_preprocess_script="python cell-classification/preprocessing/merge.py",
    cellular_preprocess_datasets=json.dumps(["cellular"]),
    cellular_preprocess_input_mounts=json.dumps(["/opt/dkube/input"]),
    cellular_preprocess_outputs=json.dumps(["cellular-preprocessed"]),
    cellular_preprocess_output_mounts=json.dumps(["/opt/dkube/output"]),
    
    #cellular split
    cellular_split_script="python cell-classification/split/annot_split.py",
    cellular_split_datasets=json.dumps(["cellular-preprocessed"]),
    cellular_split_input_mounts=json.dumps(["/opt/dkube/input"]),
    cellular_split_outputs=json.dumps(["cellular-train", "cellular-test"]),
    cellular_split_output_mounts=json.dumps(["/opt/dkube/output/train", "/opt/dkube/output/test"]),
    
    #Training
    #In notebook DKUBE_USER_ACCESS_TOKEN is automatically picked up from env variable
    auth_token  = os.getenv("DKUBE_USER_ACCESS_TOKEN"),
    #By default tf v1.14 image is used here, v1.13 or v1.14 can be used. 
    #Or any other custom image name can be supplied.
    #For custom private images, please input username/password
    training_container=json.dumps({'image':'docker.io/ocdr/d3-datascience-tf-cpu:v1.14', 'username':'', 'password': ''}),
    #Name of the workspace in dkube. Update accordingly if different name is used while creating a workspace in dkube.
    training_program="faster-rcnn",
    #Script to run inside the training container    
    training_script="python cell-classification/model/train_frcnn.py -o simple -p /opt/dkube/input/annot.txt --hf --vf --rot --num_epochs 1",
    #Input datasets for training. Update accordingly if different name is used while creating dataset in dkube.    
    training_datasets=json.dumps(["train"]),
    training_input_dataset_mounts=json.dumps(["/opt/dkube/input/"]),
    training_outputs=json.dumps(["faster-rcnn"]),
    training_output_mounts=json.dumps(["/opt/dkube/output"]),
    #Request gpus as needed. Val 0 means no gpu, then training_container=docker.io/ocdr/dkube-datascience-tf-cpu:v1.12    
    training_gpus=0,
    #Any envs to be passed to the training program    
    training_envs=json.dumps([{"steps": 100}]),
    
    #Evaluation
    evaluation_script="python cell-classification/model/evaluate.py --path /opt/dkube/input/annot.txt",
    evaluation_datasets=json.dumps(["test"]),
    evaluation_input_dataset_mounts=json.dumps(["/opt/dkube/inputs/"]),
    evaluation_models=json.dumps(["faster-rcnn"]),
    evaluation_input_model_mounts=json.dumps(["/opt/dkube/model"]),
    
    #Serving
    #Device to be used for serving - dkube mnist example trained on gpu needs gpu for serving else set this param to 'cpu'
    serving_device='cpu',
    serving_container=json.dumps({'image':'docker.io/ocdr/new-preprocess:satish', 'username':'', 'password': ''})):
    
    cellular_preprocess  = dkube_preprocess_op(auth_token, training_container,
                                      program=training_program, run_script=cellular_preprocess_script,
                                      datasets=cellular_preprocess_datasets, outputs=cellular_preprocess_outputs,
                                      input_dataset_mounts=cellular_preprocess_input_mounts, output_mounts=cellular_preprocess_output_mounts)


    cellular_split  = dkube_preprocess_op(auth_token, training_container,
                                      program=training_program, run_script=cellular_split_script,
                                      datasets=cellular_split_datasets, outputs=cellular_split_outputs,
                                      input_dataset_mounts=cellular_split_input_mounts,
                                      output_mounts=cellular_split_output_mounts).after(cellular_preprocess)
                                      
                                    
    train       = dkube_training_op(auth_token, training_container,
                                    program=training_program, run_script=training_script,
                                    datasets=training_datasets, outputs=training_outputs,
                                    input_dataset_mounts=training_input_dataset_mounts,
                                    output_mounts=training_output_mounts,
                                    ngpus=training_gpus,
                                    envs=training_envs).after(cellular_split)
                                    
    evaluate    = dkube_training_op(auth_token, training_container,
                                    program=training_program, run_script=evaluation_script,
                                    datasets=evaluation_datasets,
                                    input_dataset_mounts=evaluation_input_dataset_mounts,
                                    models=evaluation_models,
                                    input_model_mounts=evaluation_input_model_mounts,
                                    ngpus=training_gpus,
                                    envs=training_envs).after(train)
    # serving     = dkube_serving_op(auth_token, train.outputs['artifact'], device=serving_device, serving_container=serving_container).after(evaluate)
    #inference   = dkube_viewer_op(auth_token, serving.outputs['servingurl'],
    #                              'digits', viewtype='inference').after(serving)