# PKR-HOI: Priori Knowledge Reasoning for Human-Object Interaction Detection

<div align="center">
  <img src="Figure/overview.png" width="900px" />
    Figure 1
</div>



Figure 1. PKR-HOI is a priori knowledge based framework that models scene context, human and object features, action recognition, and interaction relationship in a unified way for accurate HOI prediction. PKR-HOI aggregates the different-level features and attention mechanism in the transformer, and as a result, achieves high HOI detection performance with HOI decoder.

<div align="center">
  <img src="Figure/f2.png" width="900px" />
    Figure 2
</div>



Figure 2. Leveraging the action recognition sub-network. The top row shows the HOI prediction without the action recognition network,and the bottom row shows the results with the action recognition sub-network.

<div align="center">
  <img src="Figure/f1.png" width="900px" />
	Figure 3
</div>



Figure 3. Visualization of the attention maps for HOI decoder. It can be seen from the figure that PKR-HOI pays different attention to the contextual information in the HOI prediction process. PKR-HOI pays more attention to the entities with interactive relationships. Moreover, our method attends to recognize different actions(e.g. *hold* and *stand*) in the same image by paying attention to different areas.

## Preparation

### Dependencies
Our implementation uses external libraries such as NumPy and PyTorch. You can resolve the dependencies with the following command.
```
pip install numpy
pip install -r requirements.txt
```
Note that this command may dump errors during installing pycocotools, but the errors can be ignored.

### Dataset

#### HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
PKR-HOI
 |─ data
 │   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |   └─ corre_hico.npy
 :       :
```

#### V-COCO
First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle` from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). Place the files and make directories as follows.
```
PKR-HOI
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 :       :
```
For our implementation, the annotation file have to be converted to the HOIA format. The conversion can be conducted as follows.
```
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```
Note that only Python2 can be used for this conversion because `vsrl_utils.py` in the v-coco repository shows a error with Python3.

V-COCO annotations with the HOIA format, `corre_vcoco.npy`, `test_vcoco.json`, and `trainval_vcoco.json` will be generated to `annotations` directory.

### Pre-trained parameters
Our PKR-HOI have to be pre-trained with the COCO object detection dataset. For the HICO-DET training, this pre-training can be omitted by using the parameters of DETR. The parameters can be downloaded from [here](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth) for the ResNet50 backbone, and [here](https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth) for the ResNet101 backbone. For the V-COCO training, this pre-training has to be carried out because some images of the V-COCO evaluation set are contained in the training set of DETR. You have to pre-train PKR-HOI without those overlapping images by yourself for the V-COCO evaluation.

For HICO-DET, move the downloaded parameters to the `params` directory and convert the parameters with the following command.
```
python convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-hico.pth
```

For V-COCO, convert the pre-trained parameters with the following command.
```
python convert_parameters.py \
        --load_path logs/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-vcoco.pth \
        --dataset vcoco
```

## Training
After the preparation, you can start the training with the following command.

For the HICO-DET training.
```
python main.py \
        --pretrained params/detr-r50-pre-hico.pth \
        --output_dir logs \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --resume logs/checkpoint.pth \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --loss_hoi_weight 0.7 \
    	--loss_detr_weight 0.3 \
    	--loss_verbs_weight 0.7 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1 \
        --obj_loss_coef 1 \
    	--loss_ce_detr 1 \
    	--loss_bbox_detr 2.5 \
    	--loss_giou_detr 1
```

For the V-COCO training.
```
python main.py \
        --pretrained params/detr-r50-pre-vcoco.pth \
        --output_dir logs \
        --hoi \
        --dataset_file vcoco \
        --hoi_path data/v-coco \
        --resume logs/checkpoint.pth \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone resnet50 \
        --loss_hoi_weight 0.7 \
    	--loss_detr_weight 0.3 \
    	--loss_verbs_weight 0.7 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1 \
        --obj_loss_coef 1 \
    	--loss_ce_detr 1 \
    	--loss_bbox_detr 2.5 \
    	--loss_giou_detr 1
```
Note that the number of object classes is 81 because one class is added for missing object.

If you have multiple GPUs on your machine, you can utilize them to speed up the training. The number of GPUs is specified with the `--nproc_per_node` option. You can also specify the GPU number use `CUDA_VISIBLE_DEVICES` option. The following command starts the training with 8 GPUs for the HICO-DET training.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained params/detr-r50-pre-hico.pth \
        --output_dir logs \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --resume logs/checkpoint.pth \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --loss_hoi_weight 0.7 \
    	--loss_detr_weight 0.3 \
    	--loss_verbs_weight 0.7 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1 \
        --obj_loss_coef 1 \
    	--loss_ce_detr 1 \
    	--loss_bbox_detr 2.5 \
    	--loss_giou_detr 1
```

In addition, if you have only 4 GPUs available，then you can add a parameter `--simulate_double_gpus 1` which is used to simulate double GPUs and you can set its value to 0 to cancel this function. By the way, the parameter `--batch_size` should be doubled  at the same time. 

Examples of matched parameters:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --simulate_double_gpus 0 \
        --batch_size 2 \
        ...(other parameters)
```

or

```
CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env \
        main.py \
        --simulate_double_gpus 1 \
        --batch_size 4 \
        ...(other parameters)
```



## Evaluation
The evaluation is conducted at the end of each epoch during the training. The results are written in `logs/log.txt`.

You can also conduct the evaluation with trained parameters as follows.
```
python main.py \
        --pretrained pkr_hoi_resnet50_hico.pth \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --eval

```

For the official evaluation of V-COCO, a pickle file of detection results have to be generated. You can generate the file as follows.
```
python generate_vcoco_official.py \
        --param_path logs/checkpoint.pth \
        --save_path vcoco.torch \
        --hoi_path data/v-coco

```

## Results

V-COCO.

|| Scenario 1 | Scenario 2 |
| :--- | :---: | :---: |
|PKR-HOI (ResNet50)| 63.3 | 65.5 |

HICO-DET.
|| Full | Rare | Non-rare |
| :--- | :---: | :---: | :---: |
|PKR-HOI (ResNet50)| 30.90 | 27.01 | 32.42 |

