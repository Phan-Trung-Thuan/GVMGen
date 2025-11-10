## GVMGen: A General Video-to-Music Generation Model With Hierarchical Attentions

[[Paper](https://arxiv.org/abs/2501.09972)]
[[Project Website](https://chouliuzuo.github.io/GVMGen/)]
![Model Architecture](./static/images/model.png)

### 1. Example Demos

```./examples_of_results```: 32 test videos

### 2. Environment Preparation

Please first clone the repo and install the required environment, which can be done by running the following commands:

```
conda env create -n gvmgen python=3.9.0

conda activate gvmgen

cd GVMGen

pip install -r requirements.txt
```

### 3. Data Preprocessing

Please refer to ```./data_preprocess```

### 4. Training

* Download MusicGen model from [MusicGen (small)](https://huggingface.co/facebook/musicgen-small) or [MusicGen (medium)](https://huggingface.co/facebook/musicgen-medium) and put them into ```./checkpoints``` folder.
* Modify the config files
  There are some variables you **must** modify before your training. Other changes are optional and you can refer to each ```default.yaml```

  ```
  config/dset/train.yaml datasource.evaluate path/to/eval_folder
  config/dset/train.yaml datasource.generate path/to/eval_folder
  config/dset/train.yaml datasource.train path/to/train_folder
  config/dset/train.yaml datasource.valid path/to/eval_folder

  config/solver/gvmgen/gvmgen.yaml compression_model_checkpoint path/to/musicgen_compression_model
  config/teams/default.yaml default.dora_dir path/to/GVMGen
  config/teams/default.yaml default.reference_dir path/to/GVMGen
  config/teams/default.yaml darwin.dora_dir path/to/GVMGen
  config/teams/default.yaml darwin.reference_dir path/to/GVMGen
  ```
* run training

  ```
  bash run.sh
  ```

### 5. Inference

* transform model weights (Run this step **only when loading your own trained model**. If you want to test our published model, please skip it.)

```
python load_model.py --checkpoint_path path/to/your_checkpoint --output_path path/to/output
```

* inference

```
python test.py --model_path ./checkpoints --video_path test.mp4 --syn_path output --fps 1 --duration 30
```

### 6. Evaluation Model

Please refer to ```./evaluation_model``` folder.

### 7. Dataset and Model weights

We will release our dataset soon.
The pretrained model's parameter weights have been published and can be accessed at [here](https://drive.google.com/drive/folders/1OKVQlz7TPKEGTPEK-BiH1tJ3FYT11GON?usp=sharing).

### 8. Acknowledgements

you may refer to related work that serves as foundations for our framework and code repository, CLIP, MusicGen. Thanks for their wonderful works.

### 9. Citation
```
@inproceedings{zuo2025gvmgen,
        title={GVMGen: A General Video-to-Music Generation Model With Hierarchical Attentions},
        author={Zuo, Heda and You, Weitao and Wu, Junxian and Ren, Shihong and Chen, Pei and Zhou, Mingxu and Lu, Yujia and Sun, Lingyun},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2025}
    }
```
