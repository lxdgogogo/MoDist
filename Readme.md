# Mixture of Graph Distillation Learning



## Dependencies

- Pytorch 2.1.2
- DGL 2.4.0
- sklearn
- tqdm
- Numpy


## Model:

![](https://raw.githubusercontent.com/lxdgogogo/MoDist/master/Figure/model.png)

***

## Dataset

The datasets are in the "datasets" folder. First, unzip these datasets.

***

## How to run

Train Teacher Model:

```
cd Global

python pretrain_teacher.py --dataset amazon_ratings --epoch 100 --num_layers_teacher 2 \
--dropout 0.6 --hidden_size 128 --device cuda
```


Train Student Model:

```

python distill_more.py --dataset amazon_ratings --num_layers_teacher 2 --num_layers_student 2 \
--num_heads 8 --lambda1 0.3 --lambda2 0.4 --tau 1.0 --hidden_size 128 --dropout 0.6


```
