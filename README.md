# How to it work ?

## How to compress data ?
```
python ./CustomKnowledgeGraphEmbedding/compress_data/main.py \
-idr ../data/wn18rr \
-odr ./split_data/wn18rr -bz 1024
```

## Run project
Step 1: Clone project
```
git clone https://github.com/NguyenThaiHoc1/CustomKnowledgeGraphEmbedding.git
```

Step 2: Run project
```
python ./CustomKnowledgeGraphEmbedding/tensorflow_codes/run.py \
-ip "gs://hien7613storage2/datasets/KGE/wn18rr.tfrec" \
-bz 16 \
-sf "InterHT" \
--nentity 40943 \
--nrelation 11 \
--hidden_dim 1000 \
--gamma 24.0 \
--epochs 10 \
--steps_per_epoch 100 \
-de -tr
```
