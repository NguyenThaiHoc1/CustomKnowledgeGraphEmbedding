# How to it work ?

Clone project
```
git clone https://github.com/NguyenThaiHoc1/CustomKnowledgeGraphEmbedding.git
```

Run project
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
