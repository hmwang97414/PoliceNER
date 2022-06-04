# PoliceNER
UMF-PNER: A Unified Multi-feature Fusion Model for
Police Named Entity Recognition

# Requirement:
Python	3.6.13
Pytorch	1.10.2
# data format
CoNLL format (prefer BIOES tag scheme), with each character its label for one line. Sentences are splited with a null line.
```
美   B-LOC  
国   E-LOC  
的   O  
华   B-PER  
莱   I-PER  
士   E-PER  

我   O  
跟   O  
他   O  
谈   O  
笑   O  
风   O  
生   O   
```
pre_recognized labels is in data/Police2000/train_tag.txt
phonetic feature is in data/Police2000/train_pinyin.txt

# Pretrained character Embeddings
Character embeddings:https://pan.baidu.com/s/1hJKTAz6PwS7wmz9wQgmYeg?_at_=1654347393410
pretrained_embedding.npz :https://pan.baidu.com/s/1yrOmXxfiNaUk-srOccfOdQ?pwd=up1e 
提取码：up1e
