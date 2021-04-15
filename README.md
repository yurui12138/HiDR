## HiDR

#### Chinese Sentence Semantic Matching with Hierarchical CNN Based on Dimension-augmented Representation
Rui Yu, Wenpeng Lu, Yifeng Li, Jiguo Yu, Guoqiang Zhang, Xu Zhang
The paper has been accepted by IJCNN.

#### Prerequisites
python 3.6  
numpy==1.16.4  
pandas==0.22.0  
tensorboard==1.12.0  
tensorflow-gpu==1.12.0  
keras==2.2.4  
gensim==3.0.0  

#### Example to run the codes
Run HiDR.py  
`python3 HiDR.py`  

#### Dataset
We used two datasets: BQ & LCQMC.  
1. "The BQ Corpus: A Large-scale Domain-specific Chinese Corpus For Sentence Semantic Equivalence Identification", https://www.aclweb.org/anthology/D18-1536/.  
2. "LCQMC: A Large-scale Chinese Question Matching Corpus", https://www.aclweb.org/anthology/C18-1166/.

### Note
Due to the differences between the two data sets, some parameters adopted by HiDR are different. Therefore, we provide two versions of the code for the two data sets.
