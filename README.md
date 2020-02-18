# Graph Convolution on Structured Documents
This repo contains code to convert Structured Documents to Graphs and implement a Graph Convolution Neural Network (incomplete) for Node Classification, each node being an entity in the document.

Check out the article for an intuitive explanation on Towards Data Science: [Using Graph Convolutional Neural Networks on Structured Documents for Information Extraction](https://towardsdatascience.com/using-graph-convolutional-neural-networks-on-structured-documents-for-information-extraction-c1088dcd2b8f) 

## Code
The `grapher.py` file contains the code to convert a structured document to a graph. <br>
An object map made using a Commercial OCR Tool is needed as the input which provides the bounding-box coordinates of each entity in the image along with it's recognized text. The script can then be used to generate an `object_tree.png` file and a 
`connections.csv` file. The script joins each object to it's nearest object to the right and underneath thus generating a graph. <br> Here is what the generated graph looks like:
![Graph](/outputs/object_tree.jpg)
## Graph Convolution Model
The implementation is still in progress and is being built using Tensorflow 1.8. The implementation details can be found in [1].
## References
1. Riba, Dutta et al - Table Detection in Invoice Documents by Graph Neural Networks - [Link](https://priba.github.io/assets/publi/conf/2019_ICDAR_PRiba.pdf)
2. Adam W. Harley, Alex Ufkes, and Konstantinos G. Derpanis - Department of Computer Science, Ryerson University, Toronto, Ontario - Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval - [Link](https://arxiv.org/abs/1502.07058)
3. Victor Garcia, Joan Bruna - Few-Shot Learning with Graph Neural Networks - [Link](https://arxiv.org/abs/1711.04043) 
