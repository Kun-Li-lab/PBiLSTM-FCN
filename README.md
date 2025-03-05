# PBiLSTM-FCN
Execution steps:
1. Download the dataset in ". txt. gz" format from the Uniprot official website
2. Run data.Py, which can preprocess the dataset and generate three dataset files: bp.Pkl, mf.Pkl, and cc.Pkl
3. Create a new folder directly in the same level directory and name it "data". Place go.obo and the three dataset files generated in step 2 into the "data" folder
4. Run PBiLSM-FCN.py

The 'go. obo' file is used to store all GO term relationships. To download the latest 'go. obo' file, please follow these steps:
1. Enter the official website http://geneontology.org/docs/download-ontology/
2. See the item with Name "go. obo" and a link displayed next to it http://purl.obolibrary.org/obo/go.obo
3. Right click http://purl.obolibrary.org/obo/go.obo Select 'Save Link As' to download the latest version of go.obo file


Dependencies
The code was developed and tested using python 3.7.

click==8.0.4

Keras==2.3.1

keras_radam==0.15.0

keras_rectified_adam==0.20.0

keras_self_attention==0.51.0

matplotlib==3.5.2

numpy==1.21.5

pandas==1.3.5

scikit_learn==1.1.2

tensorflow==2.1.0

