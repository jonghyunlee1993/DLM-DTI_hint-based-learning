# DLM-DTI_hint-based-learning

The drug discovery process is demanding and time-consuming, and machine learning-based research is increasingly being proposed to enhance efficiency. A significant challenge in this field is predicting whether a drug molecule structure will interact with a target protein. A recent study attempted to address this challenge by utilizing an encoder that leverages prior knowledge of molecular and protein structures, resulting in notable improvements in the prediction performance of the drug-target interactions task. Nonetheless, the target encoders employed in previous studies exhibit computational complexity that increases quadratically with the input length, thereby limiting their practical utility. To overcome this challenge, we adopt a hint-based learning strategy, a knowledge distillation technique, to develop a compact and efficient target encoder. This approach yielded considerable performance and enhancements of learning efficiency on three benchmark datasets: BIOSNAP, DAVIS, and BindingDB. Additionally, hint-based learning enables the target sequence length to be independent of the prediction performance. The proposed model was also tested on a case study comprising 13 widely recognized drug and target pairs and correctly classified with higher probability. Our proposed model, which utilizes dual pre-trained encoders with hint-based learning, demonstrated significantly improved performance in predicting drug-target interaction tasks.

![conceptual_diagram](https://user-images.githubusercontent.com/37280722/230818993-072d3c21-b580-4d16-9651-aa745f30153b.jpg)

## Related works
- Drug (molecule) encoder
	- ChemBERTa v1:	https://arxiv.org/abs/2010.09885; https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
	- ChemBERTa v2: https://arxiv.org/abs/2209.01712; https://huggingface.co/DeepChem
- Target (protein) encoder
	- https://arxiv.org/abs/2007.06225; https://huggingface.co/Rostlab/prot_bert
- Fine-tuning of BERT Model to Accurately Predict Drug–Target Interactions
	- https://www.mdpi.com/1999-4923/14/8/1710
- Fit-Net (hint-based learning)
	- https://arxiv.org/abs/1412.6550

## Datasets
- Binary classification: DAVIS, Binding DB, and BIOSNAP
- Downloaded from [MolTrans](https://github.com/kexinhuang12345/MolTrans/tree/master/dataset)

## Performances
| Dataset   | Metric      | MolTrans | fine-tuning BERT | DLM-DTI   |
| --------- | ----------- | -------- | ---------------- | --------- |
| DAVIS     | AUROC       | 0.907    | 0.942            | **0.982** |
|           | AUPRC       | 0.404    | 0.517            | **0.980** |
|           | Sensitivity | 0.800    | 0.903            | **0.965** |
|           | Specificity | 0.876    | 0.889            | **0.899** |
| BindingDB | AUROC       | 0.914    | 0.926            | **0.980** |
|           | AUPRC       | 0.622    | 0.636            | **0.979** |
|           | Sensitivity | 0.797    | 0.814            | **0.951** |
|           | Specificity | 0.896    | **0.928**        | 0.895     |
| BIOSNAP   | AUROC       | 0.895    | 0.914            | **0.979** |
|           | AUPRC       | 0.901    | 0.900            | **0.978** |
|           | Sensitivity | 0.775    | 0.862            | **0.941** |
|           | Specificity | 0.851    | 0.863            | **0.909** |

