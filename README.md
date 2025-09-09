This repository contains the code for a 100 million parameter Transformer model for English-to-French machine translation, trained from scratch using PyTorch.

The model was trained on a large-scale corpus of 3 billion tokens and is publicly available on the Hugging Face Hub.

🚀 Highlights
Large-Scale Training: Trained on a 3 billion token corpus aggregated from ParaCrawl and MultiUN datasets.

Strong Performance: Achieves a BLEU score of 20.45 on unseen data from the OPUS-Wikipedia test set.

Open-Sourced: The pre-trained model is available on the Hugging Face Hub for anyone to use or fine-tune.  

📊 Model Performance  
The model was evaluated on two test sets it was not trained on to verify its generalization capabilities.  

|Test Set (Unseen Data)|	BLEU Score |	Full Metric	|  
|---|---|---|  
|📝 OPUS-Wikipedia|	     | 20.45|	     | 90.6/41.9/13.3/3.4|	 
|🌐 OPUS-ParaCrawl|      |18.14|	     |95.8/52.2/9.1/2.4	|  

⚙️ Fine-Tuning  
This model serves as a strong baseline for English-to-French translation and can be fine-tuned on a more specific domain (e.g., legal or medical texts).  

To fine-tune the model:  

Prepare your dataset: Format your parallel corpus into source and target text files.  

Download the pre-trained model from huggingface using this link "https://huggingface.co/Rohit2sali/en-fr-translation-transformer-100M".   

Update the data loader: Point the script to your custom dataset.  

The dataset I used for training can be downloaded from here,  
"https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/en-fr.txt.zip"  
"https://object.pouta.csc.fi/OPUS-ParaCrawl/v1/moses/en-fr.txt.zip"  
"https://object.pouta.csc.fi/OPUS-ParaCrawl/v4/moses/en-fr.txt.zip"  
"https://object.pouta.csc.fi/OPUS-ParaCrawl/v5/moses/en-fr.txt.zip"  
