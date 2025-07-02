This repository is to solve the machine translation task using transformer. The Transformer is implemented using python and PyTorch.

The model is trained on MultiUN dataset with 8 million examples for 4 epochs, the achieved BLEU score on test set of MultiUN is 
BLEU = 75.06 100.0/88.9/62.5/57.1 (BP = 1.000 ratio = 1.000 hyp_len = 10 ref_len = 10).

The transformer architecture uses pre-normalization layers for smooth training. The tokenizer used in this model is MarianTokenizer, this model is trained to translate english sentences to french, but if someone want to train it for any other language, they can. They only need to change the tokenizer and training data.

The model is smaller in size compared to other translation models that are highly effective at translating sentences from various contexts, this model gives good translation on sentences from its training data or the data related to MultiUN data. Due to lack of hardware the model is only trained to handle sentences related to it's training data. But if you have the compute power then you can scale its parameters and improve its performance.
The whole model is trained on kaggle T4 gpus from scratch.


Observation and changes during training.
1. Firstly I had kept the d_model to 256 but even after training 2 epoch the model was not able to give good BLEU score. so then I changed it to 512 which brought good increase in score.

2. The max norm in gradient clipping was set to 1.0 for first epoch, but from second epoch to improve the accuracy and generalization I set the max norm to 5.0 which worked well and didn't produce NaN values.

3. The first training dataset had 10 million examples in which almost 3 million exampels had length between 5 to 10 tokens, so even after shuffling the dataset, some batches were having very large gradients and some were having very small, which distabilized the training and the model started producing NaN values. So I removed the sentences between length 5 to 10 tokens.

4. The model uses pre-layer normalization, but in original transformer paper "Attention is all you need" they have used post-layer normaliation. I had also used post-layer normalization but it was producing nan values in attention mechanism. So I swithed to pre-layer norm which worked well.

Train the model - 
If you have to train the model, you first have to give the path of files of your source language and target language, make sure that both languages should be stored as list of strings. If the source and target languages are different from english and french then you will have to change the tokenizer, which you can do in tokenization.py. After this you have to define the hyper parameters in train.py file if you want to change some and then run that file.

The model is uploaded on huggingface library.
You can download it from this link https://huggingface.co/Rohit2sali/transformer/blob/main/transformer.pth 
