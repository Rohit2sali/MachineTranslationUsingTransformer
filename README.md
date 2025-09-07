This is the architecture and training code for a machine translation model using transformer. 

This model is 100 million parameters of model trained on 3 billion tokens from the dataset of MultiUN and Paracrawl from OPUS.

The dataset links are given below : 
https://object.pouta.csc.fi/OPUS-MultiUN/v1/moses/en-fr.txt.zip
https://object.pouta.csc.fi/OPUS-ParaCrawl/v1/moses/en-fr.txt.zip
https://object.pouta.csc.fi/OPUS-ParaCrawl/v4/moses/en-fr.txt.zip
https://object.pouta.csc.fi/OPUS-ParaCrawl/v5/moses/en-fr.txt.zip

The model was evaluated on 5k examples of wikipedia data from OPUS,
where it achieved the score BLEU = 20.45 90.6/41.9/13.3/3.4 (BP = 1.000 ratio = 1.000 hyp_len = 32 ref_len = 32)
the model was not trained on this data, so to check it's performance on unseen data this experiment was done.

The model was also evaluated on the unseen data of paracrawl, 
where it achieved the score BLEU = 20.45 90.6/41.9/13.3/3.4 (BP = 1.000 ratio = 1.000 hyp_len = 32 ref_len = 32)

The model is saved at hugging face at this link, 

If you want to fine tune this model for different dataset, then you can download the model from here, and then fine tune using the same training code. Just need to load the pre trained model and then train it on the data you want. 

Things I learned while training this model.
i) The transformer model is very sensitive, at first i was not adding special tokens like <sos> and <eos> for the encoders input,
   so due to that the model was not permorming well, because it was not understanding where the sentence starts or ends.

ii) As we know that deep learning is about learning the patterns in data, that's why data shuffling is very important, if the data is 
    not shuffled very well then the model might do good at first in training, but later it will crash and for sure while inference and
    testing.

iii) At early training the model needs little bit of big steps, so for that learning rate scheduler is very important, as it keeps the      learning rate high at first and then slowly reduces. 
