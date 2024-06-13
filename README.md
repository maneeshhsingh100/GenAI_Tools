# GenAI_Tools
==============================================================================
1- FST_Gen.py -- this code helps you to generate n numbers of sentnces. also you can modify this code according to your requirement. FST stands for Finite State of Transducer.

2- Sentiment.py -- this code helps you to check the positive and negative sense of any sentence using pre-trained model.

3- ASR.py -- this code helps you to use whisper model to  generate speech to text.

4- Sentiment_Checker_Bert.py -- this code is using pre/finetuned bert model to check posittive and negative sense of any sentence. To use this tool first install "pip install transformers torch".

---------------------------------------------------------------------------------
 the output will be 
 "Sentence: I love this product! It works great and is exactly what I needed.
Sentiment: [{'label': 'POSITIVE', 'score': 0.9998726844787598}]

Sentence: I am very disappointed with this service. It was terrible and I won't be using it again.
Sentiment: [{'label': 'NEGATIVE', 'score': 0.9995607733726501}]

---------------------------------------------------------------------------------

==========================================================================

If you want to TensorFlow(TF) to handle your NLP tasks, These 5 tools help you to do simple NLP tasks.

        NER_checker.py
        sentiment_checker_tf.py
        text_classification.py
        tokenizer_tool.py
        word_embedding.py

 To use this tools first of all you need to install tensorflow.

 		pip install tensorflow 
 				or
 		pip install tensorflow==2.8.0

===================================================================================


5- Sentiment_data_generator.py -- if you have token/word list. Then this tool helps you to generate data for positive and negative sentiment data set. 

please install 

 		pip install transformers
 				

==================================================
