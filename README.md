# LLM-from-scratch
LLM created in python,  completely from scratch with numpy.


Architecture of LLM...

                  Text 
                    |
                    v
           character level tokenization
                    |
                    v
           384 Dimention embeddings
                    |
                    v
           Transformer Block 1-6
                    |
                    v
           multiplication with Wo(weights) & adding Bo(Biases)
                    |
                    v
           softmax for probability distribution
                    |
                    v
             next token predicted



Transformer Block....
i used classical PE(Positional Encoding), MHA (multi head attention), Plain FFN(Dense - Feed forward layer), Layer Normalization, and added residual after attention and FFN. 
One transformer block consist of ~1.77 Million parameter, and LLM is made of 6 of these transformer blocks.
This LLM is trained on TinyShakespere, 32 batches, and have 128 context window.
    
    

