# Cooperative models for chess prediction and position explaination

## Idea and intuition

The idea of this architecture is to build a solution, that can both predict the value of a current chess position aswell as explain it to the player in natural language.  

Final solution should be able to do the following:

- accept an arbitrary chess position embedding from a pretrained model (model at the embedding layer should sufficiently encapsulate the "knowladge" about the chess position)
  - this allows tuning different models for the embedding and offers greater flexibility.
  - decouples the embedding from the explaination model
- first (explainer) should generate a meaningfull explaination embedding from the current position embedding
  - explaination embedding should be a vector that can be fed into a pretrained language model (e.g. GPT-2) to generate a natural language explaination
- second (predictor) should predict the value of the current position from the explaination embedding
  - value prediction should be a centipawn value (cpv) prediction

## Approach

Given that a neural network can be trained to process a chess position and output a metric of some sort (a centipawn position or a probability distribution over the next move), we know that if we cut the network at some layer, we can get a vector representation of the position, holding all the needed data to acomp,lish the task.  

We can use this model, freeze it and use it as a feature extractor for the explaination model.

## problem...didnt really think it through...

We cant just feed the output of the embedding layer into the explaination model, since the embedding wont make any sense to the explaination model.  
We first need a pretrained model that can describe chess positions in a meaningful way.