# chess AI something something

  research heavily based on [this paper](https://www.cs.tau.ac.il/~wolf/papers/deepchess.pdf)

## vector representation of each board position

  we will use some sort of [dbn](https://en.wikipedia.org/wiki/Deep_belief_network) to compress the states of each chess board in to a representation vector.

  Frist attempt is to calculate the position "strenght" from given position.  
  A supervised approach with games from [lichess](https://lichess.org/) evaluated by stockfish.  
  Model should learn to give the simmilar score to the one given by [stockfish](https://stockfishchess.org/)

  
