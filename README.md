# Deep RL Exploration Framework
To run,

```
python main.py
```

Alter openai gym environments by changing the `params` for `make_env` in `main.py`.

Learner classes are in `learners.py`, exploration functions are in `exploration.py`.

Implements Double Deep Q-Learning, with Epsilon Greedy, Count-based Optimism and Value-Difference-based Exploration. 
