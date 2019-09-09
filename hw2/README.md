# CS294-112 HW 2: Policy Gradient
To reproduce run the following:

```
    ./p4.sh
    ./p5.sh
    python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n \
        100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name \
        ll_b40000_r0.005
    ./p8HyperParamSearch.sh
    ./p8.sh
```

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only file that you need to look at is `train_pg_f18.py`, which you will implement.

See the [HW2 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf) for further instructions.
