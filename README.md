# active-area-search


Demo_1d
---

A minimal 1d demo for active area search, with contrast to expected improvements.

Instruction: run region_reward_1d.m ; the comparison is expected_reward_1d.m .

Misc: Self-inclusive; gif and pdf files will be overwritten. 

Demo_2d
---

Reproduction of the simulation experiment (Figure 3) in the paper, using a more generic matlab class object.

Instruction: run startup.m ; then run demo_10x10.m .

Misc: The demo data contains one draw from a Gaussian process; one can verify its validity by attempting to optimize the GP hyper-parameters which returns the same choice of parameters.

Ref: Yifei Ma, Roman Garnett, Jeff Schneider. [Active Area Search via Bayesian Quadrature. ](http://www.jmlr.org/proceedings/papers/v33/ma14.pdf) AISTATS 2014.

