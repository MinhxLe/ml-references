Score matching

# intro
In score matching, the objective is to match the gradient of the log marginal data distribution aka the "score". This requires estimating the gradient of the unknown grad(P_data). THis is done by minimizing [[Fisher Divergence]]. 


Rewritten, the objective can be found w/o having P_data but requires  computing the trace of the Hessian.


# denosing score matching
This addreses handling discrete distributions by adding a smooth continuous noise.


# Random
Your loss is a function of the gradient which means you need to roll out th ecomputation graph. 
