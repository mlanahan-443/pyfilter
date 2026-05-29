from pyfilter.models.linear._transitions import IntegratorChainTransition

c = IntegratorChainTransition(n=2, p=3)
print(c.state_dim)  # must be 6
print(c._eye_n.shape)  # must be (2, 2)
print([f.shape for f in c._temporal_factors])  # all (3, 3)
