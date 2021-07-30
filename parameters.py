import argparse
parser = argparse.ArgumentParser(description='DRL-LSJSP')

# env parameters
parser.add_argument('--j', type=int, default=10)
parser.add_argument('--m', type=int, default=10)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--h', type=int, default=99)
parser.add_argument('--init_type', type=str, default='fdd-divide-mwkr')
parser.add_argument('--reward_type', type=str, default='yaoxin')
# model parameters
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--embedding_layer', type=int, default=4)
parser.add_argument('--policy_layer', type=int, default=4)
# training parameters
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--steps_learn', type=int, default=10)
parser.add_argument('--transit', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--episodes', type=int, default=128000)
parser.add_argument('--step_validation', type=int, default=10)

args = parser.parse_args()