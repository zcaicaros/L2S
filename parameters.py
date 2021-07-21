import argparse
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--j', type=int, default=10)
parser.add_argument('--m', type=int, default=10)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--h', type=int, default=99)
parser.add_argument('--transit', type=int, default=300)
parser.add_argument('--episodes', type=int, default=64000)  # 384000
args = parser.parse_args()