import argparse
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--j', type=float, default=10)
parser.add_argument('--m', type=float, default=10)
parser.add_argument('--l', type=float, default=1)
parser.add_argument('--h', type=float, default=99)
parser.add_argument('--transit', type=int, default=32)
parser.add_argument('--episodes', type=int, default=384000)  # 384000
args = parser.parse_args()