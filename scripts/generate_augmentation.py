"""Generate and cache training-only SMS variants under the shared authorized cap."""

import argparse

from moneyme_sms_spam_classifier.augmentation import generate_pool

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--limit", type=int, default=12, help="Number of parents; 0 means all spam parents"
)
parser.add_argument("--rounds", type=int, default=1)
args = parser.parse_args()
generate_pool(args.limit, args.rounds)
