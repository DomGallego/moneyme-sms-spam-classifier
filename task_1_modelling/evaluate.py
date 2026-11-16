"""Standalone task entry point; delegates to the shared package CLI."""

import sys

from moneyme_sms_spam_classifier.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "finalize")
    main()
