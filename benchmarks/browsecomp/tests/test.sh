#!/bin/bash
# Close the answer log (finalize.py); the answer is graded on the host after the run.
mkdir -p /logs/verifier
python3 /tests/finalize.py 2>&1 | tee /logs/verifier/evaluation.log
if [ ! -f /logs/verifier/reward.txt ]; then
  echo "0.0" > /logs/verifier/reward.txt
fi
