"""Download upstream dataset without redistributing it; review upstream terms first."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", default="data/telco.csv")
a = p.parse_args()
u = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
b = urllib.request.urlopen(u, timeout=60).read()
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(b)
print("sha256", hashlib.sha256(b).hexdigest())
