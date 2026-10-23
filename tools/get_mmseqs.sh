#!/usr/bin/env bash
# Fetch the static MMseqs2 build used by build_dataset.py.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x mmseqs/bin/mmseqs ]; then
    echo "already present: $(pwd)/mmseqs/bin/mmseqs"
    exit 0
fi

curl -L https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -o mmseqs-linux-avx2.tar.gz
tar xzf mmseqs-linux-avx2.tar.gz
echo "installed: $(pwd)/mmseqs/bin/mmseqs"
mmseqs/bin/mmseqs version
