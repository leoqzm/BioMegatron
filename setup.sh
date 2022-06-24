%%writefile setup.sh
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5d8c8a8eedaf567d56f0762a45431baf9c0e800e
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" ./