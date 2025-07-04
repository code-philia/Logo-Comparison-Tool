# Logo-Comparison-Tool

## Setup

On Ubuntu/Mac/Windows, download anaconda [https://www.anaconda.com/download](https://www.anaconda.com/download).
Turn on a VPN that can access google drive.

1. For Ubuntu/Mac users, open terminal, run setup.sh, this takes some time
```bash
chmod +x setup.sh
./setup.sh
```

For Windows users, open a Command Prompt (cmd.exe), run setup.bat, this takes some time
```bash
C:\path\to\your\project> setup.bat
```

After this setup, a **models/** directory will be created with the following structure
```
models/
  |_ resnetv2_rgb_new.pth.tar # matching model checkpoint
  |_ expand_targetlist/ # all set of logo reference database
    |_ adidas
    |_ Adobe
    |_ ...
```

2. Run inference with
```bash
conda activate phishpedia
python main.py --query_img [query image path, e.g. test.png] --visualize_path [where to save the visualization, e.g. test_result.png]
```

See the results in **test_result.png**
<img src="test_result.png">

