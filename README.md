# QuadTAG
This is the official implementation for "[AQE: Argument Quadruplet Extraction via a Quad-Tagging Augmented Generative Approach](https://github.com/guojiapub/QuadTAG)" (Findings of ACL 2023).

## Citation
If you find our work helpful for your research, please cite our paper:

```bibtex
@inproceedings{guo-etal-2023-aqe,
    title = "{AQE}: Argument Quadruplet Extraction via a Quad-Tagging Augmented Generative Approach",
    author = "Guo, Jia  and
      Cheng, Liying  and
      Zhang, Wenxuan  and
      Kok, Stanley  and
      Li, Xin  and
      Bing, Lidong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
    url = "https://aclanthology.org/2023.findings-acl.59",
}
```

## Dependencies
- Python 3.6+
- PyTorch 1.7.1+
- cudatoolkit 11.0+
- NumPy 1.19.2+
- tqdm 4.63.0+

### Training and Evaluation
```shell script
python main.py --do_train --do_dev --do_test --output_dir ./checkpoints/QuadTAG 
```

