<h1 align="center">
Evaluating Attribute Confusion in Fashion 
<br>Text-to-Image Generation
</h1>

<div>
    <p align="center">
    🔥 ICIAP 2025 - <a href="https://intelligolabs.github.io/L-VQAScore">Project Page</a></strong> 🔥
    </p>
</div>

<div>
    <p align="center">
    Ziyue Liu, Federico Girella, Yiming Wang, Davide Talon
    </p>
</div>

<h3 align="center">

<hr>

## Abstract
Despite the rapid advances in Text-to-Image (T2I) generation models, their evaluation remains challenging in domains like fashion, involving complex compositional generation. Recent automated T2I evaluation methods leverage pre-trained vision-language models to measure cross-modal alignment. However, our preliminary study reveals that they are still limited in assessing rich entity-attribute semantics, facing challenges in attribute confusion, i.e., when attributes are correctly depicted but associated to the wrong entities. To address this, we build on a Visual Question Answering (VQA) localization strategy targeting one single entity at a time across both visual and textual modalities. 

We propose a localized human evaluation protocol and introduce a novel automatic metric, Localized VQAScore (L-VQAScore), that combines visual localization with VQA probing both correct (reflection) and miss-localized (leakage) attribute generation. On a newly curated dataset featuring challenging compositional alignment scenarios, L-VQAScore outperforms state-of-the-art T2I evaluation methods in terms of correlation with human judgments, demonstrating its strength in capturing fine-grained entity-attribute associations. We believe L-VQAScore can be a reliable and scalable alternative to subjective evaluations.

---

## L-VQAScore

L-VQAScore pipeline perform automatic item cropping and VQA-style scoring, with images and corresponding JSON annotation file describing item-attribute pair.


## 📦 Requirements

Install dependencies following requirements. <a href="https://github.com/IDEA-Research/Grounded-SAM-2">Grounded-SAM-2</a> and <a href="https://github.com/linzhiqiu/t2v_metrics">T2V</a> need to be installed.


## 🗂 JSON Format

Provide your annotation data structure like this:
```
    [
        {
            "image_id": "001",
            "image_path": "/path/to/image_001.jpg",
            "items": [
                {
                    "item_name": "shirt",
                    "attributes": ["white", "striped"],
                    "questions": ["a white shirt", "a striped shirt"]
                },
                ...
            ]
        },
        ...
    ]
```

We recommend using data that contains multiple items and attributes, as it leads to more reliable and stable evaluation regarding attribute confusion.

## 🚀 Quick Start

Run the Script: `bash main.sh`

## ✨ Citation

If you find our work usefull, please cite our work:
```
@inproceedings{liu2025evaluating,
  title={Evaluating Attribute Confusion in Fashion Text-to-Image Generation},
  author={Liu, Ziyue and Girella, Federico and Wang, Yiming and Talon, Davide and others},
  booktitle={23rd International Conference on Image Analysis and Processing},
  year={2025}
}
```





