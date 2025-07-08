<h1 align="center">
Evaluating Attribute Confusion in Fashion 
    Text-to-Image Generation
</h1>

<div>
    <p align="center">
    ICIAP 2025 - <a href="https://intelligolabs.github.io/L-VQAScore">Project Page</a></strong>  
    </p>
</div>

<div>
    <p align="center">
    <a Ziyue Liu, Federico Girella, Yiming Wang, Davide Talon
    </p>
</div>

<h3 align="center">

<hr>

## Abstract
Despite the rapid advances in Text-to-Image (T2I) generation models, their evaluation remains challenging in domains like fashion, involving complex compositional generation. Recent automated T2I evaluation methods leverage pre-trained vision-language models to measure cross-modal alignment. However, our preliminary study reveals that they are still limited in assessing rich entity-attribute semantics, facing challenges in attribute confusion, i.e., when attributes are correctly depicted but associated to the wrong entities. To address this, we build on a Visual Question Answering (VQA) localization strategy targeting one single entity at a time across both visual and textual modalities. We propose a localized human evaluation protocol and introduce a novel automatic metric, Localized VQAScore (L-VQAScore), that combines visual localization with VQA probing both correct (reflection) and miss-localized (leakage) attribute generation. On a newly curated dataset featuring challenging compositional alignment scenarios, L-VQAScore outperforms state-of-the-art T2I evaluation methods in terms of correlation with human judgments, demonstrating its strength in capturing fine-grained entity-attribute associations. We believe L-VQAScore can be a reliable and scalable alternative to subjective evaluations.


