"""
Training data under paths.CORPUS (paper §2, §3): the pretraining corpus's acquisition and curation in pretrain/, the
SFT data in posttrain/, and the shared steps here (decontamination, tokenizer, tokenization, packing, loading).
Pretraining pipeline, in order: pretrain.prepare, pretrain.curate, decontaminate, train_tokenizer, pretrain.publish
(push uploads the curated shards, the decontamination record and the tokenizer; pull downloads them), tokenize, pack.
Then posttrain.build builds the SFT data; loader reads the packed sequences during training.
"""
