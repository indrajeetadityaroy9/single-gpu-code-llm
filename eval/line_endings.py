"""The corpus's line endings for text a benchmark tool sends the model (paper §6.1): CR LF, then a lone CR,
as LF, as corpus/pretrain/prepare.py `lines` normalizes every training document, so the model never sees a "\\r" it
never trained on. Standard library only: the official tools' environments import it."""


def lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
