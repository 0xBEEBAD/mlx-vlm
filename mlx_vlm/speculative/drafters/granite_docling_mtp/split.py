from ..mtp_split import MTPSplitter
from .granite_docling_mtp import GraniteDoclingMTPDraftModel


class GraniteDoclingMTPSplitter(MTPSplitter):
    output_model_type = "granite_docling_mtp"
    draft_model_cls = GraniteDoclingMTPDraftModel
    tie_word_embeddings_default = True
    depth_field = "num_mtp_layers"
    tokenizer_files = ()  # The drafter uses the embeddings of the target

    def select_keys(self, key, text_config):
        return key.startswith("mtp.layers.")

    def depth(self, text_config):
        return int(text_config.get(self.depth_field) or 0)

    def rename(self, tensors, text_config):
        return {key.removeprefix("mtp."): value for key, value in tensors.items()}
