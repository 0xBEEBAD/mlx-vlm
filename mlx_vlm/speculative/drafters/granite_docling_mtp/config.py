from dataclasses import dataclass
from typing import Optional

from ....models.base import BaseModelConfig
from ....models.granite_docling.config import TextConfig


@dataclass
class GraniteDoclingMTPConfig(BaseModelConfig):
    model_type: str = "granite_docling_mtp"
    text_config: Optional[TextConfig] = None
    block_size: int = 4  # One verified token and one draft token for each head
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if self.text_config is None:
            raise ValueError("Granite Docling MTP requires text_config.")
        if self.text_config.num_mtp_layers < 1:
            raise ValueError("Granite Docling MTP requires num_mtp_layers >= 1.")
