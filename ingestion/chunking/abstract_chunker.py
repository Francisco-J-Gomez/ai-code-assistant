from ingestion.chunking.base import Chunk
from ingestion.models import PaperMetadata


class AbstractChunker:
    """MVP ChunkerStrategy: one chunk per paper, the abstract itself.

    No splitting needed at this length (docs/data-ingestion-plan.md step 6).
    """

    def chunk(self, paper: PaperMetadata) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{paper.arxiv_id}:0",
                arxiv_id=paper.arxiv_id,
                chunk_index=0,
                chunk_type="abstract",
                text=paper.abstract,
                title=paper.title,
                authors=paper.authors,
                categories=paper.categories,
                published=paper.published,
                source_url=paper.pdf_url,
            )
        ]
