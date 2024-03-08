"""Pebblo Retrieval Chain with Identity & Semantic Enforcement for question-answering against a vector database."""

from typing import List

from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from pydantic import Field

from langchain.chains.retrieval_qa.base import RetrievalQA


class PebbloRetrievalQA(RetrievalQA):
    """Retrieval Chain with Identity & Semantic Enforcement for question-answering against a vector database."""

    auth_context: dict = Field(exclude=True)
    """Auth context to use in the retriever."""

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""

        #  Identity Enforcement
        search_kwargs = self.retriever.search_kwargs
        if (
            "filter" in search_kwargs
            and "authorized_identities" in search_kwargs["filter"]
        ):
            raise ValueError(
                "authorized_identities already exists in search_kwargs['filter']"
            )

        search_kwargs.setdefault("filter", {})["authorized_identities"] = (
            self.auth_context
        )

        docs = super()._get_docs(question, run_manager=run_manager)
        return docs

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        raise NotImplementedError("PebbloRetrievalQA does not support async")

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "pebblo_retrieval_qa"
