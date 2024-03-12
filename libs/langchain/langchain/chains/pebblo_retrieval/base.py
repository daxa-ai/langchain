"""
Pebblo Retrieval Chain with Identity & Semantic Enforcement for question-answering
against a vector database.
"""

from typing import Any, List

from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.vectorstores import VectorStoreRetriever

from langchain.chains.retrieval_qa.base import RetrievalQA


class PebbloRetrievalQA(RetrievalQA):
    """
    Retrieval Chain with Identity & Semantic Enforcement for question-answering
    against a vector database.
    """

    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStore to use for retrieval."""

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

        search_kwargs.setdefault("filter", {})[
            "authorized_identities"
        ] = self.auth_context
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

    @validator("auth_context")
    def validate_auth_context(cls, auth_context: Any) -> dict:
        """
        Validate auth_context
        """
        # auth_context must be a dictionary
        if not isinstance(auth_context, dict):
            raise ValueError("auth_context must be a dictionary")
        return auth_context

    @validator("retriever", pre=True, always=True)
    def validate_vectorstore(
        cls, retriever: VectorStoreRetriever
    ) -> VectorStoreRetriever:
        """
        Validate that the vectorstore of the retriever is supported vectorstores.
        """
        supported_vectorstores = [Pinecone]
        if not any(
            isinstance(retriever.vectorstore, supported_class)
            for supported_class in supported_vectorstores
        ):
            raise ValueError(
                f"vectorstore must be an instance of one of the supported "
                f"vectorstores: {supported_vectorstores}. "
                f"Got {type(retriever.vectorstore).__name__} instead."
            )
        return retriever
