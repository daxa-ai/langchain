"""
Pebblo Retrieval Chain with Identity & Semantic Enforcement for question-answering
against a vector database.
"""

from typing import Any, List

from langchain_community.vectorstores import Pinecone, Qdrant, Weaviate
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.vectorstores import VectorStoreRetriever

from langchain.chains.retrieval_qa.base import RetrievalQA

SUPPORTED_VECTORSTORES = [Pinecone, Weaviate, Qdrant]


class PebbloRetrievalQA(RetrievalQA):
    """
    Retrieval Chain with Identity & Semantic Enforcement for question-answering
    against a vector database.
    """

    retriever: VectorStoreRetriever = Field(exclude=True)
    """VectorStore to use for retrieval."""

    auth_context: list = Field(exclude=True)
    """Auth context to use in the retriever."""

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        # Identity Enforcement
        search_kwargs = self.retriever.search_kwargs
        if isinstance(self.retriever.vectorstore, Pinecone):
            if (
                "filter" in search_kwargs
                and "authorized_identities" in search_kwargs["filter"]
            ):
                raise ValueError(
                    "authorized_identities already exists in search_kwargs['filter']"
                )

            search_kwargs.setdefault("filter", {})["authorized_identities"] = {
                "$in": self.auth_context
            }
        elif isinstance(self.retriever.vectorstore, Qdrant):
            from qdrant_client.http import models as rest

            filters = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.authorized_identities",
                        match=rest.MatchAny(any=self.auth_context),
                    )
                ]
            )
            search_kwargs.setdefault("filter", filters)
        elif isinstance(self.retriever.vectorstore, Weaviate):
            where_filter = {
                "path": ["authorized_identities"],
                "operator": "ContainsAny",
                "valueText": self.auth_context,
            }
            search_kwargs.setdefault("where_filter", where_filter)
        else:
            raise ValueError(
                f"Vectorstore must be an instance of one of the supported "
                f"vectorstores: {SUPPORTED_VECTORSTORES}. "
                f"Got {type(self.retriever.vectorstore).__name__} instead."
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

    @validator("auth_context")
    def validate_auth_context(cls, auth_context: Any) -> list:
        """
        Validate auth_context
        """
        # auth_context must be a list
        if not isinstance(auth_context, list):
            raise ValueError("auth_context must be a list")
        return auth_context

    @validator("retriever", pre=True, always=True)
    def validate_vectorstore(
        cls, retriever: VectorStoreRetriever
    ) -> VectorStoreRetriever:
        """
        Validate that the vectorstore of the retriever is supported vectorstores.
        """
        if not any(
            isinstance(retriever.vectorstore, supported_class)
            for supported_class in SUPPORTED_VECTORSTORES
        ):
            raise ValueError(
                f"Vectorstore must be an instance of one of the supported "
                f"vectorstores: {SUPPORTED_VECTORSTORES}. "
                f"Got {type(retriever.vectorstore).__name__} instead."
            )
        return retriever
