"""
Unit tests for the PebbloRetrievalQA chain
"""
from typing import List
from unittest.mock import Mock

import pytest
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from langchain.chains.pebblo_retrieval.base import PebbloRetrievalQA
from tests.unit_tests.llms.fake_llm import FakeLLM


class FakeRetriever(VectorStoreRetriever):
    """
    Test util that parrots the query back as documents
    """

    vectorstore: VectorStore = Mock()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=query)]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [Document(page_content=query)]


class UnsupportedVectorStore(VectorStore):
    """
    Unsupported vectorstore class for testing
    """

    pass


@pytest.fixture
def unsupported_retriever() -> FakeRetriever:
    """
    Create a FakeRetriever instance
    """
    retriever = FakeRetriever()
    retriever.search_kwargs = {}
    # Set the class of vectorstore to Chroma
    retriever.vectorstore.__class__ = UnsupportedVectorStore
    return retriever


@pytest.fixture
def retriever() -> FakeRetriever:
    """
    Create a FakeRetriever instance
    """
    retriever = FakeRetriever()
    retriever.search_kwargs = {}
    # Set the class of vectorstore to Pinecone
    retriever.vectorstore.__class__ = Pinecone
    return retriever


@pytest.fixture
def pebblo_retrieval_qa(retriever: FakeRetriever) -> PebbloRetrievalQA:
    """
    Create a PebbloRetrievalQA instance
    """
    # Create a fake auth context
    auth_context = ["fake_user", "fake_user2"]
    pebblo_retrieval_qa: PebbloRetrievalQA = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
        auth_context=auth_context,
    )
    return pebblo_retrieval_qa


def test_invoke(pebblo_retrieval_qa: PebbloRetrievalQA) -> None:
    """
    Test invoke method
    """
    question = "What is the meaning of life?"
    response = pebblo_retrieval_qa.invoke({"query": question})
    assert response is not None


@pytest.mark.asyncio
async def test_ainvoke(pebblo_retrieval_qa: PebbloRetrievalQA) -> None:
    """
    Test ainvoke method (async)
    """
    with pytest.raises(NotImplementedError):
        _ = await pebblo_retrieval_qa.ainvoke({"query": "hello"})


def test_validate_auth_context(retriever: FakeRetriever) -> None:
    """
    Test auth_context validation
    """
    # Test with valid auth_context
    valid_auth_context = ["fake_user", "fake_user2"]
    _ = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
        auth_context=valid_auth_context,
    )

    # Test with invalid auth_context
    invalid_auth_context = "invalid_auth_context"
    with pytest.raises(ValueError):
        _ = PebbloRetrievalQA.from_chain_type(
            llm=FakeLLM(),
            chain_type="stuff",
            retriever=retriever,
            auth_context=invalid_auth_context,
        )

    # Test with None auth_context
    none_auth_context = None
    with pytest.raises(ValueError):
        _ = PebbloRetrievalQA.from_chain_type(
            llm=FakeLLM(),
            chain_type="stuff",
            retriever=retriever,
            auth_context=none_auth_context,
        )


def test_validate_vectorstore(
    retriever: FakeRetriever, unsupported_retriever: FakeRetriever
) -> None:
    """
    Test vectorstore validation
    """
    auth_context = ["fake_user", "fake_user2"]

    # Test with a supported vectorstore (Pinecone)
    _ = PebbloRetrievalQA.from_chain_type(
        llm=FakeLLM(),
        chain_type="stuff",
        retriever=retriever,
        auth_context=auth_context,
    )

    # Test with an unsupported vectorstore
    with pytest.raises(ValueError):
        _ = PebbloRetrievalQA.from_chain_type(
            llm=FakeLLM(),
            chain_type="stuff",
            retriever=unsupported_retriever,
            auth_context=auth_context,
        )
