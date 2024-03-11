import textwrap
import time
from typing import List

from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI

from langchain.chains import PebbloRetrievalQA
from langchain.vectorstores import Pinecone as PineconeVectorStore

load_dotenv()


class AuthAcmeCorpRAG:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.app_name = "auth-acme-corp-rag-1"
        self.index_name = "langchain-retrieval-augmentation"

        # # Load documents
        # print("Loading RAG documents ...")
        # self.loader = CSVLoader(
        #     self.file_path,
        #     source_column="Data",
        #     metadata_columns=["authorized_identities"],
        # )
        # self.documents = self.loader.load()
        # self.filtered_docs = filter_complex_metadata(self.documents)
        # print(f"Loaded {len(self.documents)} documents ...\n")

        # Load documents into VectorDB
        print("Hydrating Vector DB ...")
        self.vectordb = self.embeddings()
        print("Finished hydrating Vector DB ...\n")

        self.llm = OpenAI()

    def embeddings(self):
        embeddings = OpenAIEmbeddings()
        # vectordb = PineconeVectorStore.from_documents(
        #     docs, embeddings, index_name=self.index_name
        # )
        vectordb = PineconeVectorStore.from_existing_index(
            self.index_name, embedding=embeddings
        )
        return vectordb

    def ask(self, question: str, auth_identifiers: List[str]):
        # Prepare retriever QA chain
        auth = {"$in": auth_identifiers}
        retriever = PebbloRetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(),
            verbose=True,
            auth_context=auth,
        )

        # # Error condition
        # auth = {"$in": auth_identifiers}
        # search_kwargs = {"filter": {"authorized_identities": {"$in": auth_identifiers}}}
        # retriever = PebbloRetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=self.vectordb.as_retriever(search_kwargs=search_kwargs),
        #     verbose=True,
        #     auth_context=auth,
        # )

        # # Other search_kwargs in retriever
        # auth = {"$in": auth_identifiers}
        # search_kwargs = {"filter": {"topic": auth_identifiers}}
        # retriever = PebbloRetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=self.vectordb.as_retriever(search_kwargs=search_kwargs),
        #     verbose=True,
        #     auth_context=auth,
        # )

        return retriever.invoke(question)


if __name__ == "__main__":
    prompts = [
        {
            "query": "What does the document say about Wipro?",
            "authorized_identities": ["Governance", "Finance"],
        },
        {
            "query": "What does the document say about Wipro?",
            "authorized_identities": ["Finance"],
        },
        {
            "query": "Why did Eva Cheng decide not to stand for re-election?",
            "authorized_identities": ["Governance"],
        },
        {
            "query": "Why did Eva Cheng decide not to stand for re-election?",
            "authorized_identities": ["Legal"],
        },
        {
            "query": "What are the outcomes of the board meetings for each company?",
            "authorized_identities": ["Governance"],
        },
        {
            "query": "What are the outcomes of the board meetings for each company?",
            "authorized_identities": ["Finance", "Legal"],
        },
    ]

    #  start time
    start = time.time()

    rag_app = AuthAcmeCorpRAG("data/identity-enf-data.csv")
    print(120 * "#", "\n")

    limit = None
    for prompt in prompts[:limit]:
        query = prompt["query"]
        authorized_identities = prompt["authorized_identities"]
        print(f"Authorized_identities: {authorized_identities}, \nQuery:\n{query}")
        response = rag_app.ask(query, authorized_identities)
        formatted_response = textwrap.fill(response["result"], width=120)
        print(f"Response:\n{formatted_response}")
        print(120 * "#", "\n")

    #  end time
    end = time.time()

    print(f"Time taken: {end - start} seconds")
