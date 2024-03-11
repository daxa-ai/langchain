import os
import time

from langchain_openai.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, PodSpec, ServerlessSpec

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import Pinecone as PineconeVectorStore

use_serverless = True
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# configure client
pc = Pinecone(api_key=pinecone_api_key)

if use_serverless:
    spec = ServerlessSpec(cloud="aws", region="us-west-2")
else:
    # if not using a starter index, you should specify a pod_type too
    spec = PodSpec()

# check for and delete index if already exists
index_name = "langchain-retrieval-augmentation"
if index_name not in pc.list_indexes().names():
    # create a new index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric="dotproduct",
        spec=spec,
    )
else:
    print(f"Index {index_name} already exists. ")
    # #  Delete and create a new index
    # pc.delete_index(index_name)
    # # create a new index
    # pc.create_index(
    #     index_name,
    #     dimension=1536,  # dimensionality of text-embedding-ada-002
    #     metric="dotproduct",
    #     spec=spec,
    # )

# wait for index to be initialized
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

index = pc.Index(index_name)
index.describe_index_stats()

file_path = "data/identity-enf-data.csv"

loader = CSVLoader(
    file_path,
    source_column="Data",
    metadata_columns=["authorized_identities"],
)
documents = loader.load()

print("Creating embeddings and index...")
embeddings = OpenAIEmbeddings(client="")
texts = [t.page_content for t in documents]
metadatas = [t.metadata for t in documents]
docsearch = PineconeVectorStore.from_texts(
    texts, embeddings, metadatas=metadatas, index_name=index_name
)
print("Done!")

# pc_interface = Pinecone.from_existing_index(
#     index_name,
#     embedding=OpenAIEmbeddings(),
#     namespace="SessionIndex"
# )

index.describe_index_stats()
