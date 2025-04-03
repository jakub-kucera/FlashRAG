import argparse
import json
import numpy as np
import weaviate
import weaviate.classes as wvc


def create_weaviate_index(
        host,
        port,
        collection_name,
        corpus_path,
        embeddings_path,
        embedding_dim,
        batch_size
):
    # Connect to Weaviate
    client = weaviate.connect_to_local(host=host, port=port)

    existing_collections = client.collections.list_all()
    print(f"Existing collections: {existing_collections}")

    print(f"Creating collection: {collection_name}")
    # Create the collection
    questions = client.collections.create(
        collection_name,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        properties=[
            wvc.config.Property(
                name="orig_id",
                data_type=wvc.config.DataType.INT,
            ),
            wvc.config.Property(
                name="file",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=False,
                index_searchable=False,
            ),
            wvc.config.Property(
                name="name",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=False,
                index_searchable=False,
            ),
            wvc.config.Property(
                name="name_lemmas",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
            ),
            wvc.config.Property(
                name="contents",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=False,
                index_searchable=False,
            ),
            wvc.config.Property(
                name="contents_lemmas",
                data_type=wvc.config.DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
            ),
        ]
    )

    # Optional: list all collections to confirm creation
    existing_collections = client.collections.list_all()
    print(f"Existing collections: {existing_collections}")

    # Read the corpus
    with open(corpus_path, "r") as f:
        corpus = [json.loads(line) for line in f]

    # Load embeddings via memmap
    embeddings = np.memmap(
        embeddings_path,
        dtype='float32',
        mode='r'
    )
    # Reshape to (num_embeddings, embedding_dim)
    num_embeddings = embeddings.shape[0] // embedding_dim
    embeddings = embeddings.reshape((num_embeddings, embedding_dim))

    # Just to confirm that embeddings loaded correctly, print the first vector
    # (optional for debugging)
    print("First embedding preview:", embeddings[0][:5], "...")

    # Prepare objects
    weaviate_objs = []
    for i, item in enumerate(corpus):
        weaviate_objs.append(
            wvc.data.DataObject(
                properties={
                    "orig_id": item["id"],
                    "file": item["file"],
                    "name": item["name"],
                    "name_lemmas": item["name_lemmatized"],
                    "contents": item["contents"],
                    "contents_lemmas": item["contents_lemmatized"],
                },
                vector=embeddings[i],
            )
        )

    # Get the collection reference
    nss_corpus = client.collections.get(collection_name)

    # Insert in batches
    for i in range(0, len(weaviate_objs), batch_size):
        print(f"Inserting batch from {i} to {i + batch_size} ...")
        batch = weaviate_objs[i:i + batch_size]
        nss_corpus.data.insert_many(batch)
    print("All objects inserted.")
    client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create a Weaviate index from a JSONL corpus and memmapped embeddings.")
    parser.add_argument("--host", type=str, default="localhost", help="Weaviate host address.")
    parser.add_argument("--port", type=int, default=8080, help="Weaviate port.")
    parser.add_argument("--collection_name", type=str, default="NSS_corpus_full",
                        help="The Weaviate collection name to create.")
    parser.add_argument("--corpus_path", type=str, help="Path to the corpus JSONL file.", required=True)
    parser.add_argument("--embeddings_path", type=str, help="Path to the embeddings memmap file.", required=True)
    parser.add_argument("--embedding_dim", type=int, default=768, help="Dimensionality of the embeddings.")
    parser.add_argument("--batch_size", type=int, default=10000, help="Number of objects to insert per batch.")

    args = parser.parse_args()
    create_weaviate_index(
        host=args.host,
        port=args.port,
        collection_name=args.collection_name,
        corpus_path=args.corpus_path,
        embeddings_path=args.embeddings_path,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
