import argparse

from utils.precompute_embeddings import precompute_embeddings
from utils.retrieve_docs import retrieve_docs


def get_args():
    parser = argparse.ArgumentParser()
    # action args
    parser.add_argument("--do_precompute", action="store_true")
    parser.add_argument("--do_retrieve", action="store_true")
    # model args
    parser.add_argument("--load_model_path", type=str, default="./data/msmarco-passage/checkpoints/step-430000")
    # data args
    parser.add_argument("--tokenize_dir", type=str, default="./data/msmarco-passage/tokenize")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/msmarco-passage/collection_memmap/complete")
    parser.add_argument("--embedding_dir", type=str, default="./data/msmarco-passage/embeddings")
    parser.add_argument("--retrieved_result_dir", type=str, default="./data/msmarco-passage/retrieved_result")
    # inference args
    parser.add_argument("--offline_precompute_device", type=str, default="cuda:2")
    parser.add_argument("--online_retrieve_device", type=str, default="cuda:2")
    parser.add_argument("--offline_precompute_batch_size_per_gpu", type=int, default=40)
    parser.add_argument("--online_retrieve_batch_size", type=int, default=100)
    parser.add_argument("--hit_num", type=int, default=1000)
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=256)
    parser.add_argument("--docs_per_gpu", type=int, default=1500000)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.do_precompute:
        precompute_embeddings(
            load_model_path=args.load_model_path,
            collection_memmap_dir=args.collection_memmap_dir,
            embedding_dir=args.embedding_dir,
            device=args.offline_precompute_device,
            batch_size_per_gpu=args.offline_precompute_batch_size_per_gpu,
            max_doc_length=args.max_doc_length
        )
    if args.do_retrieve:
        retrieve_docs(
            load_model_path=args.load_model_path,
            device=args.online_retrieve_device,
            batch_size=args.offline_precompute_batch_size_per_gpu,
            hit_num=args.hit_num,
            embedding_dir=args.embedding_dir,
            tokenize_dir=args.tokenize_dir,
            retrieved_result_dir=args.retrieved_result_dir,
            docs_per_gpu=args.docs_per_gpu,
            max_query_length=args.max_query_length
        )


if __name__ == "__main__":
    main()