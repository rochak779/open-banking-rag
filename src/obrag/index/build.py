"""One command to go from nothing to a queryable index."""

import argparse

from obrag.config import load_settings
from obrag.index.embedder import DEFAULT_BATCH_SIZE, Embedder
from obrag.index.store import COLLECTIONS, ChunkStore
from obrag.ingest.fetch import LEGISLATION_SOURCES, fetch_legislation, fetch_obl_spec
from obrag.ingest.legislation import parse_all_legislation
from obrag.ingest.obl_spec import parse_spec_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the obrag vector collections.")
    parser.add_argument("--skip-fetch", action="store_true", help="use the existing data/raw snapshot")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="chunks per Voyage request; lower it if you hit rate limits",
    )
    args = parser.parse_args()

    settings = load_settings()
    raw_dir = settings.raw_dir

    if not args.skip_fetch:
        for name in LEGISLATION_SOURCES:
            print(f"fetching {name}...")
            fetch_legislation(name, raw_dir=raw_dir)
        print(f"fetching OBL spec {settings.obl_spec_tag}...")
    spec_dir = fetch_obl_spec(raw_dir=raw_dir, tag=settings.obl_spec_tag)

    chunks = parse_all_legislation(raw_dir) + parse_spec_dir(spec_dir)
    print(f"{len(chunks)} chunks to embed with {settings.embedding_model}")

    embedder = Embedder(settings, batch_size=args.batch_size)
    store = ChunkStore(settings)
    # Upsert batch by batch so a failure partway keeps everything embedded so far.
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        store.upsert(batch, embedder.embed_documents([c.text for c in batch]))
        print(f"  {start + len(batch)}/{len(chunks)}")

    for name in COLLECTIONS:
        print(f"{name}: {store.count(name)} chunks indexed")


if __name__ == "__main__":
    main()
