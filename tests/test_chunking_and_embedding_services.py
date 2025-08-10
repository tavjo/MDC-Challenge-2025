import pytest
import pandas as pd
import re
from types import SimpleNamespace
from datetime import datetime

import api.services.chunking_and_embedding_services as service
from src.models import DocumentChunkingResult, ChunkingResult, Chunk, ChunkMetadata, CitationEntity, Document


def test_prepare_document_happy(monkeypatch):
    # Arrange
    doc = SimpleNamespace(doi='doc1')
    citations = []
    pre_df = pd.DataFrame({'citation_id': ['cid1'], 'count': [2]})
    monkeypatch.setattr(service, 'create_pre_chunk_entity_inventory', lambda d, c: pre_df)
    dummy_chunk = SimpleNamespace(text='dummy', chunk_metadata=SimpleNamespace(token_count=3))
    monkeypatch.setattr(service, 'create_chunks_from_document', lambda d, c, cs, co: [dummy_chunk])
    monkeypatch.setattr(service, 'link_adjacent_chunks', lambda chunks: chunks)
    empty_df = pd.DataFrame({'citation_id': []})
    monkeypatch.setattr(service, 'validate_chunk_integrity', lambda chunks, p, c: (True, empty_df))
    monkeypatch.setattr(service, 'repair_lost_citations_strict', lambda d, ch, ld: ([dummy_chunk], {}))
    monkeypatch.setattr(service, 'make_pattern', lambda cid: re.compile('x^'))

    # Act
    result = service.prepare_document(
        doc, citations,
        chunk_size=10,
        chunk_overlap=5,
        cfg_path='cfg-path',
        collection_name='col'
    )

    # Assert
    assert result['document'] is doc
    assert result['chunks'] == [dummy_chunk]
    assert result['pre_total_citations'] == 2
    assert result['post_total_citations'] == 0
    assert result['validation_passed'] is True
    assert result['lost_entities'] is None
    assert result['total_chunks'] == 1
    assert result['total_tokens'] == 3
    assert result['avg_tokens'] == 3 / 1
    assert 'pipeline_started_at' in result
    assert result['chunk_size'] == 10
    assert result['chunk_overlap'] == 5


def test_prepare_document_error(monkeypatch):
    # Arrange
    doc = SimpleNamespace(doi='doc1')
    citations = []
    pre_df = pd.DataFrame({'citation_id': ['cid1'], 'count': [4]})
    monkeypatch.setattr(service, 'create_pre_chunk_entity_inventory', lambda d, c: pre_df)
    dummy_chunk = SimpleNamespace(text='dummy', chunk_metadata=SimpleNamespace(token_count=3))
    monkeypatch.setattr(service, 'create_chunks_from_document', lambda d, c, cs, co: [dummy_chunk])
    monkeypatch.setattr(service, 'link_adjacent_chunks', lambda chunks: chunks)
    lost_df = pd.DataFrame({'citation_id': ['cid1']})
    monkeypatch.setattr(service, 'validate_chunk_integrity', lambda chunks, p, c: (False, lost_df))
    monkeypatch.setattr(service, 'repair_lost_citations_strict', lambda d, ch, ld: ([dummy_chunk], {}))
    monkeypatch.setattr(service, 'make_pattern', lambda cid: re.compile('x^'))

    # Act
    result = service.prepare_document(
        doc, citations,
        chunk_size=7,
        chunk_overlap=2,
        cfg_path='cfg-path',
        collection_name='col'
    )

    # Assert
    assert result['pre_total_citations'] == 4
    assert result['post_total_citations'] == 0
    assert result['validation_passed'] is False
    assert result['lost_entities'] == lost_df.to_dict()
    assert result['total_chunks'] == 1


def test_save_chunks_to_duckdb_failure(monkeypatch):
    # Arrange: force helper to raise inside save
    class DummyHelper:
        def store_chunks_batch(self, chunks):
            raise Exception('db error')
        def close(self):
            pass
    monkeypatch.setattr(service, 'get_duckdb_helper', lambda db: DummyHelper())

    # Act & Assert
    assert service.save_chunks_to_duckdb([], db_path='db') is False


def test_run_pipeline_chromadb_failure(monkeypatch):
    # Arrange: three dummy docs and citations
    docs = [Document(doi=f'doc{i}', has_dataset_citation=True, full_text=['x'], total_char_length=1,
                     parsed_timestamp='now', file_hash='h', file_path='p', n_pages=1) for i in range(1)]
    cites = [CitationEntity(data_citation='CID', document_id='doc0', pages=[1])]
    monkeypatch.setattr(service, 'load_input_data', lambda dp, cp: (docs, cites))

    # prepare_document: return one valid chunk and validation passed
    def fake_prepare(doc, citations, chunk_size, chunk_overlap, cfg_path, collection_name):
        meta = ChunkMetadata(chunk_id='c1', token_count=1, citation_entities=[citations[0]])
        ck = Chunk(chunk_id='c1', document_id=doc.doi, text='t', chunk_metadata=meta)
        return {
            'document': doc,
            'chunks': [ck],
            'pre_total_citations': 1,
            'post_total_citations': 1,
            'validation_passed': True,
            'lost_entities': None,
            'total_chunks': 1,
            'total_tokens': 1,
            'avg_tokens': 1.0,
            'pipeline_started_at': 'start',
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'cfg_path': cfg_path,
            'collection_name': collection_name
        }
    monkeypatch.setattr(service, 'prepare_document', fake_prepare)
    # DuckDB succeeds, Chroma fails in batch persistence
    monkeypatch.setattr(service, 'save_chunks_to_duckdb', lambda chunks, db_path: True)
    monkeypatch.setattr(service, 'save_chunk_objs_to_chroma', lambda chunks, collection_name, cfg_path: (_ for _ in ()).throw(Exception('chroma error')))

    # Act & Assert: batch stage should raise
    with pytest.raises(Exception, match='chroma error'):
        service.run_semantic_chunking_pipeline(
            documents_path='dp',
            citation_entities_path='cp',
            use_duckdb=False,
            max_workers=1
        )


def test_run_pipeline_integration(monkeypatch):
    # Arrange: three dummy docs
    docs = [SimpleNamespace(doi=f'doc{i}') for i in range(3)]
    monkeypatch.setattr(service, 'load_input_data', lambda dp, cp: (docs, []))

    # prepare_document: fail for doc1
    def fake_prepare(doc, cites, chunk_size, chunk_overlap, cfg_path, collection_name):
        if doc.doi == 'doc1':
            raise ValueError('fail prepare')
        return {
            'document': doc,
            'chunks': [],
            'pre_total_citations': 1,
            'post_total_citations': 1,
            'validation_passed': True,
            'lost_entities': None,
            'total_chunks': 0,
            'total_tokens': 0,
            'avg_tokens': 0,
            'pipeline_started_at': 'start',
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'cfg_path': cfg_path,
            'collection_name': collection_name
        }
    monkeypatch.setattr(service, 'prepare_document', fake_prepare)

    # commit_document: mark successes
    monkeypatch.setattr(service, 'commit_document', lambda prep, db: DocumentChunkingResult(
        document_id=prep['document'].doi,
        success=True,
        error=None,
        chunk_size=prep['chunk_size'],
        chunk_overlap=prep['chunk_overlap'],
        cfg_path=prep['cfg_path'],
        collection_name=prep['collection_name'],
        pre_chunk_total_citations=prep['pre_total_citations'],
        post_chunk_total_citations=prep['post_total_citations'],
        validation_passed=prep['validation_passed'],
        entity_retention=100.0,
        lost_entities=None,
        total_chunks=prep['total_chunks'],
        total_tokens=prep['total_tokens'],
        avg_tokens_per_chunk=prep['avg_tokens'],
        pipeline_started_at=prep['pipeline_started_at'],
        pipeline_completed_at='end'
    ))

    # Act
    result = service.run_semantic_chunking_pipeline(
        documents_path='dp',
        citation_entities_path='cp',
        use_duckdb=False,
        max_workers=2
    )

    # Assert
    assert isinstance(result, ChunkingResult)
    assert result.total_documents == 3
    assert result.success is False
    assert 'fail prepare' in result.error 