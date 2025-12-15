"""
Tests for pipeline_orchestrator use case.
"""

import pytest
from src.core.ingest_use_cases.pipeline_orchestrator import PipelineOrchestrator
from src.core.ingest_use_cases.ingest_messages import IngestMessages
from src.core.ingest_use_cases.process_chunks import ProcessChunks
from src.core.ingest_use_cases.generate_embeddings import GenerateEmbeddings
from src.core.ingest_use_cases.sync_to_vector_store import SyncToVectorStore


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator class."""
    
    def test_init(self):
        """Test initialization."""
        class MockIngestMessages:
            pass
        
        class MockProcessChunks:
            pass
        
        class MockGenerateEmbeddings:
            pass
        
        class MockSyncToVectorStore:
            pass
        
        orchestrator = PipelineOrchestrator(
            ingest_messages=MockIngestMessages(),
            process_chunks=MockProcessChunks(),
            generate_embeddings=MockGenerateEmbeddings(),
            sync_to_vector_store=MockSyncToVectorStore()
        )
        assert orchestrator is not None
    
    def test_run_all(self, tmp_path):
        """Test run_all method."""
        class MockIngestMessages:
            def execute(self, file_path):
                return 10
        
        class MockProcessChunks:
            def execute(self):
                return 5
        
        class MockGenerateEmbeddings:
            pass
        
        class MockSyncToVectorStore:
            pass
        
        orchestrator = PipelineOrchestrator(
            ingest_messages=MockIngestMessages(),
            process_chunks=MockProcessChunks(),
            generate_embeddings=MockGenerateEmbeddings(),
            sync_to_vector_store=MockSyncToVectorStore()
        )
        
        # Create a dummy file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        stats = orchestrator.run_all(str(test_file))
        assert "stage0_messages_saved" in stats
        assert "stage1_chunks_saved" in stats

