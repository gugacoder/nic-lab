import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add modules to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "modules"))

from metadata_management import (
    NICSchemaManager, MetadataStatus, ProcessingStage, 
    EnrichmentContext, create_nic_schema_manager
)

class TestNICSchemaManager:
    
    @pytest.fixture
    def schema_manager(self):
        return create_nic_schema_manager()
    
    @pytest.fixture
    def valid_metadata(self):
        return {
            'document_id': 'test_doc_001',
            'title': 'Test Document',
            'description': 'A test document for validation',
            'file_path': '/path/to/test.pdf',
            'file_name': 'test.pdf',
            'status': MetadataStatus.APPROVED.value,
            'created_date': datetime.utcnow(),
            'author': 'Test Author',
            'organization': 'NIC',
            'processing_timestamp': datetime.utcnow(),
            'document_type': 'policy',
            'category': 'technical',
            'language': 'pt-BR',
            'version': '1.0',
            'gitlab_project': 'nic/docs',
            'gitlab_branch': 'main',
            'approver': 'Test Approver'  # Required for approved documents
        }
    
    def test_valid_metadata_validation(self, schema_manager, valid_metadata):
        """Test validation of valid metadata"""
        result = schema_manager.validate_document_metadata(valid_metadata)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.missing_fields) == 0
    
    def test_missing_required_fields(self, schema_manager):
        """Test validation with missing required fields"""
        incomplete_metadata = {
            'title': 'Test Document'
        }
        
        result = schema_manager.validate_document_metadata(incomplete_metadata)
        
        assert result.is_valid is False
        assert len(result.missing_fields) > 0
        assert 'document_id' in result.missing_fields
    
    def test_invalid_field_types(self, schema_manager, valid_metadata):
        """Test validation with invalid field types"""
        invalid_metadata = valid_metadata.copy()
        invalid_metadata['page_count'] = "not_a_number"
        
        result = schema_manager.validate_document_metadata(invalid_metadata)
        
        # Should attempt type conversion or report error
        has_conversion_warning = any('page_count' in warning for warning in result.warnings)
        has_type_error = any('page_count' in error for error in result.errors)
        
        assert has_conversion_warning or has_type_error
    
    def test_business_rule_validation(self, schema_manager, valid_metadata):
        """Test business rule validation"""
        # Test approved document without approver
        invalid_metadata = valid_metadata.copy()
        invalid_metadata['status'] = MetadataStatus.APPROVED.value
        del invalid_metadata['approver']
        
        result = schema_manager.validate_document_metadata(invalid_metadata)
        
        assert len(result.business_rule_violations) > 0 or len(result.errors) > 0
    
    def test_metadata_enrichment(self, schema_manager, valid_metadata):
        """Test metadata enrichment functionality"""
        
        context = EnrichmentContext(
            processing_stage=ProcessingStage.PROCESSING,
            source_metadata={'ocr_applied': True},
            processing_results={'quality_score': 0.85, 'page_count': 5},
            pipeline_config={'version': '1.0'}
        )
        
        enriched = schema_manager.enrich_metadata(valid_metadata, context)
        
        assert 'processing_timestamp' in enriched
        assert enriched['quality_score'] == 0.85
        assert enriched['page_count'] == 5
    
    def test_search_facet_extraction(self, schema_manager, valid_metadata):
        """Test search facet extraction"""
        facets = schema_manager.extract_search_facets(valid_metadata)
        
        assert 'status' in facets
        assert 'document_type' in facets
        assert 'author' in facets
        assert facets['status'] == [MetadataStatus.APPROVED.value]
        assert facets['document_type'] == ['policy']
    
    def test_derived_fields_computation(self, schema_manager):
        """Test computation of derived fields"""
        base_metadata = {
            'file_path': '/test/path/doc.pdf',
            'gitlab_commit': 'abc123',
            'processing_timestamp': datetime.utcnow().isoformat(),
            'total_tokens': 1000,
            'page_count': 10,
            'total_chunks': 5
        }
        
        derived = schema_manager._compute_derived_fields(base_metadata)
        
        assert 'document_id' in derived
        assert 'tokens_per_page' in derived
        assert 'avg_tokens_per_chunk' in derived
        assert 'complexity_score' in derived
        
        assert derived['tokens_per_page'] == 100.0  # 1000/10
        assert derived['avg_tokens_per_chunk'] == 200.0  # 1000/5
    
    def test_lineage_record_creation(self, schema_manager):
        """Test lineage record creation"""
        input_metadata = {'stage': 'input'}
        output_metadata = {'stage': 'output', 'processed': True}
        processing_params = {'model': 'test-model'}
        performance_metrics = {'time': 1.5, 'memory': 100}
        
        lineage = schema_manager.create_lineage_record(
            ProcessingStage.PROCESSING,
            input_metadata,
            output_metadata,
            processing_params,
            performance_metrics
        )
        
        assert lineage.stage == ProcessingStage.PROCESSING
        assert lineage.input_metadata == input_metadata
        assert lineage.output_metadata == output_metadata
        assert lineage.processing_parameters == processing_params
        assert lineage.performance_metrics == performance_metrics
        assert len(schema_manager.lineage_records) == 1
    
    def test_metadata_merge(self, schema_manager):
        """Test metadata merging from multiple sources"""
        sources = [
            {'title': 'Default Title', 'author': 'Default Author', '_source_type': 'default'},
            {'title': 'Extracted Title', 'category': 'extracted_cat', '_source_type': 'extracted'},
            {'title': 'User Title', '_source_type': 'user_provided'}
        ]
        
        merged = schema_manager.merge_metadata(sources)
        
        # User provided should have highest priority
        assert merged['title'] == 'User Title'
        # Other fields should come from appropriate sources
        assert merged['author'] == 'Default Author'
        assert merged['category'] == 'extracted_cat'
    
    def test_quality_facets(self, schema_manager):
        """Test quality tier facet generation"""
        high_quality = {'quality_score': 0.9}
        medium_quality = {'quality_score': 0.7}
        low_quality = {'quality_score': 0.4}
        
        high_facets = schema_manager.extract_search_facets(high_quality)
        medium_facets = schema_manager.extract_search_facets(medium_quality)
        low_facets = schema_manager.extract_search_facets(low_quality)
        
        assert high_facets['quality_tier'] == ['high']
        assert medium_facets['quality_tier'] == ['medium']
        assert low_facets['quality_tier'] == ['low']
    
    def test_schema_documentation_export(self, schema_manager):
        """Test schema documentation export"""
        documentation = schema_manager.export_schema_documentation()
        
        assert 'schema_version' in documentation
        assert 'fields' in documentation
        assert 'validation_rules' in documentation
        assert 'business_rules' in documentation
        
        # Check that key fields are documented
        assert 'document_id' in documentation['fields']
        assert 'title' in documentation['fields']
        assert 'status' in documentation['fields']
        
        # Check validation rules
        assert 'required_fields' in documentation['validation_rules']
        assert 'field_types' in documentation['validation_rules']
        assert 'valid_values' in documentation['validation_rules']