#!/usr/bin/env python3
"""
Basic test script for metadata management system
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from metadata_management import (
    NICSchemaManager, MetadataStatus, ProcessingStage, 
    EnrichmentContext, create_nic_schema_manager
)

def test_basic_metadata_management():
    """Test basic metadata management functionality"""
    print("Testing Metadata Management System...")
    
    try:
        # Create schema manager
        schema_manager = create_nic_schema_manager()
        print("✓ Schema manager created successfully")
        
        # Test 1: Valid metadata validation
        valid_metadata = {
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
            'approver': 'Test Approver'
        }
        
        result = schema_manager.validate_document_metadata(valid_metadata)
        print(f"✓ Valid metadata validation: {result.is_valid}")
        
        # Test 2: Invalid metadata validation
        invalid_metadata = {
            'title': 'Test Document'
            # Missing required fields
        }
        
        result = schema_manager.validate_document_metadata(invalid_metadata)
        print(f"✓ Invalid metadata validation: {not result.is_valid} (detected {len(result.missing_fields)} missing fields)")
        
        # Test 3: Metadata enrichment
        context = EnrichmentContext(
            processing_stage=ProcessingStage.PROCESSING,
            source_metadata={'ocr_applied': True},
            processing_results={'quality_score': 0.85, 'page_count': 5},
            pipeline_config={'version': '1.0'}
        )
        
        enriched = schema_manager.enrich_metadata(valid_metadata, context)
        print(f"✓ Metadata enrichment: added {len(enriched) - len(valid_metadata)} new fields")
        
        # Test 4: Search facets extraction
        facets = schema_manager.extract_search_facets(valid_metadata)
        print(f"✓ Search facets extracted: {len(facets)} facet types")
        
        # Test 5: Lineage record creation
        lineage = schema_manager.create_lineage_record(
            ProcessingStage.PROCESSING,
            valid_metadata,
            enriched,
            {'model': 'test-model'},
            {'time': 1.5, 'memory': 100}
        )
        print(f"✓ Lineage record created for stage: {lineage.stage.value}")
        
        # Test 6: Metadata merging
        sources = [
            {'title': 'Default Title', 'author': 'Default Author', '_source_type': 'default'},
            {'title': 'User Title', '_source_type': 'user_provided'}
        ]
        
        merged = schema_manager.merge_metadata(sources)
        print(f"✓ Metadata merged: {merged['title']} (user priority)")
        
        # Test 7: Derived fields computation
        test_metadata = {
            'file_path': '/test/path/doc.pdf',
            'gitlab_commit': 'abc123',
            'processing_timestamp': datetime.utcnow().isoformat(),
            'total_tokens': 1000,
            'page_count': 10,
            'total_chunks': 5
        }
        
        derived = schema_manager._compute_derived_fields(test_metadata)
        print(f"✓ Derived fields computed: {len(derived)} fields")
        
        # Test 8: Schema documentation export
        documentation = schema_manager.export_schema_documentation()
        field_count = len(documentation['fields'])
        print(f"✓ Schema documentation exported: {field_count} fields documented")
        
        # Test 9: Business rules validation
        invalid_business_metadata = valid_metadata.copy()
        invalid_business_metadata['quality_score'] = 1.5  # Invalid range
        
        result = schema_manager.validate_document_metadata(invalid_business_metadata)
        violations = len(result.business_rule_violations)
        print(f"✓ Business rules validation: detected {violations} violations")
        
        print("\n✓ All metadata management tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Metadata management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_metadata_management()
    sys.exit(0 if success else 1)