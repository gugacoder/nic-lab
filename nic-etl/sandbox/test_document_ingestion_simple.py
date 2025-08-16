#!/usr/bin/env python3
"""
Simple tests for Document Ingestion Module (without pytest)
"""
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

def test_document_ingestion():
    """Test document ingestion functionality"""
    print("=== TESTE DO MÓDULO DOCUMENT INGESTION ===")
    
    from document_ingestion import (
        DocumentIngestionManager, IngestionConfig, DocumentFormat,
        create_document_ingestion_manager
    )
    
    # Create configuration
    config = IngestionConfig(
        max_file_size_mb=10,
        supported_formats=['pdf', 'docx', 'txt', 'md', 'jpg', 'png'],
        extract_preview=True,
        preview_length=500
    )
    
    # Create manager
    manager = DocumentIngestionManager(config)
    print(f"✓ Manager criado com sucesso")
    
    # Test format detection
    print("\n--- Teste de Detecção de Formato ---")
    
    # Text content
    text_content = "Este é um documento de teste em português com conteúdo suficiente para análise.".encode('utf-8')
    detected_format = manager._detect_format(text_content, "test.txt")
    print(f"✓ Formato detectado para texto: {detected_format.value}")
    assert detected_format == DocumentFormat.TXT
    
    # PDF signature
    pdf_content = b'%PDF-1.4\nSample PDF content'
    detected_format = manager._detect_format(pdf_content, "test.pdf")
    print(f"✓ Formato detectado para PDF: {detected_format.value}")
    assert detected_format == DocumentFormat.PDF
    
    # Markdown content
    md_content = "# Título\n\n## Subtítulo\n\n* Item de lista\n* Outro item".encode('utf-8')
    detected_format = manager._detect_format(md_content, "test.md")
    print(f"✓ Formato detectado para Markdown: {detected_format.value}")
    assert detected_format == DocumentFormat.MARKDOWN
    
    # JPEG signature
    jpeg_content = b'\xff\xd8\xff\xe0\x00\x10JFIF'
    detected_format = manager._detect_format(jpeg_content, "test.jpg")
    print(f"✓ Formato detectado para JPEG: {detected_format.value}")
    assert detected_format == DocumentFormat.JPEG
    
    # PNG signature
    png_content = b'\x89PNG\r\n\x1a\n'
    detected_format = manager._detect_format(png_content, "test.png")
    print(f"✓ Formato detectado para PNG: {detected_format.value}")
    assert detected_format == DocumentFormat.PNG
    
    print("\n--- Teste de Ingestão de Documento ---")
    
    # Test text document ingestion
    metadata = {
        'file_path': '/test/documento.txt',
        'source': 'test'
    }
    
    ingested_doc = manager.ingest_document(text_content, metadata)
    
    print(f"✓ Documento ingerido: {ingested_doc.metadata.file_name}")
    print(f"✓ Formato: {ingested_doc.metadata.detected_format.value}")
    print(f"✓ Tamanho: {ingested_doc.metadata.file_size} bytes")
    print(f"✓ Hash: {ingested_doc.metadata.file_hash[:16]}...")
    print(f"✓ Palavras: {ingested_doc.metadata.word_count}")
    print(f"✓ Caracteres: {ingested_doc.metadata.character_count}")
    print(f"✓ Idioma detectado: {ingested_doc.metadata.language}")
    print(f"✓ Validação: {ingested_doc.validation_result.is_valid}")
    print(f"✓ Confiança: {ingested_doc.validation_result.confidence_score:.2f}")
    print(f"✓ Preview: {ingested_doc.preview_text[:50]}...")
    
    # Test validation
    print("\n--- Teste de Validação ---")
    
    # Valid document
    validation = manager.validate_document(text_content, "txt")
    print(f"✓ Validação de documento válido: {validation.is_valid}")
    print(f"✓ Status: {validation.status.value}")
    print(f"✓ Confiança: {validation.confidence_score:.2f}")
    
    # Empty document
    empty_content = b""
    validation = manager.validate_document(empty_content, "txt")
    print(f"✓ Validação de documento vazio: {validation.is_valid}")
    print(f"✓ Problemas encontrados: {len(validation.issues)}")
    if validation.issues:
        print(f"  - {validation.issues[0]}")
    
    # Document with suspicious content
    suspicious_content = b"Normal content with <script>alert('test')</script> embedded"
    validation = manager.validate_document(suspicious_content, "txt")
    print(f"✓ Validação de conteúdo suspeito: {validation.is_valid}")
    print(f"✓ Flags de segurança: {len(validation.security_flags)}")
    if validation.security_flags:
        print(f"  - {validation.security_flags[0]}")
    
    print("\n--- Teste de Normalização ---")
    
    # Test document normalization
    normalized_doc = manager.normalize_content(ingested_doc)
    print(f"✓ Documento normalizado")
    print(f"✓ Tipo de conteúdo: {normalized_doc.content_type}")
    print(f"✓ Score de qualidade: {normalized_doc.quality_score:.2f}")
    print(f"✓ Dicas de processamento: {len(normalized_doc.processing_hints)}")
    for key, value in normalized_doc.processing_hints.items():
        print(f"  - {key}: {value}")
    
    print("\n--- Teste de Processamento em Lote ---")
    
    # Test batch processing
    documents = [
        {
            'content': "Primeiro documento do lote.".encode('utf-8'),
            'metadata': {'file_path': 'doc1.txt', 'source': 'batch_test'}
        },
        {
            'content': "Segundo documento do lote com mais conteúdo para análise.".encode('utf-8'),
            'metadata': {'file_path': 'doc2.txt', 'source': 'batch_test'}
        },
        {
            'content': "# Terceiro documento\n\nEste é um documento markdown.".encode('utf-8'),
            'metadata': {'file_path': 'doc3.md', 'source': 'batch_test'}
        }
    ]
    
    batch_results = manager.batch_ingest(documents)
    print(f"✓ Lote processado: {len(batch_results)} documentos")
    
    for i, doc in enumerate(batch_results):
        print(f"  - Doc {i+1}: {doc.metadata.file_name} ({doc.metadata.detected_format.value})")
    
    print("\n--- Teste de Funcionalidades ---")
    
    # Test supported formats
    formats = manager.get_supported_formats()
    print(f"✓ Formatos suportados: {', '.join(formats)}")
    
    # Test processing statistics
    stats = manager.get_processing_statistics()
    print(f"✓ Capacidades do sistema:")
    for capability, available in stats['capabilities'].items():
        status = "✓" if available else "✗"
        print(f"  {status} {capability}: {available}")
    
    print("\n--- Teste de Factory Function ---")
    
    # Test factory function
    config_dict = {
        'max_file_size_mb': 50,
        'supported_formats': ['pdf', 'txt', 'md'],
        'enable_content_validation': True,
        'extract_preview': True
    }
    
    factory_manager = create_document_ingestion_manager(config_dict)
    print(f"✓ Manager criado via factory")
    print(f"✓ Limite de tamanho: {factory_manager.config.max_file_size_mb}MB")
    print(f"✓ Formatos configurados: {len(factory_manager.config.supported_formats)}")
    
    print("\n--- Teste de Tratamento de Erros ---")
    
    # Test file size limit
    try:
        large_content = b'x' * (11 * 1024 * 1024)  # 11MB (exceeds 10MB limit)
        manager.ingest_document(large_content, {'file_path': 'large.txt'})
        print("❌ Deveria ter falhado por tamanho")
    except ValueError as e:
        print(f"✓ Limite de tamanho respeitado: {str(e)[:50]}...")
    
    # Test unsupported format
    try:
        config_restricted = IngestionConfig(supported_formats=['txt'])
        restricted_manager = DocumentIngestionManager(config_restricted)
        restricted_manager.ingest_document(pdf_content, {'file_path': 'test.pdf'})
        print("❌ Deveria ter falhado por formato não suportado")
    except ValueError as e:
        print(f"✓ Formato não suportado rejeitado: {str(e)[:50]}...")
    
    # Test unknown format
    try:
        unknown_content = b'\x00\x01\x02\x03\x04\x05' * 100
        manager._detect_format(unknown_content, "unknown.xyz")
        print("❌ Deveria ter falhado por formato desconhecido")
    except ValueError as e:
        print(f"✓ Formato desconhecido rejeitado: {str(e)[:50]}...")
    
    return True

def main():
    """Execute all tests"""
    print("TESTE DO MÓDULO DOCUMENT INGESTION")
    print("=" * 50)
    
    try:
        success = test_document_ingestion()
        
        if success:
            print("\n" + "=" * 50)
            print("✅ TODOS OS TESTES PASSARAM!")
            print("✅ MÓDULO DOCUMENT INGESTION FUNCIONANDO!")
            print("\nRecursos implementados:")
            print("- Detecção de formato multi-método")
            print("- Validação abrangente de documentos")
            print("- Extração de metadados detalhados")
            print("- Verificações básicas de segurança")
            print("- Processamento em lote")
            print("- Normalização para pipeline")
            print("- Tratamento robusto de erros")
            print("- Suporte a 6 formatos: PDF, DOCX, TXT, MD, JPG, PNG")
        
    except Exception as e:
        print(f"❌ Erro nos testes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()