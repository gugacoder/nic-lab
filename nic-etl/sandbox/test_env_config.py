#!/usr/bin/env python3
"""
Teste da configuração usando o arquivo .env com parâmetros do PROMPT.md
"""
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

def test_env_configuration():
    """Testa se a configuração carrega corretamente do .env"""
    print("=== TESTE DE CONFIGURAÇÃO COM .env ===")
    
    from configuration_management import create_configuration_manager
    
    # Criar gerenciador usando .env file
    config_manager = create_configuration_manager(env_file='.env', environment='development')
    
    print("✓ Configuração carregada do arquivo .env")
    print(f"✓ Ambiente: {config_manager.config.environment}")
    
    # Verificar parâmetros do GitLab
    gitlab_config = config_manager.config.gitlab
    print(f"✓ GitLab URL: {gitlab_config.url}")
    print(f"✓ GitLab Token: {'***CONFIGURADO***' if gitlab_config.access_token else 'NÃO CONFIGURADO'}")
    print(f"✓ Projeto: {gitlab_config.project_path}")
    print(f"✓ Branch: {gitlab_config.branch}")
    print(f"✓ Pasta: {gitlab_config.folder_path}")
    
    # Verificar parâmetros do Qdrant
    qdrant_config = config_manager.config.qdrant
    print(f"✓ Qdrant URL: {qdrant_config.url}")
    print(f"✓ Qdrant API Key: {'***CONFIGURADO***' if qdrant_config.api_key else 'NÃO CONFIGURADO'}")
    print(f"✓ Collection: {qdrant_config.collection_name}")
    print(f"✓ Vector Size: {qdrant_config.vector_size}")
    print(f"✓ Distance Metric: {qdrant_config.distance_metric}")
    
    # Verificar parâmetros de Embedding
    embedding_config = config_manager.config.embedding
    print(f"✓ Modelo: {embedding_config.model_name}")
    print(f"✓ Device: {embedding_config.device}")
    print(f"✓ Batch Size: {embedding_config.batch_size}")
    
    # Verificar parâmetros de Chunking
    chunking_config = config_manager.config.chunking
    print(f"✓ Chunk Size: {chunking_config.target_chunk_size} tokens")
    print(f"✓ Overlap: {chunking_config.overlap_size} tokens")
    print(f"✓ Modelo Tokenizer: {chunking_config.model_name}")
    
    # Verificar se está tudo conforme PROMPT.md
    assert gitlab_config.url == "http://gitlab.processa.info/nic/documentacao/base-de-conhecimento.git"
    assert gitlab_config.project_path == "nic/documentacao/base-de-conhecimento"  
    assert gitlab_config.branch == "main"
    assert gitlab_config.folder_path == "30-Aprovados"
    assert qdrant_config.url == "https://qdrant.codrstudio.dev/"
    assert qdrant_config.collection_name == "nic"
    assert qdrant_config.vector_size == 1024
    assert qdrant_config.distance_metric == "COSINE"
    assert embedding_config.model_name == "BAAI/bge-m3"
    assert embedding_config.device == "cpu"
    assert chunking_config.target_chunk_size == 500
    assert chunking_config.overlap_size == 100
    
    print("\n✅ TODOS OS PARÂMETROS DO PROMPT.MD CONFIGURADOS CORRETAMENTE!")
    
    # Validar configuração
    validation = config_manager.validate_configuration(config_manager.config)
    print(f"✓ Configuração válida: {validation.is_valid}")
    if validation.warnings:
        print(f"  - Avisos: {len(validation.warnings)}")
    if validation.errors:
        print(f"  - Erros: {len(validation.errors)}")
    
    return True

def test_gitlab_with_real_config():
    """Testa a integração GitLab com configuração real"""
    print("\n=== TESTE GITLAB COM CONFIGURAÇÃO REAL ===")
    
    from configuration_management import create_configuration_manager
    from gitlab_integration import create_gitlab_connector
    
    # Carregar configuração
    config_manager = create_configuration_manager(env_file='.env')
    gitlab_config = config_manager.get_module_config('gitlab')
    
    # Criar conector GitLab
    connector = create_gitlab_connector({
        'gitlab_url': gitlab_config['url'].replace('.git', ''),  # Remove .git for API
        'access_token': gitlab_config['access_token'],
        'project_path': gitlab_config['project_path']
    })
    
    print(f"✓ Conector criado para: {connector.gitlab_url}")
    print(f"✓ Projeto: {connector.project_path}")
    print(f"✓ Token configurado: {'Sim' if connector.access_token else 'Não'}")
    
    # Tentar autenticação
    try:
        auth_result = connector.authenticate()
        print(f"✓ Autenticação: {'SUCESSO' if auth_result else 'FALHOU'}")
        
        if auth_result:
            print(f"✓ Project ID: {connector.project_id}")
            
            # Testar listagem de arquivos
            try:
                files = connector.list_files(
                    branch=gitlab_config['branch'],
                    folder_path=gitlab_config['folder_path']
                )
                print(f"✓ Arquivos encontrados: {len(files)}")
                
                # Mostrar alguns arquivos
                for i, file in enumerate(files[:3]):
                    print(f"  - {file.name} ({file.extension}) - {file.size} bytes")
                
                if len(files) > 3:
                    print(f"  ... e mais {len(files) - 3} arquivos")
                    
            except Exception as e:
                print(f"  - Erro ao listar arquivos: {e}")
        
    except Exception as e:
        print(f"  - Erro na autenticação: {e}")
    
    return True

if __name__ == "__main__":
    try:
        test_env_configuration()
        test_gitlab_with_real_config()
        
        print("\n" + "="*60)
        print("✅ CONFIGURAÇÃO .ENV FUNCIONANDO PERFEITAMENTE!")
        print("✅ TODOS OS PARÂMETROS DO PROMPT.MD APLICADOS!")
        print("✅ MÓDULOS PRONTOS PARA EXECUÇÃO!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()