#!/usr/bin/env python3
"""
Test connections for GitLab and Qdrant using real configuration from .env
"""
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

def test_gitlab_connection():
    """Test GitLab connection with real credentials"""
    print("=== TESTE DE CONEXÃO GITLAB ===")
    
    from configuration_management import create_configuration_manager
    from gitlab_integration import create_gitlab_connector
    
    try:
        # Load configuration from .env
        config_manager = create_configuration_manager(env_file='.env')
        gitlab_config = config_manager.get_module_config('gitlab')
        
        print(f"✓ GitLab URL: {gitlab_config['url']}")
        print(f"✓ Projeto: {gitlab_config['project_path']}")
        print(f"✓ Branch: {gitlab_config['branch']}")
        print(f"✓ Pasta: {gitlab_config['folder_path']}")
        
        # Create connector with real config
        connector = create_gitlab_connector({
            'gitlab_url': gitlab_config['url'].replace('.git', ''),  # Remove .git for API
            'access_token': gitlab_config['access_token'],
            'project_path': gitlab_config['project_path']
        })
        
        print(f"✓ Conector criado: {connector.gitlab_url}")
        
        # Test authentication
        print("⏳ Testando autenticação...")
        auth_result = connector.authenticate()
        
        if auth_result:
            print("✅ AUTENTICAÇÃO GITLAB: SUCESSO!")
            print(f"✓ Project ID: {connector.project_id}")
            
            # Test file listing
            print("⏳ Testando listagem de arquivos...")
            try:
                files = connector.list_files(
                    branch=gitlab_config['branch'],
                    folder_path=gitlab_config['folder_path']
                )
                
                print(f"✅ LISTAGEM DE ARQUIVOS: SUCESSO!")
                print(f"✓ Arquivos encontrados: {len(files)}")
                
                # Show first few files
                for i, file in enumerate(files[:5]):
                    print(f"  - {file.name} ({file.extension}) - {file.size} bytes")
                
                if len(files) > 5:
                    print(f"  ... e mais {len(files) - 5} arquivos")
                
                # Test downloading one file
                if files:
                    print("⏳ Testando download de arquivo...")
                    test_file = files[0]
                    try:
                        content = connector.download_file(
                            test_file.path,
                            branch=gitlab_config['branch']
                        )
                        print(f"✅ DOWNLOAD DE ARQUIVO: SUCESSO!")
                        print(f"✓ Arquivo baixado: {test_file.name} ({len(content)} bytes)")
                    except Exception as e:
                        print(f"❌ Erro no download: {e}")
                        
            except Exception as e:
                print(f"❌ Erro na listagem: {e}")
                
        else:
            print("❌ AUTENTICAÇÃO GITLAB: FALHOU!")
            
    except Exception as e:
        print(f"❌ Erro geral no GitLab: {e}")
        import traceback
        traceback.print_exc()

def test_qdrant_connection():
    """Test Qdrant connection with real credentials"""
    print("\n=== TESTE DE CONEXÃO QDRANT ===")
    
    from configuration_management import create_configuration_manager
    import json
    
    try:
        # Load configuration from .env
        config_manager = create_configuration_manager(env_file='.env')
        qdrant_config = config_manager.get_module_config('qdrant')
        
        print(f"✓ Qdrant URL: {qdrant_config['url']}")
        print(f"✓ Collection: {qdrant_config['collection_name']}")
        print(f"✓ Vector Size: {qdrant_config['vector_size']}")
        print(f"✓ Distance Metric: {qdrant_config['distance_metric']}")
        
        # Test connection using urllib (no external dependencies)
        import urllib.request
        import urllib.parse
        
        # Test basic connection
        print("⏳ Testando conexão básica...")
        
        # Create headers with API key
        headers = {
            'api-key': qdrant_config['api_key'],
            'Content-Type': 'application/json'
        }
        
        # Test cluster info endpoint
        cluster_url = f"{qdrant_config['url'].rstrip('/')}/cluster"
        
        request = urllib.request.Request(cluster_url, headers=headers)
        
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    print("✅ CONEXÃO QDRANT: SUCESSO!")
                    
                    cluster_info = json.loads(response.read().decode())
                    print(f"✓ Status: {cluster_info.get('status', 'unknown')}")
                    
                else:
                    print(f"❌ Status HTTP inesperado: {response.status}")
                    
        except urllib.error.HTTPError as e:
            print(f"❌ Erro HTTP: {e.code} - {e.reason}")
        except urllib.error.URLError as e:
            print(f"❌ Erro de URL: {e.reason}")
        except Exception as e:
            print(f"❌ Erro na conexão: {e}")
        
        # Test collections endpoint
        print("⏳ Testando endpoint de collections...")
        
        collections_url = f"{qdrant_config['url'].rstrip('/')}/collections"
        request = urllib.request.Request(collections_url, headers=headers)
        
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status == 200:
                    print("✅ ENDPOINT COLLECTIONS: SUCESSO!")
                    
                    collections_data = json.loads(response.read().decode())
                    collections = collections_data.get('result', {}).get('collections', [])
                    
                    print(f"✓ Collections encontradas: {len(collections)}")
                    
                    # Check if NIC collection exists
                    nic_collection_exists = False
                    for collection in collections:
                        collection_name = collection.get('name', '')
                        if collection_name == qdrant_config['collection_name']:
                            nic_collection_exists = True
                            print(f"✓ Collection '{qdrant_config['collection_name']}' encontrada!")
                            break
                    
                    if not nic_collection_exists:
                        print(f"⚠️  Collection '{qdrant_config['collection_name']}' não encontrada")
                        print("   (será criada automaticamente quando necessário)")
                    
                else:
                    print(f"❌ Status HTTP inesperado: {response.status}")
                    
        except Exception as e:
            print(f"❌ Erro no endpoint collections: {e}")
            
        # Test collection info if it exists
        if 'nic_collection_exists' in locals() and nic_collection_exists:
            print("⏳ Testando informações da collection...")
            
            collection_info_url = f"{qdrant_config['url'].rstrip('/')}/collections/{qdrant_config['collection_name']}"
            request = urllib.request.Request(collection_info_url, headers=headers)
            
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    if response.status == 200:
                        print("✅ INFORMAÇÕES DA COLLECTION: SUCESSO!")
                        
                        collection_info = json.loads(response.read().decode())
                        result = collection_info.get('result', {})
                        config = result.get('config', {})
                        
                        # Check vector configuration
                        params = config.get('params', {})
                        vectors_config = params.get('vectors', {})
                        
                        if isinstance(vectors_config, dict):
                            vector_size = vectors_config.get('size', 'unknown')
                            distance = vectors_config.get('distance', 'unknown')
                            
                            print(f"✓ Vector size configurado: {vector_size}")
                            print(f"✓ Distance metric configurado: {distance}")
                            
                            # Validate configuration
                            if vector_size == qdrant_config['vector_size']:
                                print("✓ Vector size está correto!")
                            else:
                                print(f"⚠️  Vector size incorreto. Esperado: {qdrant_config['vector_size']}, Atual: {vector_size}")
                            
                            if distance.upper() == qdrant_config['distance_metric'].upper():
                                print("✓ Distance metric está correto!")
                            else:
                                print(f"⚠️  Distance metric incorreto. Esperado: {qdrant_config['distance_metric']}, Atual: {distance}")
                        
                        # Check points count
                        points_count = result.get('points_count', 0)
                        print(f"✓ Pontos na collection: {points_count}")
                        
                    else:
                        print(f"❌ Status HTTP inesperado: {response.status}")
                        
            except Exception as e:
                print(f"❌ Erro nas informações da collection: {e}")
                
    except Exception as e:
        print(f"❌ Erro geral no Qdrant: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Execute all connection tests"""
    print("TESTE DE CONEXÕES - PIPELINE NIC ETL")
    print("=" * 50)
    
    test_gitlab_connection()
    test_qdrant_connection()
    
    print("\n" + "=" * 50)
    print("✅ TESTES DE CONEXÃO CONCLUÍDOS!")

if __name__ == "__main__":
    main()