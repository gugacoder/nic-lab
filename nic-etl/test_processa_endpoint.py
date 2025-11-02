#!/usr/bin/env python3
"""
Teste do endpoint /processa/info
"""
import json
import sys
import os
from pathlib import Path

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

# Importar módulos necessários
import nbformat
from nbconvert import PythonExporter

def test_processa_info():
    """Testa o endpoint de informações da Processa"""
    
    print("🧪 Testando endpoint /processa/info\n")
    print("=" * 60)
    
    # Carregar notebook
    notebook_path = Path("notebooks/rag-processa-info.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ Notebook não encontrado: {notebook_path}")
        return
    
    try:
        # Converter notebook para Python
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Exportar para Python
        exporter = PythonExporter()
        source, _ = exporter.from_notebook_node(nb)
        
        # Criar módulo temporário
        temp_module = type(sys)('rag_processa_info')
        exec(source, temp_module.__dict__)
        
        # Importar função
        get_processa_info_api = temp_module.get_processa_info_api
        
        # Teste 1: Query padrão
        print("\n📍 Teste 1: Query padrão")
        print("Query: 'Quem é a Processa Sistemas'")
        result = get_processa_info_api()
        
        print(f"\nStatus: {result.get('status')}")
        
        if result['status'] == 'success':
            print("✅ Busca realizada com sucesso!")
            company_info = result.get('company_info', {})
            
            # Mostrar descrição
            desc = company_info.get('description', 'N/A')
            if desc and len(desc) > 200:
                desc = desc[:200] + "..."
            print(f"\n📝 Descrição: {desc}")
            
            # Mostrar pontos principais
            key_points = company_info.get('key_points', [])
            if key_points:
                print(f"\n🎯 Pontos principais ({len(key_points)}):")
                for i, point in enumerate(key_points[:3], 1):
                    print(f"  {i}. {point[:100]}...")
            
            # Mostrar fontes
            sources = company_info.get('sources', [])
            if sources:
                print(f"\n📚 Fontes ({len(sources)}):")
                for source in sources[:3]:
                    doc = source.get('document', 'unknown')
                    score = source.get('relevance_score', 0)
                    print(f"  - {doc} (score: {score:.3f})")
            
            # Metadata
            search_meta = result.get('search_metadata', {})
            print(f"\n📊 Metadados da busca:")
            print(f"  - Total de resultados: {search_meta.get('total_results', 0)}")
            print(f"  - Score máximo: {search_meta.get('top_score', 0):.3f}")
            print(f"  - Collection: {search_meta.get('collection', 'N/A')}")
            print(f"  - Modelo: {search_meta.get('model', 'N/A')}")
            
        elif result['status'] == 'no_results':
            print("⚠️ Nenhum resultado encontrado")
            print(f"Mensagem: {result.get('message', 'N/A')}")
        else:
            print(f"❌ Erro: {result.get('message', 'Desconhecido')}")
        
        # Teste 2: Query customizada
        print("\n" + "=" * 60)
        print("\n📍 Teste 2: Query customizada")
        print("Query: 'serviços oferecidos pela Processa'")
        
        custom_result = get_processa_info_api("serviços oferecidos pela Processa")
        
        print(f"\nStatus: {custom_result.get('status')}")
        
        if custom_result['status'] == 'success':
            print("✅ Busca customizada realizada!")
            
            company_info = custom_result.get('company_info', {})
            desc = company_info.get('description', 'N/A')
            if desc and len(desc) > 200:
                desc = desc[:200] + "..."
            print(f"\n📝 Descrição: {desc}")
            
            search_meta = custom_result.get('search_metadata', {})
            print(f"\n📊 Total de resultados: {search_meta.get('total_results', 0)}")
            
        print("\n" + "=" * 60)
        print("\n✨ Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao testar endpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_processa_info()