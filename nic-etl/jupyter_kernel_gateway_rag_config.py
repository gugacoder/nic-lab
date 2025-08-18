"""
Configuração do Jupyter Kernel Gateway para RAG API
Porta 5002 - Endpoints de busca semântica e RAG
"""

# Configurações básicas
c.KernelGatewayApp.ip = '127.0.0.1'
c.KernelGatewayApp.port = 5002
c.KernelGatewayApp.allow_credentials = '*'
c.KernelGatewayApp.allow_headers = '*'
c.KernelGatewayApp.allow_methods = '*'
c.KernelGatewayApp.allow_origin = '*'

# Timeout e configurações de performance
c.MappingKernelManager.cull_idle_timeout = 300
c.MappingKernelManager.cull_connected = False
c.MappingKernelManager.cull_interval = 60

# Configurações de logging
c.Application.log_level = 'INFO'

# Configurações específicas para RAG API
c.KernelGatewayApp.api = 'notebook-http'
c.KernelGatewayApp.seed_uri = 'rag-api.ipynb'

# Kernel específico para Python
c.KernelGatewayApp.default_kernel_name = 'python3'

# Configurações de segurança
c.KernelGatewayApp.auth_token = ''
c.KernelGatewayApp.certfile = ''
c.KernelGatewayApp.keyfile = ''

# Headers CORS específicos para API
c.KernelGatewayApp.allow_headers = 'Content-Type,Authorization,X-Requested-With,Accept,Origin,Access-Control-Request-Method,Access-Control-Request-Headers'

# Configurações adicionais para robustez
c.NotebookHTTPPersonality.allow_notebook_download = False
c.NotebookHTTPPersonality.comment_mapping = {
    'text/plain': '#',
    'application/json': '//',
}

print("🔧 Configuração RAG API carregada - Porta 5002")