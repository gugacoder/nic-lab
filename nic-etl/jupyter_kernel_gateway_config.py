# Configuration file for jupyter-kernel-gateway
c = get_config()

# Server configuration
c.KernelGatewayApp.ip = '0.0.0.0'
c.KernelGatewayApp.port = 5000
c.KernelGatewayApp.port_retries = 0

# API configuration
c.KernelGatewayApp.api = 'notebook-http'
c.KernelGatewayApp.seed_uri = 'rest-api.ipynb'

# CORS
c.KernelGatewayApp.allow_origin = '*'
c.KernelGatewayApp.allow_headers = 'Content-Type, Authorization'
c.KernelGatewayApp.allow_methods = 'GET, POST, PUT, DELETE, OPTIONS'
c.KernelGatewayApp.allow_credentials = 'True'

# Kernel
c.KernelGatewayApp.default_kernel_name = 'python3'

# Logging
c.Application.log_level = 'INFO'

# Capturar outputs dos notebooks
c.KernelGatewayApp.force_kernel_name = 'python3'
c.MappingKernelManager.cull_idle_timeout = 0  # Não matar kernels ociosos

# Logging mais detalhado para debug
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Security
c.KernelGatewayApp.auth_token = ''

# Env vars
c.KernelGatewayApp.env_process_whitelist = [
    'GITLAB_URL','GITLAB_ACCESS_TOKEN','GITLAB_BRANCH','GITLAB_TARGET_FOLDER',
    'QDRANT_URL','QDRANT_API_KEY','QDRANT_COLLECTION','EMBEDDING_MODEL',
    'CHUNK_SIZE','CHUNK_OVERLAP','BATCH_SIZE','ENVIRONMENT'
]

# Dynamic prints (mostram os valores atuais definidos acima)
print("Jupyter Kernel Gateway configuration loaded successfully")
print(f"Server will run on: http://{c.KernelGatewayApp.ip}:{c.KernelGatewayApp.port}")
print(f"CORS allow_origin: {c.KernelGatewayApp.allow_origin}")
print(f"Allowed headers: {c.KernelGatewayApp.allow_headers}")
print(f"Allowed methods: {c.KernelGatewayApp.allow_methods}")
print(f"Environment variables whitelisted: {c.KernelGatewayApp.env_process_whitelist}")
