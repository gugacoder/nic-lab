# Configuration file for jupyter-kernel-gateway

# Application configuration
c = get_config()

# Server configuration
c.KernelGatewayApp.ip = '0.0.0.0'  # Listen on all interfaces
c.KernelGatewayApp.port = 8000
c.KernelGatewayApp.port_retries = 0

# API configuration
c.KernelGatewayApp.api = 'kernel_gateway.notebook_http'

# Seed URI configuration - will be overridden by command line
c.KernelGatewayApp.seed_uri = './notebooks/nic-etl-api.ipynb'

# CORS configuration for web access
c.KernelGatewayApp.allow_origin = '*'
c.KernelGatewayApp.allow_headers = 'Content-Type'
c.KernelGatewayApp.allow_methods = 'GET,POST,PUT,DELETE,OPTIONS'
c.KernelGatewayApp.allow_credentials = 'true'

# Kernel configuration
c.KernelGatewayApp.default_kernel_name = 'python3'

# Logging configuration
c.Application.log_level = 'INFO'

# Security - disable token authentication for development
c.KernelGatewayApp.auth_token = ''

# Pool configuration
c.KernelGatewayApp.kernel_pool_size = 5
c.KernelGatewayApp.max_kernels = 10

# Request timeout (in seconds)
c.KernelGatewayApp.request_timeout = 60

# Working directory
c.KernelGatewayApp.notebook_dir = './notebooks'

# Environment variables to pass to kernels
c.KernelGatewayApp.env_whitelist = [
    'GITLAB_URL',
    'GITLAB_ACCESS_TOKEN', 
    'GITLAB_BRANCH',
    'GITLAB_TARGET_FOLDER',
    'QDRANT_URL',
    'QDRANT_API_KEY',
    'QDRANT_COLLECTION',
    'EMBEDDING_MODEL',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP',
    'BATCH_SIZE',
    'ENVIRONMENT'
]

# Enable debug mode for development
c.KernelGatewayApp.debug = True

# Request/Response size limits
c.KernelGatewayApp.max_buffer_size = 1024 * 1024 * 100  # 100MB

print("Jupyter Kernel Gateway configuration loaded successfully")
print("Server will run on: http://0.0.0.0:8000")
print("CORS enabled for all origins")
print("Environment variables whitelisted for kernels")