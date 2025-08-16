import urllib.request
import urllib.parse
import urllib.error
import base64
import json
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import time
from pathlib import Path

@dataclass
class FileMetadata:
    """Complete file metadata from GitLab"""
    path: str
    name: str
    extension: str
    size: int
    commit_sha: str
    commit_date: datetime
    author: str
    gitlab_url: str
    raw_url: str

@dataclass
class CommitMetadata:
    """Git commit information for lineage tracking"""
    sha: str
    author: str
    date: datetime
    message: str
    is_latest: bool

class GitLabIntegrationError(Exception):
    """Base exception for GitLab integration errors"""
    pass

class AuthenticationError(GitLabIntegrationError):
    """Authentication failures"""
    pass

class AccessError(GitLabIntegrationError):
    """Repository or file access errors"""
    pass

class NetworkError(GitLabIntegrationError):
    """Network connectivity issues"""
    pass

class GitLabConnector:
    """Production-ready GitLab integration with comprehensive error handling"""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.jpg', '.jpeg', '.png'}
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    API_TIMEOUT = 30.0
    
    def __init__(self, gitlab_url: str, access_token: str, project_path: str):
        self.gitlab_url = gitlab_url.rstrip('/')
        self.access_token = access_token
        self.project_path = project_path
        self.project_id = None
        self.logger = logging.getLogger(__name__)
        
        # Build API base URL
        self.api_base = f"{self.gitlab_url}/api/v4"
        
    def authenticate(self) -> bool:
        """Establish authenticated GitLab connection with validation"""
        try:
            # First, validate token by getting user info
            user_url = f"{self.api_base}/user"
            status, data = self._make_request('GET', user_url)
            
            if status == 401:
                self.logger.error("GitLab authentication failed - invalid token")
                return False
            elif status != 200:
                self.logger.error(f"GitLab API error: {status}")
                return False
            
            # Get project information and validate access
            project_url = f"{self.api_base}/projects/{urllib.parse.quote(self.project_path, safe='')}"
            status, data = self._make_request('GET', project_url)
            
            if status == 404:
                self.logger.error(f"Project not found: {self.project_path}")
                return False
            elif status == 403:
                self.logger.error(f"Access denied to project: {self.project_path}")
                return False
            elif status != 200:
                self.logger.error(f"Project access failed: {status}")
                return False
            
            self.project_id = data['id']
            
            self.logger.info(f"Successfully authenticated to GitLab project: {self.project_path}")
            return True
            
        except urllib.error.URLError as e:
            self.logger.error(f"GitLab connection error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected authentication error: {e}")
            return False
    
    def list_files(self, branch: str = 'main', folder_path: str = '30-Aprovados', 
                   extensions: Optional[List[str]] = None) -> List[FileMetadata]:
        """List all files in target folder with metadata"""
        if not self.project_id:
            raise RuntimeError("GitLab connection not established - call authenticate() first")
            
        extensions = extensions or list(self.SUPPORTED_EXTENSIONS)
        ext_set = {ext.lower().lstrip('.') for ext in extensions}
        
        files = []
        try:
            # Get latest commit info for lineage
            commit_info = self.get_commit_info(branch)
            
            # Get repository tree for the target folder
            tree_url = f"{self.api_base}/projects/{self.project_id}/repository/tree"
            params = {
                'path': folder_path,
                'ref': branch,
                'recursive': 'true',
                'per_page': '100'
            }
            
            # Build URL with parameters
            query_string = urllib.parse.urlencode(params)
            full_url = f"{tree_url}?{query_string}"
            
            status, tree_items = self._make_request('GET', full_url)
            
            if status == 404:
                self.logger.warning(f"Folder not found: {folder_path}")
                return files
            elif status != 200:
                raise AccessError(f"Failed to access repository tree: {status}")
            
            for item in tree_items:
                if item['type'] == 'blob':  # File (not directory)
                    file_path = Path(item['path'])
                    file_ext = file_path.suffix.lower().lstrip('.')
                    
                    if file_ext in ext_set:
                        file_metadata = FileMetadata(
                            path=item['path'],
                            name=file_path.name,
                            extension=file_ext,
                            size=item.get('size', 0),
                            commit_sha=commit_info.sha,
                            commit_date=commit_info.date,
                            author=commit_info.author,
                            gitlab_url=f"{self.gitlab_url}/{self.project_path}",
                            raw_url=f"{self.gitlab_url}/{self.project_path}/-/raw/{branch}/{item['path']}"
                        )
                        files.append(file_metadata)
            
            self.logger.info(f"Found {len(files)} files in {folder_path}")
            return files
            
        except urllib.error.URLError as e:
            self.logger.error(f"Network error listing files in {folder_path}: {e}")
            raise NetworkError(f"Failed to list files: {e}")
        except Exception as e:
            self.logger.error(f"Error listing files in {folder_path}: {e}")
            raise
    
    def download_file(self, file_path: str, branch: str = 'main') -> bytes:
        """Download file content with retry mechanism"""
        if not self.project_id:
            raise RuntimeError("GitLab connection not established - call authenticate() first")
        
        for attempt in range(self.MAX_RETRIES):
            try:
                # Use GitLab API to get file content
                file_url = f"{self.api_base}/projects/{self.project_id}/repository/files/{urllib.parse.quote(file_path, safe='')}"
                params = {'ref': branch}
                query_string = urllib.parse.urlencode(params)
                full_url = f"{file_url}?{query_string}"
                
                status, file_data = self._make_request('GET', full_url)
                
                if status == 404:
                    raise AccessError(f"File not found: {file_path}")
                elif status != 200:
                    raise AccessError(f"Failed to download file: {status}")
                
                # Decode base64 content
                if file_data.get('encoding') == 'base64':
                    content = base64.b64decode(file_data['content'])
                else:
                    content = file_data['content'].encode('utf-8')
                
                self.logger.debug(f"Successfully downloaded: {file_path} ({len(content)} bytes)")
                return content
                
            except urllib.error.URLError as e:
                if attempt < self.MAX_RETRIES - 1:
                    self.logger.warning(f"Download attempt {attempt + 1} failed for {file_path}: {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    self.logger.error(f"Failed to download {file_path} after {self.MAX_RETRIES} attempts: {e}")
                    raise NetworkError(f"Download failed after {self.MAX_RETRIES} attempts: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error downloading {file_path}: {e}")
                raise
    
    def get_commit_info(self, branch: str = 'main') -> CommitMetadata:
        """Get latest commit information for lineage tracking"""
        if not self.project_id:
            raise RuntimeError("GitLab connection not established - call authenticate() first")
        
        try:
            commits_url = f"{self.api_base}/projects/{self.project_id}/repository/commits"
            params = {'ref_name': branch, 'per_page': '1'}
            query_string = urllib.parse.urlencode(params)
            full_url = f"{commits_url}?{query_string}"
            
            status, commits = self._make_request('GET', full_url)
            
            if status != 200:
                raise AccessError(f"Failed to get commit info: {status}")
            
            if not commits:
                raise ValueError(f"No commits found for branch: {branch}")
            
            latest_commit = commits[0]
            
            # Parse the commit date
            commit_date_str = latest_commit.get('created_at') or latest_commit.get('committed_date')
            if commit_date_str:
                # Handle both formats: with 'Z' suffix and with timezone
                if commit_date_str.endswith('Z'):
                    commit_date = datetime.fromisoformat(commit_date_str.replace('Z', '+00:00'))
                else:
                    commit_date = datetime.fromisoformat(commit_date_str)
            else:
                commit_date = datetime.utcnow()
            
            return CommitMetadata(
                sha=latest_commit['id'],
                author=latest_commit.get('author_name', 'Unknown'),
                date=commit_date,
                message=latest_commit.get('message', ''),
                is_latest=True
            )
            
        except urllib.error.URLError as e:
            self.logger.error(f"Network error getting commit info for branch {branch}: {e}")
            raise NetworkError(f"Failed to get commit info: {e}")
        except Exception as e:
            self.logger.error(f"Error getting commit info for branch {branch}: {e}")
            raise
    
    def validate_access(self) -> Dict[str, bool]:
        """Validate access permissions for key operations"""
        validation_result = {
            'authentication': False,
            'project_access': False,
            'repository_read': False,
            'target_folder_access': False
        }
        
        try:
            # Test authentication
            validation_result['authentication'] = self.authenticate()
            
            if validation_result['authentication']:
                validation_result['project_access'] = True
                
                # Test repository read access
                try:
                    self.get_commit_info()
                    validation_result['repository_read'] = True
                except Exception:
                    pass
                
                # Test target folder access
                try:
                    files = self.list_files(folder_path='30-Aprovados')
                    validation_result['target_folder_access'] = True
                except Exception:
                    pass
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
        
        return validation_result
    
    def _make_request(self, method: str, url: str) -> tuple:
        """Make HTTP request with proper headers and return (status, data)"""
        try:
            req = urllib.request.Request(url)
            req.add_header('Authorization', f'Bearer {self.access_token}')
            req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=self.API_TIMEOUT) as response:
                status = response.getcode()
                data = json.loads(response.read().decode('utf-8'))
                return status, data
                
        except urllib.error.HTTPError as e:
            # Return error status and empty data
            return e.code, {}
        except urllib.error.URLError as e:
            raise NetworkError(f"Connection error: {e}")
        except json.JSONDecodeError:
            raise NetworkError("Invalid JSON response")
        except Exception as e:
            raise NetworkError(f"Request error: {e}")

def create_gitlab_connector(config: Dict[str, str]) -> GitLabConnector:
    """Factory function for GitLab connector creation"""
    required_keys = ['gitlab_url', 'access_token', 'project_path']
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return GitLabConnector(
        gitlab_url=config['gitlab_url'],
        access_token=config['access_token'],
        project_path=config['project_path']
    )

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry logic with exponential backoff"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (NetworkError, urllib.error.URLError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise NetworkError(f"Operation failed after {max_retries} attempts: {e}")
    return wrapper