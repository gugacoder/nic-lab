"""
GitLab API Client Wrapper

This module provides a high-level interface for GitLab operations including
repository browsing, file operations, search functionality, and proper error handling.
"""

import logging
import time
from typing import Optional, List, Dict, Any, Union, Generator, Tuple
from dataclasses import dataclass
from datetime import datetime

import gitlab
from gitlab.exceptions import GitlabError, GitlabGetError, GitlabCreateError

from .gitlab_auth import GitLabAuthenticator, get_gitlab_authenticator
from ..config.gitlab_config import GitLabInstanceConfig, get_gitlab_config
from ..utils.secrets import secure_logging

logger = logging.getLogger(__name__)


@dataclass
class GitLabProject:
    """Simplified GitLab project information"""
    
    id: int
    name: str
    path_with_namespace: str
    description: Optional[str]
    default_branch: str
    web_url: str
    last_activity_at: datetime
    visibility: str
    wiki_enabled: bool = False
    issues_enabled: bool = False
    
    @classmethod
    def from_gitlab_project(cls, project) -> 'GitLabProject':
        """Create from python-gitlab project object"""
        return cls(
            id=project.id,
            name=project.name,
            path_with_namespace=project.path_with_namespace,
            description=getattr(project, 'description', None),
            default_branch=getattr(project, 'default_branch', 'main'),
            web_url=project.web_url,
            last_activity_at=datetime.fromisoformat(
                project.last_activity_at.replace('Z', '+00:00')
            ),
            visibility=getattr(project, 'visibility', 'private'),
            wiki_enabled=getattr(project, 'wiki_enabled', False),
            issues_enabled=getattr(project, 'issues_enabled', False)
        )


@dataclass
class GitLabFile:
    """GitLab file information"""
    
    project_id: int
    file_path: str
    content: str
    encoding: str
    size: int
    blob_id: str
    commit_id: str
    last_commit_id: str
    ref: str = 'main'
    
    @property
    def is_text(self) -> bool:
        """Check if file is text-based"""
        return self.encoding in ['text', 'base64'] and self.size < 1024 * 1024  # 1MB limit
    
    @property
    def extension(self) -> str:
        """Get file extension"""
        return self.file_path.split('.')[-1].lower() if '.' in self.file_path else ''


@dataclass
class GitLabSearchResult:
    """Search result from GitLab"""
    
    project_id: int
    project_name: str
    file_path: str
    ref: str
    startline: int
    content: str
    wiki: bool = False
    
    @property
    def web_url(self) -> str:
        """Generate web URL for the search result"""
        # This would need the GitLab base URL to construct properly
        return f"project/{self.project_id}/blob/{self.ref}/{self.file_path}#L{self.startline}"


class GitLabClient:
    """High-level GitLab API client"""
    
    def __init__(
        self,
        instance_name: Optional[str] = None,
        authenticator: Optional[GitLabAuthenticator] = None
    ):
        """Initialize GitLab client
        
        Args:
            instance_name: GitLab instance to use, uses primary if None
            authenticator: Authentication manager, uses global if None
        """
        self.instance_name = instance_name
        self.authenticator = authenticator or get_gitlab_authenticator()
        self.config_manager = get_gitlab_config()
        self._gitlab: Optional[gitlab.Gitlab] = None
        self._projects_cache: Dict[int, GitLabProject] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_time = 0
    
    @property
    def gitlab(self) -> gitlab.Gitlab:
        """Get authenticated GitLab client"""
        if self._gitlab is None:
            self._gitlab = self.authenticator.get_connection(self.instance_name)
            if self._gitlab is None:
                raise RuntimeError(f"Failed to authenticate with GitLab instance: {self.instance_name}")
        return self._gitlab
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test GitLab connection
        
        Returns:
            Tuple of (success, message)
        """
        return self.authenticator.test_connection(self.instance_name)
    
    def get_projects(
        self,
        owned: bool = False,
        membership: bool = True,
        search: Optional[str] = None,
        limit: int = 100
    ) -> List[GitLabProject]:
        """Get accessible projects
        
        Args:
            owned: Only owned projects
            membership: Include projects where user is a member
            search: Search term for project names
            limit: Maximum number of projects to return
            
        Returns:
            List of GitLab projects
        """
        with secure_logging():
            logger.info(f"Fetching projects (owned={owned}, membership={membership}, search={search})")
            
            try:
                projects = self.gitlab.projects.list(
                    owned=owned,
                    membership=membership,
                    search=search,
                    order_by='last_activity_at',
                    sort='desc',
                    all=False,
                    per_page=min(limit, 100)
                )
                
                result = []
                for project in projects[:limit]:
                    try:
                        gitlab_project = GitLabProject.from_gitlab_project(project)
                        result.append(gitlab_project)
                        # Cache the project
                        self._projects_cache[gitlab_project.id] = gitlab_project
                    except Exception as e:
                        logger.warning(f"Error processing project {project.id}: {e}")
                
                logger.info(f"Retrieved {len(result)} projects")
                return result
                
            except GitlabError as e:
                logger.error(f"Error fetching projects: {e}")
                raise
    
    def get_project(self, project_id: int) -> Optional[GitLabProject]:
        """Get specific project by ID
        
        Args:
            project_id: Project ID
            
        Returns:
            GitLab project if found, None otherwise
        """
        # Check cache first
        if project_id in self._projects_cache:
            return self._projects_cache[project_id]
        
        try:
            project = self.gitlab.projects.get(project_id)
            gitlab_project = GitLabProject.from_gitlab_project(project)
            self._projects_cache[project_id] = gitlab_project
            return gitlab_project
            
        except GitlabGetError:
            logger.warning(f"Project {project_id} not found or not accessible")
            return None
        except GitlabError as e:
            logger.error(f"Error fetching project {project_id}: {e}")
            return None
    
    def get_file(
        self,
        project_id: int,
        file_path: str,
        ref: str = 'main'
    ) -> Optional[GitLabFile]:
        """Get file content from GitLab
        
        Args:
            project_id: Project ID
            file_path: Path to file in repository
            ref: Git reference (branch, tag, commit)
            
        Returns:
            GitLab file if found, None otherwise
        """
        try:
            project = self.gitlab.projects.get(project_id)
            file = project.files.get(file_path=file_path, ref=ref)
            
            # Decode content if base64
            content = file.content
            if file.encoding == 'base64':
                import base64
                content = base64.b64decode(content).decode('utf-8')
            
            return GitLabFile(
                project_id=project_id,
                file_path=file_path,
                content=content,
                encoding=file.encoding,
                size=file.size,
                blob_id=file.blob_id,
                commit_id=file.commit_id,
                last_commit_id=file.last_commit_id,
                ref=ref
            )
            
        except GitlabGetError:
            logger.debug(f"File not found: {project_id}/{file_path} at {ref}")
            return None
        except GitlabError as e:
            logger.error(f"Error fetching file {project_id}/{file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching file {project_id}/{file_path}: {e}")
            return None
    
    def search_files(
        self,
        query: str,
        project_ids: Optional[List[int]] = None,
        file_extensions: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[GitLabSearchResult]:
        """Search for files across projects
        
        Args:
            query: Search query
            project_ids: Specific projects to search, searches all accessible if None
            file_extensions: File extensions to filter by
            limit: Maximum results to return
            
        Returns:
            List of search results
        """
        with secure_logging():
            logger.info(f"Searching files for query: '{query}'")
            
            results = []
            projects_to_search = project_ids or []
            
            # If no specific projects, get accessible projects
            if not projects_to_search:
                projects = self.get_projects(membership=True, limit=20)
                projects_to_search = [p.id for p in projects]
            
            for project_id in projects_to_search:
                if len(results) >= limit:
                    break
                
                try:
                    project = self.gitlab.projects.get(project_id)
                    project_results = project.search('blobs', query)
                    
                    for result in project_results:
                        if len(results) >= limit:
                            break
                        
                        # Filter by file extension if specified
                        if file_extensions:
                            file_ext = result['filename'].split('.')[-1].lower()
                            if file_ext not in file_extensions:
                                continue
                        
                        search_result = GitLabSearchResult(
                            project_id=project_id,
                            project_name=project.name,
                            file_path=result['filename'],
                            ref=result.get('ref', 'main'),
                            startline=result.get('startline', 1),
                            content=result.get('data', '')
                        )
                        results.append(search_result)
                
                except GitlabError as e:
                    logger.warning(f"Error searching project {project_id}: {e}")
                    continue
            
            logger.info(f"Found {len(results)} search results")
            return results
    
    def search_wikis(
        self,
        query: str,
        project_ids: Optional[List[int]] = None,
        limit: int = 20
    ) -> List[GitLabSearchResult]:
        """Search wiki content across projects
        
        Args:
            query: Search query
            project_ids: Specific projects to search wikis
            limit: Maximum results to return
            
        Returns:
            List of wiki search results
        """
        results = []
        projects_to_search = project_ids or []
        
        # If no specific projects, get projects with wikis enabled
        if not projects_to_search:
            projects = self.get_projects(membership=True, limit=20)
            projects_to_search = [p.id for p in projects if p.wiki_enabled]
        
        for project_id in projects_to_search:
            if len(results) >= limit:
                break
            
            try:
                project = self.gitlab.projects.get(project_id)
                
                # Search wiki pages
                wiki_results = project.search('wiki_blobs', query)
                
                for result in wiki_results:
                    if len(results) >= limit:
                        break
                    
                    search_result = GitLabSearchResult(
                        project_id=project_id,
                        project_name=project.name,
                        file_path=result['filename'],
                        ref='main',
                        startline=result.get('startline', 1),
                        content=result.get('data', ''),
                        wiki=True
                    )
                    results.append(search_result)
            
            except GitlabError as e:
                logger.warning(f"Error searching wiki for project {project_id}: {e}")
                continue
        
        logger.info(f"Found {len(results)} wiki search results")
        return results
    
    def create_file(
        self,
        project_id: int,
        file_path: str,
        content: str,
        commit_message: str,
        branch: Optional[str] = None,
        author_email: Optional[str] = None,
        author_name: Optional[str] = None
    ) -> bool:
        """Create a new file in GitLab
        
        Args:
            project_id: Project ID
            file_path: Path for new file
            content: File content
            commit_message: Commit message
            branch: Target branch, uses default if None
            author_email: Author email
            author_name: Author name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            project = self.gitlab.projects.get(project_id)
            
            # Use default branch if not specified
            if branch is None:
                branch = getattr(project, 'default_branch', 'main')
            
            file_data = {
                'file_path': file_path,
                'branch': branch,
                'content': content,
                'commit_message': commit_message
            }
            
            if author_email:
                file_data['author_email'] = author_email
            if author_name:
                file_data['author_name'] = author_name
            
            project.files.create(file_data)
            logger.info(f"Created file {file_path} in project {project_id}")
            return True
            
        except GitlabCreateError as e:
            logger.error(f"Error creating file {file_path}: {e}")
            return False
        except GitlabError as e:
            logger.error(f"GitLab error creating file {file_path}: {e}")
            return False
    
    def update_file(
        self,
        project_id: int,
        file_path: str,
        content: str,
        commit_message: str,
        branch: Optional[str] = None
    ) -> bool:
        """Update an existing file in GitLab
        
        Args:
            project_id: Project ID
            file_path: Path to existing file
            content: New file content
            commit_message: Commit message
            branch: Target branch, uses default if None
            
        Returns:
            True if successful, False otherwise
        """
        try:
            project = self.gitlab.projects.get(project_id)
            
            # Use default branch if not specified
            if branch is None:
                branch = getattr(project, 'default_branch', 'main')
            
            file = project.files.get(file_path=file_path, ref=branch)
            file.content = content
            file.commit_message = commit_message
            file.save(branch=branch)
            
            logger.info(f"Updated file {file_path} in project {project_id}")
            return True
            
        except GitlabGetError:
            logger.error(f"File not found: {file_path} in project {project_id}")
            return False
        except GitlabError as e:
            logger.error(f"Error updating file {file_path}: {e}")
            return False
    
    def get_file_tree(
        self,
        project_id: int,
        path: str = '',
        ref: str = 'main',
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """Get file tree for a project path
        
        Args:
            project_id: Project ID
            path: Directory path (empty for root)
            ref: Git reference
            recursive: Get all files recursively
            
        Returns:
            List of file/directory information
        """
        try:
            project = self.gitlab.projects.get(project_id)
            tree = project.repository_tree(path=path, ref=ref, recursive=recursive, all=True)
            
            return [
                {
                    'id': item['id'],
                    'name': item['name'],
                    'type': item['type'],
                    'path': item['path'],
                    'mode': item['mode']
                }
                for item in tree
            ]
            
        except GitlabError as e:
            logger.error(f"Error getting file tree for project {project_id}: {e}")
            return []
    
    def clear_cache(self):
        """Clear the projects cache"""
        self._projects_cache.clear()
        self._last_cache_time = 0
        logger.debug("Cleared GitLab client cache")


# Global client instance
_gitlab_client: Optional[GitLabClient] = None


def get_gitlab_client(instance_name: Optional[str] = None) -> GitLabClient:
    """Get global GitLab client instance
    
    Args:
        instance_name: GitLab instance to use
        
    Returns:
        GitLab client instance
    """
    global _gitlab_client
    if _gitlab_client is None or _gitlab_client.instance_name != instance_name:
        _gitlab_client = GitLabClient(instance_name)
    return _gitlab_client


if __name__ == "__main__":
    # Test GitLab client functionality
    import sys
    
    print("Testing GitLab client...")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test-connection":
        # Test connection
        client = get_gitlab_client()
        success, message = client.test_connection()
        print(f"Connection test: {message}")
        if not success:
            sys.exit(1)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "list-projects":
        # List projects
        client = get_gitlab_client()
        projects = client.get_projects(membership=True, limit=10)
        print(f"Found {len(projects)} projects:")
        for project in projects:
            print(f"  {project.id}: {project.name} ({project.path_with_namespace})")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        # Search files
        if len(sys.argv) < 3:
            print("Usage: python -m src.integrations.gitlab_client search <query>")
            sys.exit(1)
        
        query = sys.argv[2]
        client = get_gitlab_client()
        results = client.search_files(query, limit=10)
        print(f"Found {len(results)} search results for '{query}':")
        for result in results:
            print(f"  {result.project_name}: {result.file_path}")
    
    else:
        print("Usage:")
        print("  python -m src.integrations.gitlab_client test-connection")
        print("  python -m src.integrations.gitlab_client list-projects")
        print("  python -m src.integrations.gitlab_client search <query>")
    
    print("GitLab client testing complete")