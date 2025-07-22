"""
GitLab Integration Example
Demonstrates python-gitlab usage for repository access
"""

import os
import gitlab
from typing import List, Dict, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitLabClient:
    """Example GitLab client implementation"""
    
    def __init__(self, url: str, private_token: str):
        """Initialize GitLab connection"""
        self.gl = gitlab.Gitlab(url, private_token=private_token)
        self._authenticate()
    
    def _authenticate(self):
        """Verify authentication"""
        try:
            self.gl.auth()
            logger.info(f"Authenticated as: {self.gl.user.username}")
        except gitlab.GitlabAuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def search_content(self, query: str, project_ids: List[int] = None) -> List[Dict]:
        """Search across projects for content"""
        results = []
        
        # Get projects to search
        if project_ids:
            projects = [self.gl.projects.get(pid) for pid in project_ids]
        else:
            projects = self.gl.projects.list(owned=True, all=True)
        
        for project in projects:
            try:
                # Search in repository files
                items = project.search('blobs', query)
                for item in items:
                    results.append({
                        'project': project.name,
                        'project_id': project.id,
                        'filename': item['filename'],
                        'path': item['path'],
                        'content': item['data'],
                        'ref': item['ref']
                    })
                
                # Search in wiki if enabled
                if project.wiki_enabled:
                    wiki_pages = project.wikis.list(all=True)
                    for page in wiki_pages:
                        if query.lower() in page.content.lower():
                            results.append({
                                'project': project.name,
                                'project_id': project.id,
                                'filename': f"{page.slug}.md",
                                'path': f"wiki/{page.slug}",
                                'content': page.content,
                                'ref': 'wiki'
                            })
                            
            except Exception as e:
                logger.warning(f"Error searching project {project.name}: {e}")
                continue
        
        return results
    
    def get_file_content(self, project_id: int, file_path: str, ref: str = 'main') -> str:
        """Retrieve specific file content"""
        try:
            project = self.gl.projects.get(project_id)
            file = project.files.get(file_path, ref=ref)
            return file.decode().decode('utf-8')
        except gitlab.GitlabGetError as e:
            logger.error(f"Failed to get file {file_path}: {e}")
            raise
    
    def create_or_update_file(self, project_id: int, file_path: str, 
                             content: str, commit_message: str, 
                             branch: str = 'main') -> Dict:
        """Create or update a file in the repository"""
        project = self.gl.projects.get(project_id)
        
        try:
            # Try to get existing file
            file = project.files.get(file_path, ref=branch)
            # Update existing file
            file.content = content
            file.save(branch=branch, commit_message=commit_message)
            action = 'updated'
        except gitlab.GitlabGetError:
            # Create new file
            file = project.files.create({
                'file_path': file_path,
                'branch': branch,
                'content': content,
                'commit_message': commit_message
            })
            action = 'created'
        
        return {
            'action': action,
            'file_path': file_path,
            'branch': branch,
            'web_url': f"{project.web_url}/-/blob/{branch}/{file_path}"
        }
    
    def list_projects(self) -> List[Dict]:
        """List accessible projects"""
        projects = []
        for project in self.gl.projects.list(owned=True, all=True):
            projects.append({
                'id': project.id,
                'name': project.name,
                'path': project.path_with_namespace,
                'web_url': project.web_url,
                'default_branch': project.default_branch,
                'last_activity': project.last_activity_at
            })
        return projects


# Example usage
if __name__ == "__main__":
    # Load configuration from environment
    GITLAB_URL = os.getenv('GITLAB_URL', 'https://gitlab.com')
    GITLAB_TOKEN = os.getenv('GITLAB_TOKEN')
    
    if not GITLAB_TOKEN:
        print("Please set GITLAB_TOKEN environment variable")
        exit(1)
    
    # Initialize client
    client = GitLabClient(GITLAB_URL, GITLAB_TOKEN)
    
    # List projects
    print("\n=== Available Projects ===")
    projects = client.list_projects()
    for proj in projects[:5]:  # Show first 5
        print(f"- {proj['name']} (ID: {proj['id']})")
    
    # Search example
    print("\n=== Search Example ===")
    results = client.search_content("README")
    print(f"Found {len(results)} results")
    for result in results[:3]:  # Show first 3
        print(f"- {result['project']}: {result['path']}")
    
    # File operations example
    if projects:
        project_id = projects[0]['id']
        print(f"\n=== File Operations on Project {projects[0]['name']} ===")
        
        # Create a test file
        test_content = f"# Test Document\nCreated at {datetime.now()}\n\nThis is a test document."
        result = client.create_or_update_file(
            project_id=project_id,
            file_path='test/example.md',
            content=test_content,
            commit_message='Add test document from NIC Chat'
        )
        print(f"File {result['action']}: {result['web_url']}")