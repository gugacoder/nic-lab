"""
Metadata Extraction for GitLab Content

This module extracts searchable metadata from GitLab repositories,
including commit information, author details, and file attributes.
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from ..integrations.gitlab_client import GitLabClient, get_gitlab_client

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Container for extracted file metadata"""
    project_id: int
    project_name: str
    file_path: str
    file_size: int
    file_extension: str
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    
    # Git metadata
    commit_sha: Optional[str] = None
    branch: str = "main"
    last_commit_date: Optional[datetime] = None
    
    # Author information
    author_name: Optional[str] = None
    author_email: Optional[str] = None
    contributors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tags and categories
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    
    # Additional metadata
    extra: Dict[str, Any] = field(default_factory=dict)


class MetadataExtractor:
    """
    Extract metadata from GitLab files and repositories.
    
    Handles:
    - Git commit information
    - Author and contributor details
    - File attributes
    - Project-specific metadata
    - Tag and category extraction
    """
    
    def __init__(self, gitlab_client: Optional[GitLabClient] = None):
        """
        Initialize metadata extractor.
        
        Args:
            gitlab_client: GitLab client instance
        """
        self.client = gitlab_client or get_gitlab_client()
        
        # Patterns for extracting tags from content
        self.tag_patterns = [
            re.compile(r'#(\w+)', re.IGNORECASE),  # Hashtags
            re.compile(r'@tag:\s*(\w+)', re.IGNORECASE),  # Explicit tags
            re.compile(r'tags?:\s*\[([^\]]+)\]', re.IGNORECASE),  # YAML-style tags
            re.compile(r'tags?:\s*"([^"]+)"', re.IGNORECASE),  # Quoted tags
        ]
        
        # Category patterns
        self.category_patterns = [
            re.compile(r'category:\s*(\w+)', re.IGNORECASE),
            re.compile(r'type:\s*(\w+)', re.IGNORECASE),
            re.compile(r'kind:\s*(\w+)', re.IGNORECASE),
        ]
        
    async def extract_file_metadata(
        self,
        project_id: int,
        file_path: str,
        branch: str = "main",
        content: Optional[str] = None
    ) -> FileMetadata:
        """
        Extract metadata for a specific file.
        
        Args:
            project_id: GitLab project ID
            file_path: Path to the file in the repository
            branch: Git branch name
            content: Optional file content (to avoid refetching)
            
        Returns:
            FileMetadata object with extracted information
        """
        try:
            # Get project info
            project = await self.client.get_project(project_id)
            project_name = project.get('name', f'Project-{project_id}')
            
            # Get file info
            file_info = await self._get_file_info(project_id, file_path, branch)
            
            # Extract basic metadata
            metadata = FileMetadata(
                project_id=project_id,
                project_name=project_name,
                file_path=file_path,
                file_size=file_info.get('size', 0),
                file_extension=os.path.splitext(file_path)[1].lower(),
                branch=branch,
                commit_sha=file_info.get('commit_id'),
                encoding=file_info.get('encoding', 'utf-8')
            )
            
            # Get commit information
            if metadata.commit_sha:
                commit_info = await self._get_commit_info(project_id, metadata.commit_sha)
                if commit_info:
                    metadata.last_commit_date = self._parse_date(commit_info.get('committed_date'))
                    metadata.author_name = commit_info.get('author_name')
                    metadata.author_email = commit_info.get('author_email')
            
            # Get contributor information
            contributors = await self._get_file_contributors(project_id, file_path, branch)
            metadata.contributors = contributors
            
            # Extract tags and categories from content
            if content:
                metadata.tags = self._extract_tags(content, file_path)
                metadata.categories = self._extract_categories(content, file_path)
            
            # Add project-specific metadata
            metadata.extra.update({
                'project_visibility': project.get('visibility', 'private'),
                'project_topics': project.get('topics', []),
                'project_namespace': project.get('namespace', {}).get('full_path', ''),
                'default_branch': project.get('default_branch', 'main'),
                'web_url': file_info.get('web_url', ''),
            })
            
            # Determine MIME type
            metadata.mime_type = self._determine_mime_type(file_path)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {file_path}: {str(e)}")
            # Return minimal metadata on error
            return FileMetadata(
                project_id=project_id,
                project_name=f"Project-{project_id}",
                file_path=file_path,
                file_size=0,
                file_extension=os.path.splitext(file_path)[1].lower(),
                branch=branch
            )
    
    async def extract_bulk_metadata(
        self,
        files: List[Tuple[int, str, str]],
        batch_size: int = 10
    ) -> List[FileMetadata]:
        """
        Extract metadata for multiple files in batches.
        
        Args:
            files: List of (project_id, file_path, branch) tuples
            batch_size: Number of files to process concurrently
            
        Returns:
            List of FileMetadata objects
        """
        import asyncio
        
        results = []
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            tasks = [
                self.extract_file_metadata(project_id, file_path, branch)
                for project_id, file_path, branch in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result, (project_id, file_path, branch) in zip(batch_results, batch):
                if isinstance(result, Exception):
                    logger.error(f"Failed to extract metadata for {file_path}: {result}")
                    # Add minimal metadata for failed files
                    results.append(FileMetadata(
                        project_id=project_id,
                        project_name=f"Project-{project_id}",
                        file_path=file_path,
                        file_size=0,
                        file_extension=os.path.splitext(file_path)[1].lower(),
                        branch=branch
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def _get_file_info(self, project_id: int, file_path: str, branch: str) -> Dict[str, Any]:
        """Get file information from GitLab API"""
        try:
            return await self.client.get_file_info(project_id, file_path, branch)
        except Exception as e:
            logger.warning(f"Could not get file info for {file_path}: {e}")
            return {}
    
    async def _get_commit_info(self, project_id: int, commit_sha: str) -> Optional[Dict[str, Any]]:
        """Get commit information from GitLab API"""
        try:
            return await self.client.get_commit(project_id, commit_sha)
        except Exception as e:
            logger.warning(f"Could not get commit info for {commit_sha}: {e}")
            return None
    
    async def _get_file_contributors(
        self,
        project_id: int,
        file_path: str,
        branch: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get list of contributors for a file"""
        try:
            # Get file blame information
            blame_data = await self.client.get_file_blame(project_id, file_path, branch)
            
            contributors = {}
            for blame_entry in blame_data[:limit]:
                commit = blame_entry.get('commit', {})
                author_email = commit.get('author_email')
                if author_email and author_email not in contributors:
                    contributors[author_email] = {
                        'name': commit.get('author_name'),
                        'email': author_email,
                        'commits': 1
                    }
                elif author_email:
                    contributors[author_email]['commits'] += 1
            
            return list(contributors.values())
            
        except Exception as e:
            logger.warning(f"Could not get contributors for {file_path}: {e}")
            return []
    
    def _extract_tags(self, content: str, file_path: str) -> List[str]:
        """Extract tags from content and file path"""
        tags = set()
        
        # Extract from content patterns
        for pattern in self.tag_patterns:
            matches = pattern.findall(content)
            for match in matches:
                if isinstance(match, str):
                    # Handle comma-separated tags
                    for tag in match.split(','):
                        tag = tag.strip().lower()
                        if tag and len(tag) > 2:
                            tags.add(tag)
        
        # Extract from file path
        path_parts = file_path.lower().split('/')
        for part in path_parts:
            if part in ['docs', 'documentation', 'api', 'examples', 'tests', 'src']:
                tags.add(part)
        
        # Add language tag based on extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext:
            lang_tag = ext[1:]  # Remove the dot
            if lang_tag:
                tags.add(lang_tag)
        
        return sorted(list(tags))
    
    def _extract_categories(self, content: str, file_path: str) -> List[str]:
        """Extract categories from content"""
        categories = set()
        
        # Extract from content patterns
        for pattern in self.category_patterns:
            matches = pattern.findall(content)
            for match in matches:
                category = match.strip().lower()
                if category and len(category) > 2:
                    categories.add(category)
        
        # Infer categories from file path
        if 'test' in file_path.lower():
            categories.add('test')
        if 'doc' in file_path.lower():
            categories.add('documentation')
        if 'example' in file_path.lower():
            categories.add('example')
        if 'config' in file_path.lower():
            categories.add('configuration')
        
        return sorted(list(categories))
    
    def _determine_mime_type(self, file_path: str) -> Optional[str]:
        """Determine MIME type from file extension"""
        mime_types = {
            # Text
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.rst': 'text/x-rst',
            
            # Code
            '.py': 'text/x-python',
            '.js': 'application/javascript',
            '.ts': 'application/typescript',
            '.java': 'text/x-java',
            '.cpp': 'text/x-c++',
            '.c': 'text/x-c',
            '.cs': 'text/x-csharp',
            '.go': 'text/x-go',
            '.rs': 'text/x-rust',
            '.rb': 'text/x-ruby',
            '.php': 'text/x-php',
            
            # Data
            '.json': 'application/json',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml',
            '.xml': 'application/xml',
            '.toml': 'application/toml',
            
            # Web
            '.html': 'text/html',
            '.css': 'text/css',
            
            # Other
            '.sh': 'application/x-sh',
            '.sql': 'application/sql',
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        return mime_types.get(ext, 'text/plain')
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse GitLab date string to datetime"""
        if not date_str:
            return None
        
        try:
            # GitLab uses ISO format with timezone
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except Exception:
            try:
                # Fallback to basic parsing
                return datetime.strptime(date_str[:19], '%Y-%m-%dT%H:%M:%S')
            except Exception:
                logger.warning(f"Could not parse date: {date_str}")
                return None
    
    def enrich_with_project_metadata(
        self,
        metadata: FileMetadata,
        project_info: Dict[str, Any]
    ) -> FileMetadata:
        """
        Enrich file metadata with project-level information.
        
        Args:
            metadata: Existing file metadata
            project_info: Project information from GitLab
            
        Returns:
            Enriched metadata
        """
        # Add project topics as tags
        topics = project_info.get('topics', [])
        for topic in topics:
            if topic and topic not in metadata.tags:
                metadata.tags.append(topic.lower())
        
        # Add project description
        if project_info.get('description'):
            metadata.extra['project_description'] = project_info['description']
        
        # Add repository statistics
        stats = project_info.get('statistics', {})
        if stats:
            metadata.extra['repo_size'] = stats.get('repository_size', 0)
            metadata.extra['commit_count'] = stats.get('commit_count', 0)
        
        return metadata


def test_metadata_extractor():
    """Test metadata extraction functionality"""
    import asyncio
    
    async def run_test():
        extractor = MetadataExtractor()
        
        # Test single file metadata extraction
        print("Testing metadata extraction...")
        
        # Mock file data
        test_file = {
            'project_id': 1,
            'file_path': 'src/example.py',
            'branch': 'main',
            'content': '''
# Example Python File
# @tag: python
# @tag: example
# category: documentation

"""
This is an example file for testing metadata extraction.
Tags: [test, demo, sample]
"""

def hello_world():
    print("Hello, World!")
            '''
        }
        
        # Test tag extraction
        tags = extractor._extract_tags(test_file['content'], test_file['file_path'])
        print(f"Extracted tags: {tags}")
        
        # Test category extraction
        categories = extractor._extract_categories(test_file['content'], test_file['file_path'])
        print(f"Extracted categories: {categories}")
        
        # Test MIME type detection
        mime_type = extractor._determine_mime_type(test_file['file_path'])
        print(f"MIME type: {mime_type}")
    
    asyncio.run(run_test())


if __name__ == "__main__":
    test_metadata_extractor()