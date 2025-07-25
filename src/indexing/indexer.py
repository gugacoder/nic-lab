"""
Main Search Indexer for GitLab Content

This module implements the core indexing engine that processes GitLab
content and maintains search indexes with incremental update support.
"""

import os
import sys
import time
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from whoosh import index
from whoosh.writing import AsyncWriter
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Term, And, Or

from .schema import IndexSchema, ContentType
from .text_processor import TextProcessor
from .metadata_extractor import MetadataExtractor
from ..integrations.gitlab_client import GitLabClient, get_gitlab_client

logger = logging.getLogger(__name__)


class IndexStatus(Enum):
    """Index operation status"""
    IDLE = "idle"
    BUILDING = "building"
    UPDATING = "updating"
    OPTIMIZING = "optimizing"
    ERROR = "error"


@dataclass
class IndexConfig:
    """Configuration for the search indexer"""
    index_dir: str = "indexes"
    batch_size: int = 100
    commit_interval: int = 500  # Commit every N documents
    enable_async: bool = True
    enable_spell_check: bool = True
    ram_buffer_size: int = 256  # MB
    merge_factor: int = 10
    max_field_length: int = 100000
    incremental_update_interval: int = 300  # seconds
    full_rebuild_interval: int = 86400  # 24 hours
    
    def __post_init__(self):
        # Ensure index directory exists
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class IndexStats:
    """Statistics about the index"""
    total_documents: int = 0
    indexed_projects: Set[int] = field(default_factory=set)
    last_update: Optional[datetime] = None
    last_full_rebuild: Optional[datetime] = None
    index_size_mb: float = 0.0
    indexing_time_seconds: float = 0.0
    status: IndexStatus = IndexStatus.IDLE
    errors: List[str] = field(default_factory=list)


class SearchIndexer:
    """
    Main indexing engine for GitLab content.
    
    Features:
    - Incremental indexing for new/modified content
    - Full index rebuilds
    - Async processing for performance
    - Index optimization and maintenance
    - Search functionality
    """
    
    def __init__(
        self,
        config: Optional[IndexConfig] = None,
        gitlab_client: Optional[GitLabClient] = None
    ):
        """
        Initialize the search indexer.
        
        Args:
            config: Indexer configuration
            gitlab_client: GitLab client instance
        """
        self.config = config or IndexConfig()
        self.client = gitlab_client or get_gitlab_client()
        self.text_processor = TextProcessor()
        self.metadata_extractor = MetadataExtractor(self.client)
        
        self._index = None
        self._writer = None
        self.stats = IndexStats()
        self._lock = asyncio.Lock()
        
        # Track indexed content for incremental updates
        self._indexed_content: Dict[str, Dict[str, Any]] = {}
        
    @property
    def index(self):
        """Get or create the search index"""
        if self._index is None:
            self._index = self._get_or_create_index()
        return self._index
    
    def _get_or_create_index(self):
        """Get existing index or create a new one"""
        index_path = os.path.join(self.config.index_dir, "main")
        
        if index.exists_in(index_path):
            logger.info(f"Opening existing index at {index_path}")
            return index.open_dir(index_path)
        else:
            logger.info(f"Creating new index at {index_path}")
            os.makedirs(index_path, exist_ok=True)
            schema = IndexSchema.get_schema()
            return index.create_in(index_path, schema)
    
    async def build_index(
        self,
        project_ids: Optional[List[int]] = None,
        file_extensions: Optional[List[str]] = None,
        force_rebuild: bool = False
    ) -> IndexStats:
        """
        Build or rebuild the search index.
        
        Args:
            project_ids: Specific projects to index (None for all)
            file_extensions: File extensions to include
            force_rebuild: Force full rebuild even if index exists
            
        Returns:
            Index statistics
        """
        async with self._lock:
            if self.stats.status != IndexStatus.IDLE and not force_rebuild:
                logger.warning("Indexing already in progress")
                return self.stats
            
            self.stats.status = IndexStatus.BUILDING
            self.stats.errors.clear()
            start_time = time.time()
            
            try:
                if force_rebuild:
                    logger.info("Starting full index rebuild")
                    self._clear_index()
                else:
                    logger.info("Starting incremental index update")
                
                # Get list of files to index
                files_to_index = await self._get_files_to_index(
                    project_ids,
                    file_extensions,
                    incremental=not force_rebuild
                )
                
                logger.info(f"Found {len(files_to_index)} files to index")
                
                # Index files in batches
                await self._index_files(files_to_index)
                
                # Update statistics
                self.stats.indexing_time_seconds = time.time() - start_time
                self.stats.last_update = datetime.now()
                if force_rebuild:
                    self.stats.last_full_rebuild = datetime.now()
                
                # Calculate index size
                self._update_index_stats()
                
                logger.info(f"Indexing completed in {self.stats.indexing_time_seconds:.1f}s")
                logger.info(f"Total documents: {self.stats.total_documents}")
                
                self.stats.status = IndexStatus.IDLE
                return self.stats
                
            except Exception as e:
                logger.error(f"Indexing failed: {str(e)}")
                self.stats.status = IndexStatus.ERROR
                self.stats.errors.append(str(e))
                raise
    
    async def _get_files_to_index(
        self,
        project_ids: Optional[List[int]],
        file_extensions: Optional[List[str]],
        incremental: bool = False
    ) -> List[Tuple[int, str, str, Dict[str, Any]]]:
        """
        Get list of files that need to be indexed.
        
        Returns:
            List of (project_id, file_path, branch, file_info) tuples
        """
        files_to_index = []
        
        # Get all accessible projects
        if project_ids is None:
            projects = await self.client.get_all_projects()
            project_ids = [p['id'] for p in projects]
        
        for project_id in project_ids:
            try:
                # Get project info
                project = await self.client.get_project(project_id)
                default_branch = project.get('default_branch', 'main')
                
                # Get repository tree
                tree = await self.client.get_repository_tree(
                    project_id,
                    ref=default_branch,
                    recursive=True
                )
                
                for item in tree:
                    if item.get('type') != 'blob':
                        continue
                    
                    file_path = item['path']
                    
                    # Filter by extension if specified
                    if file_extensions:
                        ext = os.path.splitext(file_path)[1].lower()
                        if ext not in file_extensions:
                            continue
                    
                    # Skip non-text files
                    if not self._is_indexable_file(file_path):
                        continue
                    
                    # Check if file needs indexing (for incremental updates)
                    if incremental and self._is_file_indexed(project_id, file_path, item):
                        continue
                    
                    files_to_index.append((
                        project_id,
                        file_path,
                        default_branch,
                        item
                    ))
                    
                self.stats.indexed_projects.add(project_id)
                
            except Exception as e:
                logger.error(f"Error getting files for project {project_id}: {e}")
                self.stats.errors.append(f"Project {project_id}: {str(e)}")
        
        return files_to_index
    
    async def _index_files(self, files: List[Tuple[int, str, str, Dict[str, Any]]]):
        """Index a list of files in batches"""
        writer = self._get_writer()
        documents_added = 0
        
        try:
            for i in range(0, len(files), self.config.batch_size):
                batch = files[i:i + self.config.batch_size]
                
                # Process batch concurrently
                tasks = [
                    self._process_file(project_id, file_path, branch, file_info)
                    for project_id, file_path, branch, file_info in batch
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Add documents to index
                for result, (project_id, file_path, _, _) in zip(results, batch):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process {file_path}: {result}")
                        self.stats.errors.append(f"{file_path}: {str(result)}")
                    elif result:
                        writer.add_document(**result)
                        documents_added += 1
                        
                        # Track indexed content
                        doc_id = result['doc_id']
                        self._indexed_content[doc_id] = {
                            'project_id': project_id,
                            'file_path': file_path,
                            'commit_sha': result.get('commit_sha'),
                            'updated_at': result.get('updated_at')
                        }
                
                # Commit periodically
                if documents_added % self.config.commit_interval == 0:
                    writer.commit()
                    writer = self._get_writer()
                    logger.info(f"Indexed {documents_added} documents...")
            
            # Final commit
            writer.commit()
            self.stats.total_documents = documents_added
            
        except Exception as e:
            writer.cancel()
            raise
    
    async def _process_file(
        self,
        project_id: int,
        file_path: str,
        branch: str,
        file_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single file for indexing.
        
        Returns:
            Document dictionary ready for indexing
        """
        try:
            # Get file content
            content = await self.client.get_file_content(project_id, file_path, branch)
            
            # Process text
            processed = self.text_processor.process(content, file_path)
            
            # Extract metadata
            metadata = await self.metadata_extractor.extract_file_metadata(
                project_id,
                file_path,
                branch,
                content
            )
            
            # Create document ID
            doc_id = f"{project_id}/{file_path}"
            
            # Prepare document for indexing
            document = IndexSchema.prepare_document({
                'doc_id': doc_id,
                'project_id': metadata.project_id,
                'project_name': metadata.project_name,
                'file_path': metadata.file_path,
                'content': processed.content,
                'title': processed.title or metadata.file_path,
                'headings': ' '.join(processed.headings),
                'author_name': metadata.author_name,
                'author_email': metadata.author_email,
                'tags': ','.join(metadata.tags),
                'categories': ','.join(metadata.categories),
                'file_size': metadata.file_size,
                'word_count': processed.word_count,
                'language': processed.language or metadata.extra.get('language'),
                'commit_sha': metadata.commit_sha,
                'branch': metadata.branch,
                'created_at': metadata.last_commit_date,
                'updated_at': metadata.last_commit_date,
                'extra_metadata': metadata.extra
            })
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _get_writer(self):
        """Get index writer (async if enabled)"""
        if self.config.enable_async:
            return AsyncWriter(self.index, writerargs={
                'procs': 4,
                'multisegment': True,
                'limitmb': self.config.ram_buffer_size
            })
        else:
            return self.index.writer(
                limitmb=self.config.ram_buffer_size,
                procs=1
            )
    
    def _is_indexable_file(self, file_path: str) -> bool:
        """Check if file should be indexed"""
        # Skip binary files
        binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.a', '.o',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.pyc', '.class', '.jar', '.war',
            '.db', '.sqlite', '.lock'
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in binary_extensions:
            return False
        
        # Skip hidden files and directories
        parts = file_path.split('/')
        for part in parts:
            if part.startswith('.') and part not in ['.github', '.gitlab']:
                return False
        
        # Skip common non-content directories
        skip_dirs = {'node_modules', 'venv', 'env', '__pycache__', 'dist', 'build', 'target'}
        for skip_dir in skip_dirs:
            if skip_dir in parts:
                return False
        
        return True
    
    def _is_file_indexed(
        self,
        project_id: int,
        file_path: str,
        file_info: Dict[str, Any]
    ) -> bool:
        """Check if file is already indexed and up to date"""
        doc_id = f"{project_id}/{file_path}"
        
        if doc_id not in self._indexed_content:
            return False
        
        indexed_info = self._indexed_content[doc_id]
        
        # Check if file has been modified
        if 'commit_id' in file_info:
            return indexed_info.get('commit_sha') == file_info['commit_id']
        
        return True
    
    def _clear_index(self):
        """Clear the entire index"""
        writer = self.index.writer()
        # Clear all documents
        from whoosh.query import Every
        writer.delete_by_query(Every())
        writer.commit()
        self._indexed_content.clear()
        self.stats.total_documents = 0
        self.stats.indexed_projects.clear()
    
    def _update_index_stats(self):
        """Update index statistics"""
        with self.index.searcher() as searcher:
            self.stats.total_documents = searcher.doc_count()
        
        # Calculate index size
        index_path = os.path.join(self.config.index_dir, "main")
        total_size = 0
        for root, dirs, files in os.walk(index_path):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        
        self.stats.index_size_mb = total_size / (1024 * 1024)
    
    async def update_incremental(self) -> IndexStats:
        """
        Perform incremental index update for modified files.
        
        Returns:
            Index statistics
        """
        logger.info("Starting incremental index update")
        return await self.build_index(force_rebuild=False)
    
    async def optimize_index(self):
        """Optimize the index for better search performance"""
        async with self._lock:
            self.stats.status = IndexStatus.OPTIMIZING
            
            try:
                logger.info("Starting index optimization")
                writer = self.index.writer()
                writer.commit(optimize=True)
                logger.info("Index optimization completed")
                
                self._update_index_stats()
                self.stats.status = IndexStatus.IDLE
                
            except Exception as e:
                logger.error(f"Index optimization failed: {e}")
                self.stats.status = IndexStatus.ERROR
                self.stats.errors.append(f"Optimization: {str(e)}")
                raise
    
    def search(
        self,
        query_string: str,
        fields: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 50,
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Search the index.
        
        Args:
            query_string: Search query
            fields: Fields to search (None for default fields)
            filter_dict: Filters to apply
            limit: Maximum results per page
            page: Page number (1-based)
            
        Returns:
            Search results with metadata
        """
        with self.index.searcher() as searcher:
            # Build parser
            if fields is None:
                fields = IndexSchema.get_searchable_fields()
            
            parser = MultifieldParser(fields, self.index.schema)
            
            try:
                query = parser.parse(query_string)
            except Exception as e:
                logger.error(f"Invalid query: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'results': [],
                    'total': 0
                }
            
            # Apply filters
            if filter_dict:
                filter_query = And([
                    Term(field, value)
                    for field, value in filter_dict.items()
                ])
                query = And([query, filter_query])
            
            # Execute search
            results = searcher.search_page(query, page, pagelen=limit)
            
            # Format results
            hits = []
            for hit in results:
                hits.append({
                    'doc_id': hit['doc_id'],
                    'project_name': hit['project_name'],
                    'file_path': hit['file_path'],
                    'title': hit.get('title', ''),
                    'content_preview': hit['content'][:200] + '...',
                    'score': hit.score,
                    'tags': hit.get('tags', '').split(','),
                    'updated_at': hit.get('updated_at')
                })
            
            return {
                'success': True,
                'results': hits,
                'total': len(results),
                'page': page,
                'total_pages': results.pagecount,
                'query': query_string
            }
    
    async def get_stats(self) -> IndexStats:
        """Get current index statistics"""
        self._update_index_stats()
        return self.stats


async def test_indexer():
    """Test indexer functionality"""
    config = IndexConfig(index_dir="test_indexes")
    indexer = SearchIndexer(config)
    
    # Test index building
    print("Building test index...")
    stats = await indexer.build_index(force_rebuild=True)
    print(f"Indexed {stats.total_documents} documents")
    print(f"Index size: {stats.index_size_mb:.1f} MB")
    
    # Test searching
    print("\nTesting search...")
    results = indexer.search("authentication", limit=5)
    print(f"Found {results['total']} results")
    for hit in results['results']:
        print(f"  - {hit['file_path']} (score: {hit['score']:.2f})")
    
    # Test incremental update
    print("\nTesting incremental update...")
    update_stats = await indexer.update_incremental()
    print(f"Updated {update_stats.total_documents} documents")


if __name__ == "__main__":
    # Test the indexer
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        # Parse --project argument
        project_id = None
        if "--project" in sys.argv:
            try:
                project_idx = sys.argv.index("--project")
                if project_idx + 1 < len(sys.argv):
                    project_arg = sys.argv[project_idx + 1]
                    if project_arg.isdigit():
                        project_id = int(project_arg)
            except (ValueError, IndexError):
                pass
        
        project_ids = [project_id] if project_id else None
        
        async def build():
            indexer = SearchIndexer()
            await indexer.build_index(project_ids=project_ids, force_rebuild=True)
        
        asyncio.run(build())
    
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "test"
        
        indexer = SearchIndexer()
        results = indexer.search(query)
        
        print(f"Search results for '{query}':")
        for hit in results['results']:
            print(f"  {hit['project_name']}: {hit['file_path']}")
            print(f"    {hit['content_preview']}")
            print()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        async def show_stats():
            indexer = SearchIndexer()
            stats = await indexer.get_stats()
            print(f"Index Statistics:")
            print(f"  Total documents: {stats.total_documents}")
            print(f"  Projects indexed: {len(stats.indexed_projects)}")
            print(f"  Index size: {stats.index_size_mb:.1f} MB")
            print(f"  Last update: {stats.last_update}")
            print(f"  Status: {stats.status.value}")
        
        asyncio.run(show_stats())
    
    else:
        import sys
        asyncio.run(test_indexer())