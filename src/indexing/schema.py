"""
Index Schema Definitions for GitLab Content

This module defines the Whoosh schema for indexing different types of content
from GitLab repositories, including markdown documents, code files, and wiki pages.
"""

import os
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum

from whoosh import fields, analysis, formats
from whoosh.fields import Schema, TEXT, ID, DATETIME, NUMERIC, KEYWORD, STORED
from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer, LanguageAnalyzer
from whoosh.analysis import NgramWordAnalyzer, SimpleAnalyzer, RegexTokenizer
from whoosh.analysis import LowercaseFilter, StopFilter, StemFilter


class ContentType(Enum):
    """Enumeration of content types we index"""
    MARKDOWN = "markdown"
    CODE = "code"
    WIKI = "wiki"
    JSON = "json"
    YAML = "yaml"
    TEXT = "text"
    OTHER = "other"


class IndexSchema:
    """
    Defines the search index schema for GitLab content.
    
    The schema is designed to support:
    - Full-text search with ranking
    - Metadata filtering
    - Code-specific searching
    - Multi-language support
    """
    
    # Custom analyzers for different content types
    @staticmethod
    def get_text_analyzer() -> analysis.Analyzer:
        """Get analyzer for general text content"""
        return StemmingAnalyzer(
            stoplist=analysis.STOP_WORDS,
            minsize=2,
            maxsize=40
        )
    
    @staticmethod
    def get_code_analyzer() -> analysis.Analyzer:
        """Get analyzer for code content with less aggressive tokenization"""
        # Use regex tokenizer that preserves underscores and dots
        tokenizer = RegexTokenizer(r"[a-zA-Z_][a-zA-Z0-9_\.]*|[0-9]+")
        return tokenizer | LowercaseFilter()
    
    @staticmethod
    def get_path_analyzer() -> analysis.Analyzer:
        """Get analyzer for file paths"""
        # Split on path separators and dots
        tokenizer = RegexTokenizer(r"[^/\\\\.]+")
        return tokenizer | LowercaseFilter()
    
    @staticmethod
    def get_ngram_analyzer() -> analysis.Analyzer:
        """Get n-gram analyzer for fuzzy matching"""
        return NgramWordAnalyzer(minsize=3, maxsize=4)
    
    @classmethod
    def get_schema(cls) -> Schema:
        """
        Get the Whoosh schema for indexing GitLab content.
        
        Returns:
            Whoosh Schema object defining all indexed fields
        """
        return Schema(
            # Document identification
            doc_id=ID(unique=True, stored=True),
            project_id=NUMERIC(stored=True),
            project_name=TEXT(stored=True, analyzer=cls.get_text_analyzer()),
            file_path=TEXT(stored=True, analyzer=cls.get_path_analyzer()),
            
            # Content fields with different analyzers
            content=TEXT(
                stored=True,
                analyzer=cls.get_text_analyzer(),
                phrase=True,  # Enable phrase searching
                chars=True,   # Store character positions for highlighting
                vector=formats.Positions()  # Store term positions
            ),
            
            # Code-specific content field
            code_content=TEXT(
                stored=False,  # Don't store separately, use content
                analyzer=cls.get_code_analyzer(),
                phrase=True
            ),
            
            # N-gram field for fuzzy search
            content_ngrams=TEXT(
                stored=False,
                analyzer=cls.get_ngram_analyzer()
            ),
            
            # Metadata fields
            content_type=ID(stored=True),
            file_extension=ID(stored=True),
            language=ID(stored=True),  # Programming language or document type
            
            # Temporal fields
            created_at=DATETIME(stored=True),
            updated_at=DATETIME(stored=True),
            indexed_at=DATETIME(stored=True),
            
            # Author information
            author_name=TEXT(stored=True, analyzer=SimpleAnalyzer()),
            author_email=ID(stored=True),
            
            # Document structure
            title=TEXT(
                stored=True,
                analyzer=cls.get_text_analyzer(),
                field_boost=2.0  # Boost title matches
            ),
            headings=TEXT(
                stored=True,
                analyzer=cls.get_text_analyzer(),
                field_boost=1.5  # Boost heading matches
            ),
            
            # Line information for precise matching
            start_line=NUMERIC(stored=True),
            end_line=NUMERIC(stored=True),
            
            # Tags and categories (comma-separated values)
            tags=KEYWORD(stored=True, commas=True, scorable=True),
            categories=KEYWORD(stored=True, commas=True),
            
            # Search optimization fields
            file_size=NUMERIC(stored=True),
            word_count=NUMERIC(stored=True),
            
            # Version control information
            commit_sha=ID(stored=True),
            branch=ID(stored=True),
            
            # Additional metadata as JSON
            extra_metadata=STORED  # Stored but not indexed
        )
    
    @classmethod
    def get_field_boosts(cls) -> Dict[str, float]:
        """
        Get field boost values for search ranking.
        
        Returns:
            Dictionary mapping field names to boost values
        """
        return {
            'title': 3.0,
            'file_path': 2.5,
            'headings': 2.0,
            'project_name': 1.5,
            'tags': 1.5,
            'content': 1.0,
            'code_content': 0.8,
            'content_ngrams': 0.5
        }
    
    @classmethod
    def get_searchable_fields(cls) -> List[str]:
        """
        Get list of fields that should be searched by default.
        
        Returns:
            List of searchable field names
        """
        return [
            'content',
            'title',
            'headings',
            'file_path',
            'project_name',
            'tags',
            'code_content',
            'content_ngrams'
        ]
    
    @classmethod
    def get_filterable_fields(cls) -> List[str]:
        """
        Get list of fields that can be used for filtering.
        
        Returns:
            List of filterable field names
        """
        return [
            'project_id',
            'project_name',
            'content_type',
            'file_extension',
            'language',
            'author_email',
            'branch',
            'tags',
            'categories'
        ]
    
    @classmethod
    def get_sortable_fields(cls) -> List[str]:
        """
        Get list of fields that can be used for sorting results.
        
        Returns:
            List of sortable field names
        """
        return [
            'updated_at',
            'created_at',
            'file_size',
            'word_count',
            'file_path'
        ]
    
    @classmethod
    def determine_content_type(cls, file_path: str) -> ContentType:
        """
        Determine content type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            ContentType enum value
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        # Mapping of extensions to content types
        extension_map = {
            # Markdown
            '.md': ContentType.MARKDOWN,
            '.markdown': ContentType.MARKDOWN,
            '.mdown': ContentType.MARKDOWN,
            '.mkd': ContentType.MARKDOWN,
            
            # Wiki
            '.wiki': ContentType.WIKI,
            '.mediawiki': ContentType.WIKI,
            '.rst': ContentType.WIKI,
            '.adoc': ContentType.WIKI,
            '.asciidoc': ContentType.WIKI,
            
            # Code
            '.py': ContentType.CODE,
            '.js': ContentType.CODE,
            '.ts': ContentType.CODE,
            '.java': ContentType.CODE,
            '.cpp': ContentType.CODE,
            '.c': ContentType.CODE,
            '.h': ContentType.CODE,
            '.cs': ContentType.CODE,
            '.go': ContentType.CODE,
            '.rs': ContentType.CODE,
            '.rb': ContentType.CODE,
            '.php': ContentType.CODE,
            '.swift': ContentType.CODE,
            '.kt': ContentType.CODE,
            '.scala': ContentType.CODE,
            '.r': ContentType.CODE,
            '.m': ContentType.CODE,
            '.sh': ContentType.CODE,
            '.bash': ContentType.CODE,
            '.ps1': ContentType.CODE,
            '.sql': ContentType.CODE,
            
            # Configuration
            '.json': ContentType.JSON,
            '.yaml': ContentType.YAML,
            '.yml': ContentType.YAML,
            '.toml': ContentType.YAML,
            '.ini': ContentType.YAML,
            '.conf': ContentType.YAML,
            '.config': ContentType.YAML,
            
            # Text
            '.txt': ContentType.TEXT,
            '.text': ContentType.TEXT,
            '.log': ContentType.TEXT
        }
        
        return extension_map.get(ext, ContentType.OTHER)
    
    @classmethod
    def prepare_document(cls, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a document for indexing by ensuring all required fields.
        
        Args:
            doc_data: Raw document data
            
        Returns:
            Prepared document with all schema fields
        """
        # Determine content type
        file_path = doc_data.get('file_path', '')
        content_type = cls.determine_content_type(file_path)
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Prepare the document
        prepared = {
            'doc_id': doc_data.get('doc_id'),
            'project_id': doc_data.get('project_id'),
            'project_name': doc_data.get('project_name', ''),
            'file_path': file_path,
            'content': doc_data.get('content', ''),
            'content_type': content_type.value,
            'file_extension': file_ext,
            'language': doc_data.get('language', ''),
            'created_at': doc_data.get('created_at', datetime.now()),
            'updated_at': doc_data.get('updated_at', datetime.now()),
            'indexed_at': datetime.now(),
            'author_name': doc_data.get('author_name', ''),
            'author_email': doc_data.get('author_email', ''),
            'title': doc_data.get('title', ''),
            'headings': doc_data.get('headings', ''),
            'start_line': doc_data.get('start_line', 1),
            'end_line': doc_data.get('end_line', 1),
            'tags': doc_data.get('tags', ''),
            'categories': doc_data.get('categories', ''),
            'file_size': doc_data.get('file_size', 0),
            'word_count': doc_data.get('word_count', 0),
            'commit_sha': doc_data.get('commit_sha', ''),
            'branch': doc_data.get('branch', 'main'),
            'extra_metadata': doc_data.get('extra_metadata', {})
        }
        
        # Add code content for code files
        if content_type == ContentType.CODE:
            prepared['code_content'] = prepared['content']
        
        # Add n-grams for fuzzy search
        prepared['content_ngrams'] = prepared['content']
        
        return prepared


def test_schema():
    """Test schema creation and field access"""
    schema = IndexSchema.get_schema()
    
    print("Schema fields:")
    for field_name in schema.names():
        field = schema[field_name]
        print(f"  {field_name}: {field.__class__.__name__}")
    
    print(f"\nSearchable fields: {IndexSchema.get_searchable_fields()}")
    print(f"Filterable fields: {IndexSchema.get_filterable_fields()}")
    print(f"Sortable fields: {IndexSchema.get_sortable_fields()}")
    
    # Test content type detection
    test_files = [
        "README.md",
        "main.py",
        "config.yaml",
        "index.html",
        "notes.txt"
    ]
    
    print("\nContent type detection:")
    for file_path in test_files:
        content_type = IndexSchema.determine_content_type(file_path)
        print(f"  {file_path} -> {content_type.value}")


if __name__ == "__main__":
    test_schema()