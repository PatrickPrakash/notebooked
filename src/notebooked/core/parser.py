"""Notebook parser to extract tagged cells"""

import json
import ast
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field


@dataclass
class NotebookCell:
    """Represents a notebook cell"""
    cell_type: str
    source: List[str]
    tags: List[str] = field(default_factory=list)
    execution_count: Optional[int] = None
    
    @property
    def code(self) -> str:
        """Get cell source code as string"""
        if isinstance(self.source, list):
            return ''.join(self.source)
        return str(self.source)
    
    def has_tag(self, tag: str) -> bool:
        """Check if cell has a specific tag"""
        return tag in self.tags


@dataclass
class ExtractedCode:
    """Container for extracted code sections"""
    imports: List[str] = field(default_factory=list)
    preprocess: List[str] = field(default_factory=list)
    train: List[str] = field(default_factory=list)
    inference: List[str] = field(default_factory=list)
    utils: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    
    def get_combined_code(self, section: str) -> str:
        """Get combined code for a section"""
        code_list = getattr(self, section, [])
        return '\n\n'.join(code_list)


class NotebookParser:
    """Parse Jupyter notebooks and extract tagged cells"""
    
    # Supported cell tags
    VALID_TAGS = {'imports', 'preprocess', 'train', 'inference', 'utils', 'requirements'}
    
    def __init__(self, notebook_path: str):
        self.notebook_path = Path(notebook_path)
        self.cells: List[NotebookCell] = []
        self._notebook_data: Dict = {}
        
        if not self.notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    def parse(self) -> None:
        """Parse the notebook file"""
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            self._notebook_data = json.load(f)
        
        # Extract cells
        for cell_data in self._notebook_data.get('cells', []):
            cell = self._parse_cell(cell_data)
            if cell:
                self.cells.append(cell)
    
    def _parse_cell(self, cell_data: Dict) -> NotebookCell:
        """Parse a single cell"""
        cell_type = cell_data.get('cell_type', 'code')
        source = cell_data.get('source', [])
        
        # Get tags from metadata
        metadata = cell_data.get('metadata', {})
        tags = metadata.get('tags', [])
        
        execution_count = cell_data.get('execution_count')
        
        return NotebookCell(
            cell_type=cell_type,
            source=source,
            tags=tags,
            execution_count=execution_count
        )
    
    def extract_tagged_code(self) -> ExtractedCode:
        """Extract code from tagged cells"""
        extracted = ExtractedCode()
        
        # Auto-detect imports from first few cells
        self._auto_detect_imports(extracted)
        
        # Extract tagged cells
        for cell in self.cells:
            if cell.cell_type != 'code':
                continue
            
            code = cell.code.strip()
            if not code:
                continue
            
            # Filter magic commands
            code = self._filter_magic_commands(code)
            if not code.strip():
                continue

            # Check for tags
            for tag in cell.tags:
                if tag in self.VALID_TAGS:
                    getattr(extracted, tag).append(code)
        
        return extracted
    
    def _filter_magic_commands(self, code: str) -> str:
        """Remove IPython magic commands (starting with %)"""
        lines = code.split('\n')
        filtered_lines = [line for line in lines if not line.strip().startswith('%') and not line.strip().startswith('!')]
        return '\n'.join(filtered_lines)

    def _auto_detect_imports(self, extracted: ExtractedCode) -> None:
        """Auto-detect import statements from code cells using AST"""
        for cell in self.cells:
            if cell.cell_type != 'code':
                continue
            
            # Skip if cell has explicit tags (we assume tags handle everything)
            if cell.tags:
                continue
            
            code = cell.code.strip()
            if not code:
                continue
                
            # Filter magics before AST parsing
            clean_code = self._filter_magic_commands(code)
            
            try:
                tree = ast.parse(clean_code)
                import_lines = []
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        # Reconstruct source for this node (simple approximation)
                        # In a real scenario, we might want to extract exact lines
                        # For now, we fallback to line matching if AST detects import
                        pass
                
                # Fallback to line matching for simplicity, but now we know it's valid python
                lines = clean_code.split('\n')
                for line in lines:
                    if line.strip().startswith(('import ', 'from ')):
                        import_lines.append(line)
                
                if import_lines:
                    extracted.imports.append('\n'.join(import_lines))
                    
            except SyntaxError:
                # Skip cells that aren't valid Python (e.g. complex magics)
                pass
    
    def get_cells_by_tag(self, tag: str) -> List[NotebookCell]:
        """Get all cells with a specific tag"""
        return [cell for cell in self.cells if cell.has_tag(tag)]
    
    def get_all_tags(self) -> Set[str]:
        """Get all unique tags used in the notebook"""
        tags = set()
        for cell in self.cells:
            tags.update(cell.tags)
        return tags
    
    def validate_tags(self) -> Dict[str, List[str]]:
        """Validate tags and return any issues"""
        issues = {
            'invalid_tags': [],
            'missing_sections': [],
            'warnings': []
        }
        
        all_tags = self.get_all_tags()
        
        # Check for invalid tags
        invalid = all_tags - self.VALID_TAGS
        if invalid:
            issues['invalid_tags'] = list(invalid)
        
        # Check for required sections
        required_tags = {'train'} 
        missing = required_tags - all_tags
        if missing:
            issues['missing_sections'] = list(missing)
        
        # Check if inference is missing (warning only)
        if 'inference' not in all_tags:
            issues['warnings'].append('No inference cells tagged (recommended)')
        
        return issues
    
    def get_statistics(self) -> Dict:
        """Get statistics about the notebook"""
        tag_counts = {}
        for tag in self.VALID_TAGS:
            tag_counts[tag] = len(self.get_cells_by_tag(tag))
        
        return {
            'total_cells': len(self.cells),
            'code_cells': len([c for c in self.cells if c.cell_type == 'code']),
            'markdown_cells': len([c for c in self.cells if c.cell_type == 'markdown']),
            'tag_counts': tag_counts,
            'unique_tags': len(self.get_all_tags())
        }
