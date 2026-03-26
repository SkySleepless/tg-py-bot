"""
System prompt loader for AI models.
Loads system prompts from AGENTS.md file.
"""

import re
from typing import Dict, Optional
from pathlib import Path


class SystemPromptLoader:
    """Loads and manages system prompts from AGENTS.md file."""
    
    def __init__(self, agents_file_path: str = "skills/AGENTS.md"):
        """
        Initialize the system prompt loader.
        
        Args:
            agents_file_path: Path to the AGENTS.md file
        """
        self.agents_file_path = agents_file_path
        self._system_prompt_cache = None
    
    def load_system_prompt(self) -> str:
        """
        Load the system prompt from AGENTS.md file.
        
        Returns:
            System prompt string extracted from AGENTS.md
        """
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache
            
        try:
            agents_path = Path(self.agents_file_path)
            if not agents_path.exists():
                raise FileNotFoundError(f"AGENTS.md file not found at {self.agents_file_path}")
            
            content = agents_path.read_text()
            system_prompt = self._extract_system_prompt(content)
            self._system_prompt_cache = system_prompt
            return system_prompt
            
        except Exception as e:
            # Fallback to a default system prompt if loading fails
            return self._get_default_system_prompt()
    
    def _extract_system_prompt(self, content: str) -> str:
        """
        Extract system prompt from AGENTS.md content.
        
        Args:
            content: Content of AGENTS.md file
            
        Returns:
            Extracted system prompt
        """
        # Extract the core objective and workflow
        lines = content.split('\n')
        system_prompt_parts = []
        
        # Look for key sections
        in_core_objective = False
        in_workflow = False
        in_roles = False
        
        for line in lines:
            # Extract core objective
            if 'Core Objective' in line and '##' in line:
                in_core_objective = True
                continue
            elif in_core_objective and line.startswith('---'):
                in_core_objective = False
                continue
            elif in_core_objective and line.strip() and not line.startswith('|') and not line.startswith('-'):
                # Remove markdown formatting and add to parts (skip table-like lines)
                clean_line = line.strip().replace('**', '').replace('*', '')
                if clean_line:  # Only add non-empty lines
                    system_prompt_parts.append(clean_line)
                
            # Extract workflow stages - only extract the stage headers, not the detailed content
            if 'Workflow Stages' in line and '##' in line:
                in_workflow = True
                continue
            elif in_workflow and line.startswith('###'):
                # Extract workflow stage headers only
                stage_match = re.match(r'###\s+(\d+)\..*', line)
                if stage_match:
                    clean_line = line.strip().replace('**', '').replace('*', '')
                    system_prompt_parts.append(f"\n{clean_line}")
                continue
            elif in_workflow and line.startswith('---'):
                # End of workflow stages section
                in_workflow = False
                continue
                
            # Extract roles and responsibilities
            if 'Roles & Responsibilities' in line and '##' in line:
                in_roles = True
                continue
            elif in_roles and ('| Role' in line or '|----' in line or '|-' in line):
                # Skip table header and separators
                continue
            elif in_roles and line.startswith('|') and '|' in line:
                # Extract role information from table
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 3:  # Should have at least Role, Focus Area, and Output Format
                    role = parts[0]
                    focus = parts[1]
                    # Clean up any remaining markdown
                    role = role.replace('**', '').replace('*', '')
                    focus = focus.replace('**', '').replace('*', '')
                    if role and focus:  # Only add if both role and focus are non-empty
                        system_prompt_parts.append(f"{role}: {focus}")
            elif in_roles and line.startswith('---'):
                # End of roles section
                in_roles = False
                continue
        
        if not system_prompt_parts:
            return self._get_default_system_prompt()
            
        # Create a comprehensive system prompt
        system_prompt = "Collaborative Reasoning Protocol System Prompt:\n\n" 
        system_prompt += "Core Objective: Deliver transparent, bias-checked, and actionable solutions through multi-role recursive dialogue.\n\n"
        system_prompt += "Roles and Responsibilities:\n"
        
        # Add roles information
        for part in system_prompt_parts:
            if ':' in part and not part.startswith('##') and not part.startswith('###'):
                system_prompt += f"- {part}\n"
        
        system_prompt += "\nWorkflow Stages:\n"
        
        # Add workflow information  
        for part in system_prompt_parts:
            if part.startswith('###'):
                system_prompt += f"\n{part}\n"
        
        system_prompt += "\nExecution Rules:\n"
        system_prompt += "- Use Markdown for all content with role contributions separated by ---\n"
        system_prompt += "- Minimum 3 unique roles per stage\n"
        system_prompt += "- Coordinator can restart workflow if critical errors detected\n"
        system_prompt += "- Ensure outputs are clear, labeled, and actionable\n"
        
        return system_prompt
    
    def _get_default_system_prompt(self) -> str:
        """
        Get a default system prompt if AGENTS.md cannot be loaded.
        
        Returns:
            Default system prompt
        """
        return """You are a helpful AI assistant. Provide clear, concise, and actionable responses.
Use structured thinking and consider multiple perspectives when solving problems.
Ensure your responses are transparent and address potential biases."""
    
    def clear_cache(self):
        """Clear the cached system prompt."""
        self._system_prompt_cache = None


# Global instance for easy access
def get_system_prompt() -> str:
    """
    Get the system prompt from AGENTS.md.
    
    Returns:
        System prompt string
    """
    loader = SystemPromptLoader()
    return loader.load_system_prompt()